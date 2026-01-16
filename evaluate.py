#!/usr/bin/env python3
"""
LIBERO Model Evaluation Script

This script provides comprehensive evaluation for trained H-AIF models on LIBERO benchmark.
Supports two evaluation modes:
1. Offline evaluation: Action prediction metrics on validation dataset
2. Online evaluation: Success rate in LIBERO simulation environment (requires libero)

Usage:
    python evaluate.py --config config_libero.yaml --checkpoint results/model_best.pth.tar
    uv run evaluate.py --config config_libero.yaml --checkpoint results/model_best.pth.tar --mode online --num-episodes 20
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from accelerate.utils import send_to_device

# Project imports
from model.models import ActNet
from script.load_libero_dataset import (
    load_libero_dataloader,
    LiberoNormalizer,
)


# =============================================================================
# Metrics Classes
# =============================================================================


class ActionMetrics:
    """Metrics tracker for action prediction evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.count = 0

        # Per-dimension metrics
        self.mse_per_dim = np.zeros(7)
        self.mae_per_dim = np.zeros(7)

        # Gripper metrics (binary classification)
        self.gripper_correct = 0
        self.gripper_total = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch of predictions.

        Args:
            pred: Predicted actions [B, T, 7] or [B, 7]
            target: Target actions [B, T, 7] or [B, 7]
        """
        # Ensure 3D tensors
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        # Match temporal dimensions
        min_t = min(pred.shape[1], target.shape[1])
        pred = pred[:, :min_t, :]
        target = target[:, :min_t, :]

        # Compute MSE and MAE
        diff = (pred - target).detach().cpu().numpy()
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))

        # Per-dimension metrics (average over batch and time)
        mse_dim = np.mean(diff**2, axis=(0, 1))
        mae_dim = np.mean(np.abs(diff), axis=(0, 1))

        batch_size = pred.shape[0] * pred.shape[1]

        self.mse_sum += mse * batch_size
        self.mae_sum += mae * batch_size
        self.mse_per_dim += mse_dim * batch_size
        self.mae_per_dim += mae_dim * batch_size
        self.count += batch_size

        # Gripper classification (last dimension)
        gripper_pred = (pred[:, :, -1] > 0).float()
        gripper_target = (target[:, :, -1] > 0).float()
        self.gripper_correct += (gripper_pred == gripper_target).sum().item()
        self.gripper_total += gripper_pred.numel()

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if self.count == 0:
            return {}

        results = {
            "mse": self.mse_sum / self.count,
            "mae": self.mae_sum / self.count,
            "rmse": np.sqrt(self.mse_sum / self.count),
            "gripper_accuracy": self.gripper_correct / max(self.gripper_total, 1),
        }

        # Per-dimension metrics
        mse_dim = self.mse_per_dim / self.count
        mae_dim = self.mae_per_dim / self.count

        dim_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
        for i, name in enumerate(dim_names):
            results[f"mse_{name}"] = float(mse_dim[i])
            results[f"mae_{name}"] = float(mae_dim[i])

        return results


class EpisodeMetrics:
    """Metrics tracker for online evaluation."""

    def __init__(self):
        self.episodes = []

    def add_episode(
        self, success: bool, reward: float, steps: int, task_name: str = ""
    ):
        self.episodes.append(
            {
                "success": success,
                "reward": reward,
                "steps": steps,
                "task_name": task_name,
            }
        )

    def compute(self) -> Dict[str, Any]:
        if len(self.episodes) == 0:
            return {}

        successes = [e["success"] for e in self.episodes]
        rewards = [e["reward"] for e in self.episodes]
        steps = [e["steps"] for e in self.episodes]

        # Overall metrics
        results = {
            "success_rate": np.mean(successes),
            "avg_reward": np.mean(rewards),
            "avg_steps": np.mean(steps),
            "num_episodes": len(self.episodes),
        }

        # Per-task metrics
        task_names = set(e["task_name"] for e in self.episodes if e["task_name"])
        if task_names:
            results["per_task"] = {}
            for task in task_names:
                task_eps = [e for e in self.episodes if e["task_name"] == task]
                results["per_task"][task] = {
                    "success_rate": np.mean([e["success"] for e in task_eps]),
                    "avg_reward": np.mean([e["reward"] for e in task_eps]),
                    "avg_steps": np.mean([e["steps"] for e in task_eps]),
                    "num_episodes": len(task_eps),
                }

        return results


# =============================================================================
# Model Loading
# =============================================================================


def load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ActNet, DictConfig]:
    """
    Load trained model from checkpoint.

    Args:
        config_path: Path to config yaml file
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    print(f"[INFO] Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    print("[INFO] Creating model...")
    model = ActNet(config, is_use_cuda=device.type == "cuda", device=device)
    model = model.to(device)

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle state dict (may have "module." prefix from DDP training)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Remove "module." prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Filter out keys with shape mismatch
    model_state = model.state_dict()
    keys_to_remove = []
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                print(
                    f"[WARNING] Shape mismatch for {k}: checkpoint {v.shape} != model {model_state[k].shape}. Ignoring."
                )
                keys_to_remove.append(k)

    for k in keys_to_remove:
        del state_dict[k]

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[WARNING] Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys: {unexpected_keys}")

    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model loaded successfully. Total parameters: {total_params:,}")

    return model, config


# =============================================================================
# Offline Evaluation
# =============================================================================


@torch.no_grad()
def evaluate_offline(
    model: ActNet,
    dataloader: torch.utils.data.DataLoader,
    config: DictConfig,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on validation dataset (offline evaluation).

    Args:
        model: Trained model
        dataloader: Validation dataloader
        config: Model config
        device: Device for inference
        max_batches: Maximum number of batches to evaluate (None for all)

    Returns:
        Dict with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("  Offline Evaluation (Dataset-based)")
    print("=" * 60)

    model.eval()
    metrics = ActionMetrics()

    total_batches = len(dataloader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    pbar = tqdm(enumerate(dataloader), total=total_batches, desc="Evaluating")

    inference_times = []

    for batch_idx, batch in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Skip incomplete batches
        if isinstance(batch, dict) and batch["action"].size(0) < config.batchsize:
            continue

        # Move batch to device
        batch = send_to_device(batch, device)

        # Forward pass with timing
        start_time = time.time()
        output = model(batch, phase="eval")
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Extract predictions and targets
        pred_actions = output.actions
        target_actions = batch["action"]

        # Match temporal dimensions (model may predict different horizon)
        horizon = target_actions.shape[1]
        if pred_actions.shape[1] != horizon:
            pred_actions = pred_actions[:, :horizon, :]

        # Update metrics
        metrics.update(pred_actions, target_actions)

        # Update progress bar
        current_mse = metrics.mse_sum / max(metrics.count, 1)
        pbar.set_postfix(mse=f"{current_mse:.6f}", time=f"{inference_time:.4f}s")

    # Compute final metrics
    results = metrics.compute()

    # Add timing info
    if inference_times:
        results["avg_inference_time_ms"] = np.mean(inference_times) * 1000
        results["std_inference_time_ms"] = np.std(inference_times) * 1000

    results["num_samples"] = metrics.count
    results["num_batches"] = min(batch_idx + 1, total_batches)

    # Print results
    print("\n" + "-" * 40)
    print("Offline Evaluation Results:")
    print("-" * 40)
    print(f"  MSE:              {results['mse']:.6f}")
    print(f"  MAE:              {results['mae']:.6f}")
    print(f"  RMSE:             {results['rmse']:.6f}")
    print(f"  Gripper Accuracy: {results['gripper_accuracy']:.4f}")
    print(f"  Avg Inference:    {results.get('avg_inference_time_ms', 0):.2f} ms")
    print(f"  Num Samples:      {results['num_samples']}")
    print("-" * 40)

    return results


# =============================================================================
# Online Evaluation (Environment-based)
# =============================================================================


def evaluate_online(
    model: ActNet,
    config: DictConfig,
    device: torch.device,
    num_episodes: int = 20,
    max_steps: int = 500,
    normalizer: Optional[LiberoNormalizer] = None,
    save_videos: bool = False,
    output_dir: Optional[str] = None,
    use_robosuite_only: bool = False,
    robosuite_task: str = "Lift",
    render: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model in simulation environment.

    Supports two modes:
    1. LIBERO mode: Evaluate on LIBERO benchmark tasks (requires libero package)
    2. Robosuite mode: Evaluate on basic robosuite tasks (only requires robosuite)

    Args:
        model: Trained model
        config: Model config
        device: Device for inference
        num_episodes: Number of episodes per task
        max_steps: Maximum steps per episode
        normalizer: Normalizer for action denormalization
        save_videos: Whether to save evaluation videos
        output_dir: Directory to save videos
        use_robosuite_only: If True, use basic robosuite instead of LIBERO
        robosuite_task: Robosuite task name (when use_robosuite_only=True)
        render: If True, render the environment in real-time GUI window

    Returns:
        Dict with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("  Online Evaluation (Environment-based)")
    print("=" * 60)

    # Check available simulation backends
    HAS_LIBERO = False
    HAS_ROBOSUITE = False

    try:
        import robosuite as suite

        HAS_ROBOSUITE = True
        print(f"[INFO] robosuite available (v{suite.__version__})")
    except ImportError:
        print("[WARNING] robosuite not installed")

    if not use_robosuite_only:
        try:
            from libero.libero import benchmark

            HAS_LIBERO = True
            print("[INFO] LIBERO available")
        except ImportError:
            print("[WARNING] LIBERO not installed, falling back to robosuite-only mode")
            use_robosuite_only = True

    if not HAS_ROBOSUITE and not HAS_LIBERO:
        print("[ERROR] Neither robosuite nor LIBERO is available")
        print("[INFO] Install with: pip install robosuite")
        print(
            "[INFO] Or install LIBERO from: https://github.com/Lifelong-Robot-Learning/LIBERO"
        )
        return {"error": "No simulation backend available"}

    # Import environment wrapper
    try:
        from envs.robosuite_env import RobosuiteEnvWrapper, PolicyEvaluator
    except ImportError as e:
        print(f"[ERROR] Cannot import environment wrapper: {e}")
        return {"error": str(e)}

    model.eval()

    # Prepare norm_stats for denormalization
    norm_stats = None
    if normalizer is not None:
        norm_stats = normalizer.norm_stats

    metrics = EpisodeMetrics()

    if use_robosuite_only or not HAS_LIBERO:
        # ============ Robosuite-only mode ============
        print("\n[MODE] Robosuite-only evaluation")
        print(f"[INFO] Task: {robosuite_task}")

        # Generate language instruction
        language_instruction = robosuite_task.replace("_", " ").lower()
        if "lift" in language_instruction:
            language_instruction = "lift the cube"
        print(f"[INFO] Language instruction: {language_instruction}")

        print(f"[INFO] Episodes: {num_episodes}")

        try:
            # Create robosuite environment directly
            # Enable GUI rendering if requested
            env = suite.make(
                env_name=robosuite_task,
                robots="Panda",
                has_renderer=render,  # Enable on-screen rendering for GUI
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=config.cropWidth,
                camera_widths=config.cropWidth,
                control_freq=20,
            )

            if render:
                print("[INFO] GUI rendering enabled - press ESC to close window")
            print("[INFO] Environment created: " + robosuite_task)

            # Run evaluation episodes
            for ep in range(num_episodes):
                result = _evaluate_robosuite_episode(
                    model=model,
                    env=env,
                    config=config,
                    device=device,
                    max_steps=max_steps,
                    norm_stats=norm_stats,
                    render=render,
                    language_instruction=language_instruction,
                )
                metrics.add_episode(
                    success=result["success"],
                    reward=result["total_reward"],
                    steps=result["steps"],
                    task_name=robosuite_task,
                )
                print(
                    f"  Episode {ep + 1}/{num_episodes}: "
                    f"success={result['success']}, "
                    f"reward={result['total_reward']:.2f}, "
                    f"steps={result['steps']}"
                )

            env.close()

        except Exception as e:
            print(f"[ERROR] Failed: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}

    else:
        # ============ LIBERO mode ============
        from libero.libero import benchmark

        task_suite = getattr(config, "task_suite", "libero_10")
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_obj = benchmark_dict[task_suite]()
        task_names = task_suite_obj.get_task_names()

        print("\n[MODE] LIBERO benchmark evaluation")
        print(f"[INFO] Task suite: {task_suite}")
        print(f"[INFO] Number of tasks: {len(task_names)}")
        print(f"[INFO] Episodes per task: {num_episodes}")

        # Evaluate each task
        for task_idx, task_name in enumerate(task_names):
            print(f"\n[Task {task_idx + 1}/{len(task_names)}] {task_name}")

            try:
                # Create environment
                env = RobosuiteEnvWrapper(
                    task_name=task_name,
                    task_suite=task_suite,
                    image_size=config.cropWidth,
                    norm_stats=norm_stats,
                    device=str(device),
                    has_renderer=render,
                )

                # Create evaluator
                evaluator = PolicyEvaluator(
                    model=model,
                    env=env,
                    device=str(device),
                    max_episode_steps=max_steps,
                )

                # Run episodes
                for ep in range(num_episodes):
                    result = evaluator.evaluate_episode()
                    metrics.add_episode(
                        success=result["success"],
                        reward=result["total_reward"],
                        steps=result["steps"],
                        task_name=task_name,
                    )
                    print(
                        f"  Episode {ep + 1}/{num_episodes}: "
                        f"success={result['success']}, "
                        f"reward={result['total_reward']:.2f}, "
                        f"steps={result['steps']}"
                    )

                env.close()

            except Exception as e:
                print(f"[ERROR] Failed on task {task_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    # Compute final metrics
    results = metrics.compute()

    # Print results
    print("\n" + "-" * 40)
    print("Online Evaluation Results:")
    print("-" * 40)
    print(f"  Success Rate:    {results.get('success_rate', 0):.4f}")
    print(f"  Avg Reward:      {results.get('avg_reward', 0):.4f}")
    print(f"  Avg Steps:       {results.get('avg_steps', 0):.2f}")
    print(f"  Num Episodes:    {results.get('num_episodes', 0)}")

    if "per_task" in results:
        print("\nPer-Task Results:")
        for task, task_results in results["per_task"].items():
            print(f"  {task}:")
            print(f"    Success Rate: {task_results['success_rate']:.4f}")
    print("-" * 40)

    return results


@torch.no_grad()
def _evaluate_robosuite_episode(
    model: ActNet,
    env,
    config: DictConfig,
    device: torch.device,
    max_steps: int = 500,
    norm_stats: Optional[Dict] = None,
    render: bool = False,
    language_instruction: str = "manipulate the object",
) -> Dict[str, Any]:
    """
    Evaluate one episode in robosuite environment.

    Args:
        model: Trained model
        env: Robosuite environment
        config: Model config
        device: Device for inference
        max_steps: Maximum steps per episode
        norm_stats: Normalization statistics
        render: Whether to render the environment

    Returns:
        Dict with episode statistics
    """
    obs = env.reset()

    total_reward = 0.0
    success = False
    steps = 0

    # Image history for temporal context
    image_history = {"agentview": [], "eye_in_hand": []}

    for step in range(max_steps):
        # Process observation
        batch = _prepare_robosuite_batch(
            obs, image_history, config, device, language_instruction
        )

        # Model inference
        output = model(batch, phase="eval")

        # Get action (first action from predicted sequence)
        action = output.actions[0, 0].detach().cpu().numpy()

        # Denormalize action if needed
        if norm_stats is not None and "action" in norm_stats:
            mean = np.array(norm_stats["action"]["mean"])
            std = np.array(norm_stats["action"]["std"])
            action = action * std + mean

        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Execute action
        obs, reward, done, info = env.step(action)

        total_reward += reward
        steps += 1

        # Render if enabled
        if render:
            env.render()

        # Update image history
        _update_image_history(obs, image_history, config)

        # Check termination
        if done:
            success = info.get("success", reward > 0)
            break

    return {
        "total_reward": total_reward,
        "success": success,
        "steps": steps,
    }


def _prepare_robosuite_batch(
    obs: Dict,
    image_history: Dict,
    config: DictConfig,
    device: torch.device,
    language_instruction: str = "manipulate the object",
) -> Dict:
    """Prepare batch input for model from robosuite observation."""

    # Process images
    def process_image(img):
        """Convert image to tensor [C, H, W]."""
        img = torch.from_numpy(img).float()
        if img.max() > 1.0:
            img = img / 255.0
        img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        return img

    # Get current images
    agentview_img = process_image(
        obs.get("agentview_image", np.zeros((config.cropWidth, config.cropWidth, 3)))
    )
    eye_in_hand_img = process_image(
        obs.get(
            "robot0_eye_in_hand_image",
            np.zeros((config.cropWidth, config.cropWidth, 3)),
        )
    )

    # Get robot state (7D to match model's action_dim)
    # Use: eef_pos (3D) + eef_quat first 3 components (3D) + gripper (1D) = 7D
    # This matches the model's expected state dimension (action_dim=7)
    if "robot0_eef_pos" in obs:
        eef_pos = torch.from_numpy(obs["robot0_eef_pos"]).float()  # 3D
        eef_quat = torch.from_numpy(
            obs.get("robot0_eef_quat", np.zeros(4))
        ).float()  # 4D
        gripper = torch.from_numpy(obs.get("robot0_gripper_qpos", np.zeros(2))).float()
        # State: [eef_pos (3), eef_quat[:3] (3), gripper (1)] = 7D
        state = torch.cat([eef_pos, eef_quat[:3], gripper[:1]], dim=-1)
    else:
        state = torch.zeros(7)  # 7D to match model action_dim

    # Build image sequences
    # Model expects: [B, T, C, H, W] where T = past_img_num + future_img_num + 1
    past_img_num = getattr(config, "past_img_num", 5)
    future_img_num = getattr(config, "future_img_num", 5)

    # Pad history if needed
    while len(image_history["agentview"]) < past_img_num:
        image_history["agentview"].insert(0, agentview_img.clone())
        image_history["eye_in_hand"].insert(0, eye_in_hand_img.clone())

    # Create sequences: [past images] + [current] + [future images (repeat current for eval)]
    # Past images from history
    past_agentview = image_history["agentview"][-past_img_num:]
    past_eye_in_hand = image_history["eye_in_hand"][-past_img_num:]

    # Current image
    current_agentview = [agentview_img]
    current_eye_in_hand = [eye_in_hand_img]

    # Future images: repeat current frame (since we don't have future during inference)
    future_agentview = [agentview_img.clone() for _ in range(future_img_num)]
    future_eye_in_hand = [eye_in_hand_img.clone() for _ in range(future_img_num)]

    # Combine: [past] + [current] + [future]
    agentview_seq = torch.stack(
        past_agentview + current_agentview + future_agentview
    )  # [T, C, H, W]
    eye_in_hand_seq = torch.stack(
        past_eye_in_hand + current_eye_in_hand + future_eye_in_hand
    )  # [T, C, H, W]

    # Build batch (add batch dimension) -> [B, T, C, H, W]
    batch = {
        "observation": {
            "agentview_image": agentview_img.unsqueeze(0).to(device),
            "agentview_image_seq": agentview_seq.unsqueeze(0).to(
                device
            ),  # [1, T, C, H, W]
            "eye_in_hand_image": eye_in_hand_img.unsqueeze(0).to(device),
            "eye_in_hand_image_seq": eye_in_hand_seq.unsqueeze(0).to(
                device
            ),  # [1, T, C, H, W]
            "top_image": agentview_img.unsqueeze(0).to(device),
            "top_image_seq": agentview_seq.unsqueeze(0).to(device),  # [1, T, C, H, W]
            "wrist_image": eye_in_hand_img.unsqueeze(0).to(device),
            "wrist_image_seq": eye_in_hand_seq.unsqueeze(0).to(
                device
            ),  # [1, T, C, H, W]
            "state": state.unsqueeze(0).to(device),
        },
        "language": [language_instruction],
        "action": torch.zeros(1, config.horizon, config.action_dim).to(
            device
        ),  # Placeholder
    }

    return batch


def _update_image_history(obs: Dict, image_history: Dict, config: DictConfig):
    """Update image history buffer."""

    def process_image(img):
        img = torch.from_numpy(img).float()
        if img.max() > 1.0:
            img = img / 255.0
        img = img.permute(2, 0, 1)
        return img

    if "agentview_image" in obs:
        image_history["agentview"].append(process_image(obs["agentview_image"]))
        if len(image_history["agentview"]) > 10:
            image_history["agentview"].pop(0)

    if "robot0_eye_in_hand_image" in obs:
        image_history["eye_in_hand"].append(
            process_image(obs["robot0_eye_in_hand_image"])
        )
        if len(image_history["eye_in_hand"]) > 10:
            image_history["eye_in_hand"].pop(0)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained H-AIF model on LIBERO benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to config yaml file",
    )
    parser.add_argument(
        "--checkpoint",
        "-p",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Evaluation mode
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["offline", "online", "both"],
        default="offline",
        help="Evaluation mode: offline (dataset), online (environment), or both",
    )

    # Offline evaluation options
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches for offline evaluation (None for all)",
    )

    # Online evaluation options
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes per task for online evaluation",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--robosuite-only",
        action="store_true",
        help="Use basic robosuite environment instead of LIBERO (no libero required)",
    )
    parser.add_argument(
        "--robosuite-task",
        type=str,
        default="Lift",
        choices=[
            "Lift",
            "Stack",
            "NutAssembly",
            "NutAssemblySquare",
            "NutAssemblyRound",
            "PickPlace",
            "PickPlaceSingle",
            "PickPlaceMilk",
            "PickPlaceBread",
            "PickPlaceCereal",
            "Door",
            "Wipe",
            "ToolHang",
            "TwoArmLift",
            "TwoArmPegInHole",
            "TwoArmHandover",
        ],
        help="Robosuite task to evaluate (when --robosuite-only is set)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable real-time GUI rendering (robosuite-only mode)",
    )

    # Dataset options
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=None,
        help="Override dataset path from config (required if config path doesn't exist)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for results (default: same as checkpoint)",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save evaluation videos (online mode only)",
    )

    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Setup output directory
    if args.output is None:
        args.output = str(Path(args.checkpoint).parent)
    os.makedirs(args.output, exist_ok=True)

    # Load model
    model, config = load_model(args.config, args.checkpoint, device)

    # Results container
    all_results = {
        "config_path": args.config,
        "checkpoint_path": args.checkpoint,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }

    # Load dataloader/normalizer if needed (for both offline and online)
    normalizer = None
    if args.mode in ["offline", "both"] or (
        args.mode == "online" and not args.robosuite_only
    ):
        print("\n[INFO] Loading dataset (for validation or normalization stats)...")

        task_suite = getattr(config, "task_suite", "libero_10")

        # Use command line --data-dir if provided, otherwise use config
        local_dir = args.data_dir
        if local_dir is None:
            local_dir = (
                config.datasets_path if hasattr(config, "datasets_path") else None
            )

        if local_dir is None:
            print(
                "[ERROR] Dataset path not specified. Use --data-dir or set datasets_path in config."
            )
            sys.exit(1)

        # Check if dataset exists
        local_dir_path = Path(local_dir)
        if not local_dir_path.exists():
            print(f"[ERROR] Dataset path does not exist: {local_dir}")
            print("[INFO] Please specify a valid path with --data-dir")
            sys.exit(1)

        _, val_loader, normalizer = load_libero_dataloader(
            task_suite=task_suite,
            local_dir=local_dir,
            batch_size=config.batchsize,
            num_workers=config.num_workers,
            horizon=config.horizon,
            past_img_num=config.past_img_num,
            future_img_num=config.future_img_num,
            image_size=config.cropWidth,
            train_ratio=1.0 - config.valid_datas_scale,
            normalize=True,
            seed=config.random_seed,
        )

        print(f"[INFO] Validation dataset size: {len(val_loader.dataset)} samples")

        # Run offline evaluation
        if args.mode in ["offline", "both"]:
            offline_results = evaluate_offline(
                model=model,
                dataloader=val_loader,
                config=config,
                device=device,
                max_batches=args.max_batches,
            )
            all_results["offline"] = offline_results

    # Run online evaluation
    if args.mode in ["online", "both"]:
        online_results = evaluate_online(
            model=model,
            config=config,
            device=device,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            normalizer=normalizer,
            save_videos=args.save_videos,
            output_dir=args.output,
            use_robosuite_only=args.robosuite_only,
            robosuite_task=args.robosuite_task,
            render=args.render,
        )
        all_results["online"] = online_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(args.output) / f"eval_results_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[INFO] Results saved to: {results_path}")
    print("\n" + "=" * 60)
    print("  Evaluation Complete!")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    main()

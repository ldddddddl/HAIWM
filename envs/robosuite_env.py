"""
Robosuite Environment Wrapper for H-AIF

This module provides a wrapper for robosuite environments
to evaluate trained policies on LIBERO tasks.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

# Try to import robosuite
try:
    import robosuite as suite
    from robosuite.wrappers import GymWrapper

    HAS_ROBOSUITE = True
except ImportError:
    HAS_ROBOSUITE = False
    print("[WARNING] robosuite not installed. Install with: pip install robosuite")

# Try to import libero
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    HAS_LIBERO = True
except ImportError:
    HAS_LIBERO = False
    print(
        "[WARNING] libero not installed. Install from: https://github.com/Lifelong-Robot-Learning/LIBERO"
    )


class RobosuiteEnvWrapper:
    """
    Wrapper for robosuite/LIBERO environments for policy evaluation.

    This wrapper provides a consistent interface for:
    - Environment creation and reset
    - Observation preprocessing
    - Action denormalization and execution
    - Episode management
    """

    def __init__(
        self,
        task_name: str = "KITCHEN_SCENE1_open_the_door",
        task_suite: str = "libero_10",
        image_size: int = 112,
        norm_stats: Optional[Dict] = None,
        device: str = "cuda",
        render_mode: str = "rgb_array",
        use_camera: List[str] = ["agentview", "robot0_eye_in_hand"],
        has_renderer: bool = False,
    ):
        """
        Args:
            task_name: Name of the task to load
            task_suite: LIBERO task suite name
            image_size: Target image size for observations
            norm_stats: Normalization statistics for actions
            device: Device for tensor operations
            render_mode: Render mode for environment
            use_camera: List of camera names to use
            has_renderer: Whether to enable on-screen rendering
        """
        self.task_name = task_name
        self.task_suite = task_suite
        self.image_size = image_size
        self.norm_stats = norm_stats
        self.device = device
        self.render_mode = render_mode
        self.use_camera = use_camera
        self.has_renderer = has_renderer

        self.env = None
        self._setup_environment()

    def _setup_environment(self):
        """Setup the robosuite/LIBERO environment."""
        if HAS_LIBERO:
            self._setup_libero_env()
        elif HAS_ROBOSUITE:
            self._setup_robosuite_env()
        else:
            raise ImportError("Neither libero nor robosuite is available.")

    def _setup_libero_env(self):
        """Setup LIBERO environment."""
        # Get benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_obj = benchmark_dict[self.task_suite]()

        # Find task
        task = None
        for t in task_suite_obj.get_task_names():
            if t == self.task_name or self.task_name in t:
                task = t
                break

        if task is None:
            available = task_suite_obj.get_task_names()
            raise ValueError(
                f"Task '{self.task_name}' not found. Available: {available}"
            )

        # Get task info
        task_idx = task_suite_obj.get_task_names().index(task)
        task_bddl_file = task_suite_obj.get_task_bddl_file_path(task_idx)

        # Create environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.image_size,
            "camera_widths": self.image_size,
        }

        self.env = OffScreenRenderEnv(**env_args)
        self.task_description = task.replace("_", " ")
        print(f"[RobosuiteEnv] Loaded LIBERO task: {task}")
        print(f"[RobosuiteEnv] Task description: {self.task_description}")

    def _setup_robosuite_env(self):
        """Setup basic robosuite environment."""
        # Create a basic manipulation environment
        self.env = suite.make(
            env_name="Lift",  # Default task
            robots="Panda",
            has_renderer=self.has_renderer,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.use_camera,
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            control_freq=20,
        )
        self.task_description = "lift the cube"
        print("[RobosuiteEnv] Loaded robosuite Lift task")

    def render(self):
        """Render the environment if renderer is enabled."""
        if self.has_renderer and self.env is not None:
            self.env.render()

    def reset(self) -> Dict[str, torch.Tensor]:
        """
        Reset the environment and return initial observation.

        Returns:
            Dict containing observation tensors
        """
        obs = self.env.reset()
        return self._process_observation(obs)

    def _process_observation(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """
        Process raw observation into model input format.

        Args:
            obs: Raw observation from environment

        Returns:
            Processed observation dict
        """
        processed = {
            "observation": {},
            "language": self.task_description,
        }

        # Process images
        for camera in self.use_camera:
            img_key = f"{camera}_image"
            if img_key in obs:
                img = obs[img_key]
                # Convert to tensor [C, H, W]
                img = torch.from_numpy(img).float()
                if img.max() > 1.0:
                    img = img / 255.0
                img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

                if camera == "agentview":
                    processed["observation"]["agentview_image"] = img
                    processed["observation"]["top_image"] = img
                elif "eye_in_hand" in camera:
                    processed["observation"]["eye_in_hand_image"] = img
                    processed["observation"]["wrist_image"] = img

        # Process robot state
        if "robot0_eef_pos" in obs:
            eef_pos = torch.from_numpy(obs["robot0_eef_pos"]).float()
            eef_quat = torch.from_numpy(obs.get("robot0_eef_quat", np.zeros(4))).float()
            gripper = torch.from_numpy(
                obs.get("robot0_gripper_qpos", np.zeros(2))
            ).float()
            state = torch.cat([eef_pos, eef_quat, gripper[:1]], dim=-1)
        else:
            state = torch.zeros(7)

        processed["observation"]["state"] = state

        return processed

    def step(
        self,
        action: torch.Tensor,
        denormalize: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action tensor [7] (6D pose + 1D gripper)
            denormalize: Whether to denormalize action before execution

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # Denormalize if needed
        if denormalize and self.norm_stats is not None:
            action = self._denormalize_action(action)

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Execute action
        obs, reward, done, info = self.env.step(action)

        # Process observation
        processed_obs = self._process_observation(obs)

        return processed_obs, reward, done, False, info

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action using stored statistics."""
        if "action" not in self.norm_stats:
            return action

        stats = self.norm_stats["action"]
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])

        return action * std + mean

    def render(self) -> np.ndarray:
        """Render the environment."""
        if self.has_renderer and hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()

    @property
    def action_dim(self) -> int:
        """Return action dimension (7 for LIBERO)."""
        return 7

    @property
    def observation_dim(self) -> int:
        """Return state observation dimension."""
        return 7


class PolicyEvaluator:
    """
    Evaluator for running trained policies on robosuite/LIBERO environments.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        env: RobosuiteEnvWrapper,
        device: str = "cuda",
        max_episode_steps: int = 500,
    ):
        """
        Args:
            model: Trained policy model
            env: RobosuiteEnvWrapper instance
            device: Device for model inference
            max_episode_steps: Maximum steps per episode
        """
        self.model = model
        self.env = env
        self.device = device
        self.max_episode_steps = max_episode_steps

        self.model.eval()

    @torch.no_grad()
    def evaluate_episode(self) -> Dict[str, Any]:
        """
        Evaluate one episode.

        Returns:
            Dict with episode statistics
        """
        obs = self.env.reset()

        total_reward = 0.0
        success = False
        steps = 0

        # History buffers (dict for top and wrist images)
        image_history = {"top": [], "wrist": []}

        for step in range(self.max_episode_steps):
            # Prepare batch input
            batch = self._prepare_batch(obs, image_history)

            # Move all tensors to device (handle nested dicts)
            def move_to_device(obj, device):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v, device) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [move_to_device(v, device) for v in obj]
                return obj

            batch = move_to_device(batch, self.device)

            output = self.model(batch, phase="eval")

            # Get action (use first action from predicted sequence)
            action = output.actions[0, 0]  # [action_dim]

            # Execute action
            obs, reward, done, truncated, info = self.env.step(action)
            self.env.render()

            total_reward += reward
            steps += 1

            # Update history
            self._update_history(obs, image_history)

            # Check termination
            if done or truncated:
                success = info.get("success", reward > 0)
                break

        return {
            "total_reward": total_reward,
            "success": success,
            "steps": steps,
        }

    def _prepare_batch(
        self,
        obs: Dict,
        image_history: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch input for model.

        Creates batch with:
        - observation: dict with images, image sequences, and state
        - language: task description
        - action: placeholder for model compatibility
        """
        # Get config values (defaults for LIBERO)
        past_img_num = 5
        future_img_num = 5
        action_dim = 7
        horizon = 10

        # Get current images
        top_image = obs["observation"].get(
            "top_image", obs["observation"].get("agentview_image")
        )
        wrist_image = obs["observation"].get(
            "wrist_image", obs["observation"].get("eye_in_hand_image")
        )
        state = obs["observation"].get("state", torch.zeros(7))

        # Ensure state is 7D to match model's action_dim
        # Original state is 8D: eef_pos(3) + eef_quat(4) + gripper(1) = 8
        # Model expects 7D: truncate quaternion to 3 components
        if isinstance(state, torch.Tensor) and state.shape[-1] == 8:
            # Use: pos(3) + quat[:3](3) + gripper(1) = 7
            state = torch.cat(
                [state[..., :3], state[..., 3:6], state[..., 7:8]], dim=-1
            )
        elif isinstance(state, torch.Tensor) and state.shape[-1] > 7:
            state = state[..., :7]

        # Ensure images have correct shape [C, H, W]
        if top_image is not None and top_image.dim() == 3:
            pass  # Already correct
        elif top_image is not None and top_image.dim() == 4:
            top_image = top_image.squeeze(0)

        if wrist_image is not None and wrist_image.dim() == 3:
            pass  # Already correct
        elif wrist_image is not None and wrist_image.dim() == 4:
            wrist_image = wrist_image.squeeze(0)

        # Initialize history if needed
        if "top" not in image_history:
            image_history["top"] = []
            image_history["wrist"] = []

        # Pad history if needed
        while len(image_history["top"]) < past_img_num:
            if top_image is not None:
                image_history["top"].insert(0, top_image.clone())
            if wrist_image is not None:
                image_history["wrist"].insert(0, wrist_image.clone())

        # Build sequences: [past] + [current] + [future (repeat current)]
        if top_image is not None:
            top_past = image_history["top"][-past_img_num:]
            top_current = [top_image]
            top_future = [top_image.clone() for _ in range(future_img_num)]
            top_seq = torch.stack(top_past + top_current + top_future)  # [T, C, H, W]
        else:
            top_seq = torch.zeros(past_img_num + future_img_num + 1, 3, 112, 112)

        if wrist_image is not None:
            wrist_past = image_history["wrist"][-past_img_num:]
            wrist_current = [wrist_image]
            wrist_future = [wrist_image.clone() for _ in range(future_img_num)]
            wrist_seq = torch.stack(
                wrist_past + wrist_current + wrist_future
            )  # [T, C, H, W]
        else:
            wrist_seq = torch.zeros(past_img_num + future_img_num + 1, 3, 112, 112)

        # Build batch with batch dimension [B=1, ...]
        batch = {
            "observation": {
                # Current images [B, C, H, W]
                "top_image": top_image.unsqueeze(0)
                if top_image is not None
                else torch.zeros(1, 3, 112, 112),
                "wrist_image": wrist_image.unsqueeze(0)
                if wrist_image is not None
                else torch.zeros(1, 3, 112, 112),
                "agentview_image": top_image.unsqueeze(0)
                if top_image is not None
                else torch.zeros(1, 3, 112, 112),
                "eye_in_hand_image": wrist_image.unsqueeze(0)
                if wrist_image is not None
                else torch.zeros(1, 3, 112, 112),
                # Image sequences [B, T, C, H, W]
                "top_image_seq": top_seq.unsqueeze(0),
                "wrist_image_seq": wrist_seq.unsqueeze(0),
                "agentview_image_seq": top_seq.unsqueeze(0),
                "eye_in_hand_image_seq": wrist_seq.unsqueeze(0),
                # State [B, state_dim]
                "state": state.unsqueeze(0) if state.dim() == 1 else state,
            },
            "language": [obs.get("language", "manipulate the object")],
            "action": torch.zeros(1, horizon, action_dim),  # Placeholder
        }

        return batch

    def _update_history(self, obs: Dict, image_history: Dict):
        """Update image history buffer."""
        top_image = obs["observation"].get(
            "top_image", obs["observation"].get("agentview_image")
        )
        wrist_image = obs["observation"].get(
            "wrist_image", obs["observation"].get("eye_in_hand_image")
        )

        if "top" not in image_history:
            image_history["top"] = []
            image_history["wrist"] = []

        if top_image is not None:
            if top_image.dim() == 4:
                top_image = top_image.squeeze(0)
            image_history["top"].append(top_image.clone())
            if len(image_history["top"]) > 10:
                image_history["top"].pop(0)

        if wrist_image is not None:
            if wrist_image.dim() == 4:
                wrist_image = wrist_image.squeeze(0)
            image_history["wrist"].append(wrist_image.clone())
            if len(image_history["wrist"]) > 10:
                image_history["wrist"].pop(0)

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate policy over multiple episodes.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dict with aggregated statistics
        """
        results = []

        for ep in range(num_episodes):
            result = self.evaluate_episode()
            results.append(result)
            print(
                f"Episode {ep + 1}/{num_episodes}: reward={result['total_reward']:.2f}, "
                f"success={result['success']}, steps={result['steps']}"
            )

        # Aggregate results
        success_rate = sum(r["success"] for r in results) / len(results)
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)

        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "num_episodes": num_episodes,
        }


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Robosuite Environment Wrapper")
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-name", default=None, help="Task name to load")
    parser.add_argument("--test", action="store_true", help="Run test")

    args = parser.parse_args()

    if args.test:
        if not (HAS_ROBOSUITE or HAS_LIBERO):
            print("Cannot run test: neither robosuite nor libero is installed")
        else:
            env = RobosuiteEnvWrapper(
                task_suite=args.task_suite,
                task_name=args.task_name or "open",
            )

            obs = env.reset()
            print("Observation keys:", obs.keys())
            print("State shape:", obs["observation"]["state"].shape)

            # Random action test
            action = torch.randn(7)
            obs, reward, done, _, info = env.step(action)
            print(f"Reward: {reward}, Done: {done}")

            env.close()
            print("Test passed!")

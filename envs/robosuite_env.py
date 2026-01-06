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
    print("[WARNING] libero not installed. Install from: https://github.com/Lifelong-Robot-Learning/LIBERO")


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
        """
        self.task_name = task_name
        self.task_suite = task_suite
        self.image_size = image_size
        self.norm_stats = norm_stats
        self.device = device
        self.render_mode = render_mode
        self.use_camera = use_camera
        
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
            raise ValueError(f"Task '{self.task_name}' not found. Available: {available}")
        
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
        self.task_description = task
        print(f"[RobosuiteEnv] Loaded LIBERO task: {task}")
    
    def _setup_robosuite_env(self):
        """Setup basic robosuite environment."""
        # Create a basic manipulation environment
        self.env = suite.make(
            env_name="Lift",  # Default task
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.use_camera,
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            control_freq=20,
        )
        self.task_description = "lift the cube"
        print(f"[RobosuiteEnv] Loaded robosuite Lift task")
    
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
            gripper = torch.from_numpy(obs.get("robot0_gripper_qpos", np.zeros(2))).float()
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
        if hasattr(self.env, 'render'):
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
        
        # History buffers
        image_history = []
        
        for step in range(self.max_episode_steps):
            # Prepare batch input
            batch = self._prepare_batch(obs, image_history)
            
            # Model inference
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            output = self.model(batch, phase='eval')
            
            # Get action (use first action from predicted sequence)
            action = output.actions[0, 0]  # [action_dim]
            
            # Execute action
            obs, reward, done, truncated, info = self.env.step(action)
            
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
        image_history: List,
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch input for model."""
        # Add batch dimension
        batch = {
            "observation": {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in obs["observation"].items()
            },
            "language": [obs.get("language", "")],
        }
        
        # Add image sequences from history
        if len(image_history) > 0:
            for key in ["top_image", "wrist_image"]:
                if key in obs["observation"]:
                    seq = torch.stack(image_history[-5:] + [obs["observation"][key]])
                    batch["observation"][f"{key}_seq"] = seq.unsqueeze(0)
        
        return batch
    
    def _update_history(self, obs: Dict, image_history: List):
        """Update image history buffer."""
        for key in ["top_image", "wrist_image"]:
            if key in obs["observation"]:
                image_history.append(obs["observation"][key])
                if len(image_history) > 10:
                    image_history.pop(0)
    
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
            print(f"Episode {ep+1}/{num_episodes}: reward={result['total_reward']:.2f}, "
                  f"success={result['success']}, steps={result['steps']}")
        
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

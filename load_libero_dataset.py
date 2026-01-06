"""
LIBERO Dataset Loader for H-AIF

This module provides data loading utilities for the LIBERO benchmark dataset,
compatible with robosuite simulation platform.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Try to import required packages
try:
    from huggingface_hub import snapshot_download, hf_hub_download

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ============================================================================
# Dataset Download and Verification
# ============================================================================

LIBERO_DATASETS = {
    "libero_spatial": {
        "repo_id": "libero-project/LIBERO",
        "subdir": "libero_spatial",
        "num_tasks": 10,
    },
    "libero_object": {
        "repo_id": "libero-project/LIBERO",
        "subdir": "libero_object",
        "num_tasks": 10,
    },
    "libero_goal": {
        "repo_id": "libero-project/LIBERO",
        "subdir": "libero_goal",
        "num_tasks": 10,
    },
    "libero_10": {
        "repo_id": "libero-project/LIBERO",
        "subdir": "libero_10",
        "num_tasks": 10,
    },
    "libero_90": {
        "repo_id": "libero-project/LIBERO",
        "subdir": "libero_90",
        "num_tasks": 90,
    },
}


def check_dataset_exists(local_dir: Path, task_suite: str) -> bool:
    """
    Check if the LIBERO dataset exists locally.

    Args:
        local_dir: Local directory path
        task_suite: Name of the task suite (e.g., 'libero_10')

    Returns:
        True if dataset exists, False otherwise
    """
    if not local_dir.exists():
        return False

    # Check for HDF5 files
    hdf5_files = list(local_dir.glob("*.hdf5")) + list(local_dir.glob("**/*.hdf5"))
    return len(hdf5_files) > 0


def download_libero_dataset(
    task_suite: str = "libero_10",
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """
    Download LIBERO dataset from HuggingFace Hub if not exists locally.

    Args:
        task_suite: Name of the task suite (e.g., 'libero_10', 'libero_90')
        local_dir: Local directory to save the dataset
        token: HuggingFace token for authentication

    Returns:
        Path to the local dataset directory
    """
    if not HAS_HF_HUB:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    if task_suite not in LIBERO_DATASETS:
        raise ValueError(
            f"Unknown task suite: {task_suite}. Available: {list(LIBERO_DATASETS.keys())}"
        )

    dataset_info = LIBERO_DATASETS[task_suite]

    if local_dir is None:
        local_dir = Path("./datasets/libero") / task_suite
    else:
        local_dir = Path(local_dir)

    # Check if dataset already exists
    if check_dataset_exists(local_dir, task_suite):
        print(f"[LIBERO] Dataset '{task_suite}' already exists at {local_dir}")
        return local_dir

    print(f"[LIBERO] Downloading dataset '{task_suite}' from HuggingFace...")
    print(f"[LIBERO] This may take a while depending on your connection speed.")

    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download from HuggingFace Hub
    try:
        snapshot_download(
            repo_id=dataset_info["repo_id"],
            repo_type="dataset",
            local_dir=str(local_dir.parent),
            allow_patterns=[f"{dataset_info['subdir']}/*"],
            token=token,
        )
        print(f"[LIBERO] Dataset downloaded to {local_dir}")
    except Exception as e:
        print(f"[LIBERO] Failed to download dataset: {e}")
        print(
            "[LIBERO] Please download manually from: https://huggingface.co/datasets/libero-project/LIBERO"
        )
        raise

    return local_dir


# ============================================================================
# Normalization Utilities
# ============================================================================


class LiberoNormalizer:
    """
    Normalizer for LIBERO dataset, compatible with the existing normalize.py logic.
    """

    def __init__(self, norm_stats: Optional[Dict] = None):
        """
        Args:
            norm_stats: Pre-computed normalization statistics
        """
        self.norm_stats = norm_stats or {}

    @staticmethod
    def compute_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute mean and std for normalization."""
        return {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0) + 1e-8,  # Avoid division by zero
        }

    def normalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        """
        Normalize data using pre-computed statistics.

        Args:
            data: Input tensor
            key: Key for the statistics (e.g., 'action', 'state')

        Returns:
            Normalized tensor
        """
        if key not in self.norm_stats:
            return data

        stats = self.norm_stats[key]
        mean = torch.tensor(stats["mean"], dtype=data.dtype, device=data.device)
        std = torch.tensor(stats["std"], dtype=data.dtype, device=data.device)

        return (data - mean) / std

    def denormalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        """
        Denormalize data back to original scale.

        Args:
            data: Normalized tensor
            key: Key for the statistics

        Returns:
            Denormalized tensor
        """
        if key not in self.norm_stats:
            return data

        stats = self.norm_stats[key]
        mean = torch.tensor(stats["mean"], dtype=data.dtype, device=data.device)
        std = torch.tensor(stats["std"], dtype=data.dtype, device=data.device)

        return data * std + mean

    def save(self, path: Path):
        """Save normalization statistics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        stats_serializable = {}
        for key, stats in self.norm_stats.items():
            stats_serializable[key] = {
                "mean": stats["mean"].tolist()
                if isinstance(stats["mean"], np.ndarray)
                else stats["mean"],
                "std": stats["std"].tolist()
                if isinstance(stats["std"], np.ndarray)
                else stats["std"],
            }

        with open(path, "w") as f:
            json.dump(stats_serializable, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LiberoNormalizer":
        """Load normalization statistics from JSON file."""
        with open(path, "r") as f:
            stats = json.load(f)

        # Convert lists back to numpy arrays
        for key in stats:
            stats[key]["mean"] = np.array(stats[key]["mean"])
            stats[key]["std"] = np.array(stats[key]["std"])

        return cls(norm_stats=stats)


def denormalize_action(
    normalized_action: torch.Tensor,
    norm_stats: Union[Dict, str, Path],
    key: str = "action",
) -> torch.Tensor:
    """
    Denormalize action before sending to robot.

    Args:
        normalized_action: Normalized action tensor
        norm_stats: Normalization statistics dict or path to stats file
        key: Key for the statistics

    Returns:
        Denormalized action tensor in original scale
    """
    if isinstance(norm_stats, (str, Path)):
        normalizer = LiberoNormalizer.load(Path(norm_stats))
    else:
        normalizer = LiberoNormalizer(norm_stats=norm_stats)

    return normalizer.denormalize(normalized_action, key)


# ============================================================================
# LIBERO Dataset Class
# ============================================================================


class LiberoDataset(Dataset):
    """
    PyTorch Dataset for LIBERO benchmark.

    Each sample contains:
    - observation: dict with 'agentview_image', 'eye_in_hand_image', 'state'
    - action: 7D action (6D end-effector pose + 1D gripper)
    - language: task description string
    """

    def __init__(
        self,
        data_dir: Path,
        task_suite: str = "libero_10",
        horizon: int = 50,
        past_img_num: int = 5,
        future_img_num: int = 5,
        image_size: int = 112,
        normalize: bool = True,
        norm_stats: Optional[Dict] = None,
    ):
        """
        Args:
            data_dir: Path to LIBERO dataset directory
            task_suite: Name of the task suite
            horizon: Action prediction horizon
            past_img_num: Number of past images to include
            future_img_num: Number of future images to include
            image_size: Target image size (will resize)
            normalize: Whether to normalize data
            norm_stats: Pre-computed normalization statistics
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required. Install with: pip install h5py")

        self.data_dir = Path(data_dir)
        self.task_suite = task_suite
        self.horizon = horizon
        self.past_img_num = past_img_num
        self.future_img_num = future_img_num
        self.image_size = image_size
        self.normalize_data = normalize

        # Load all HDF5 files
        self.hdf5_files = list(self.data_dir.glob("*.hdf5"))
        if len(self.hdf5_files) == 0:
            self.hdf5_files = list(self.data_dir.glob("**/*.hdf5"))

        if len(self.hdf5_files) == 0:
            raise ValueError(f"No HDF5 files found in {self.data_dir}")

        print(f"[LIBERO] Found {len(self.hdf5_files)} HDF5 files")

        # Build index of all samples
        self.samples = []
        self.task_descriptions = {}
        self._build_index()

        # Setup normalizer
        if norm_stats is not None:
            self.normalizer = LiberoNormalizer(norm_stats=norm_stats)
        elif normalize:
            self.normalizer = self._compute_norm_stats()
        else:
            self.normalizer = LiberoNormalizer()

    def _build_index(self):
        """Build index of all samples from HDF5 files."""
        for hdf5_path in self.hdf5_files:
            task_name = hdf5_path.stem

            with h5py.File(hdf5_path, "r") as f:
                # Get task description from attributes
                if "task_description" in f.attrs:
                    self.task_descriptions[task_name] = f.attrs["task_description"]
                elif "language_instruction" in f.attrs:
                    self.task_descriptions[task_name] = f.attrs["language_instruction"]
                else:
                    # Extract from filename
                    self.task_descriptions[task_name] = task_name.replace("_", " ")

                # Get demos
                if "data" in f:
                    demo_keys = list(f["data"].keys())
                else:
                    demo_keys = [k for k in f.keys() if k.startswith("demo")]

                for demo_key in demo_keys:
                    if "data" in f:
                        demo = f["data"][demo_key]
                    else:
                        demo = f[demo_key]

                    # Get episode length
                    if "actions" in demo:
                        ep_len = len(demo["actions"])
                    elif "action" in demo:
                        ep_len = len(demo["action"])
                    else:
                        continue

                    # Create samples with valid indices
                    min_idx = self.past_img_num
                    max_idx = ep_len - self.future_img_num - self.horizon

                    for idx in range(min_idx, max(min_idx, max_idx)):
                        self.samples.append(
                            {
                                "hdf5_path": hdf5_path,
                                "task_name": task_name,
                                "demo_key": demo_key,
                                "idx": idx,
                                "ep_len": ep_len,
                            }
                        )

        print(f"[LIBERO] Built index with {len(self.samples)} samples")

    def _compute_norm_stats(self) -> LiberoNormalizer:
        """Compute normalization statistics from dataset."""
        print("[LIBERO] Computing normalization statistics...")

        all_actions = []
        all_states = []

        # Sample a subset for efficiency
        sample_indices = np.random.choice(
            len(self.samples), min(1000, len(self.samples)), replace=False
        )

        for idx in sample_indices:
            sample_info = self.samples[idx]

            with h5py.File(sample_info["hdf5_path"], "r") as f:
                if "data" in f:
                    demo = f["data"][sample_info["demo_key"]]
                else:
                    demo = f[sample_info["demo_key"]]

                # Get actions
                if "actions" in demo:
                    actions = demo["actions"][:]
                elif "action" in demo:
                    actions = demo["action"][:]
                else:
                    continue

                all_actions.append(actions)

                # Get states if available
                if "obs" in demo and "robot0_eef_pos" in demo["obs"]:
                    eef_pos = demo["obs"]["robot0_eef_pos"][:]
                    eef_quat = demo["obs"]["robot0_eef_quat"][:]
                    gripper = demo["obs"]["robot0_gripper_qpos"][:]
                    state = np.concatenate([eef_pos, eef_quat, gripper], axis=-1)
                    all_states.append(state)

        norm_stats = {}

        if all_actions:
            all_actions = np.concatenate(all_actions, axis=0)
            norm_stats["action"] = LiberoNormalizer.compute_stats(all_actions)

        if all_states:
            all_states = np.concatenate(all_states, axis=0)
            norm_stats["state"] = LiberoNormalizer.compute_stats(all_states)

        print("[LIBERO] Normalization statistics computed")
        return LiberoNormalizer(norm_stats=norm_stats)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]

        with h5py.File(sample_info["hdf5_path"], "r") as f:
            if "data" in f:
                demo = f["data"][sample_info["demo_key"]]
            else:
                demo = f[sample_info["demo_key"]]

            t = sample_info["idx"]

            # Get images
            agentview_seq = self._get_image_sequence(demo, "agentview_rgb", t)
            eye_in_hand_seq = self._get_image_sequence(
                demo, "robot0_eye_in_hand_rgb", t
            )

            # Get current images (last frame in sequence)
            # agentview_seq is (T, C, H, W), so get last time step
            agentview_image = (
                agentview_seq[-1] if agentview_seq.ndim == 4 else agentview_seq
            )
            eye_in_hand_image = (
                eye_in_hand_seq[-1] if eye_in_hand_seq.ndim == 4 else eye_in_hand_seq
            )

            # Get actions (7D: 6D pose + 1D gripper)
            if "actions" in demo:
                actions = demo["actions"][t : t + self.horizon]
            else:
                actions = demo["action"][t : t + self.horizon]
            actions = torch.from_numpy(actions[:]).float()

            # Ensure 7D action
            if actions.shape[-1] != 7:
                # Pad or truncate to 7D
                if actions.shape[-1] < 7:
                    actions = torch.nn.functional.pad(
                        actions, (0, 7 - actions.shape[-1])
                    )
                else:
                    actions = actions[..., :7]

            # Get state
            if "obs" in demo:
                obs = demo["obs"]
                if "robot0_eef_pos" in obs:
                    eef_pos = torch.from_numpy(obs["robot0_eef_pos"][t]).float()
                    eef_quat = torch.from_numpy(obs["robot0_eef_quat"][t]).float()
                    gripper = torch.from_numpy(obs["robot0_gripper_qpos"][t]).float()
                    state = torch.cat(
                        [eef_pos, eef_quat, gripper[:1]], dim=-1
                    )  # 7D state
                else:
                    state = torch.zeros(7)
            else:
                state = torch.zeros(7)

            # Get language instruction
            language = self.task_descriptions.get(sample_info["task_name"], "")

            # Normalize if enabled
            if self.normalize_data:
                actions = self.normalizer.normalize(actions, "action")
                state = self.normalizer.normalize(state, "state")

            return {
                "observation": {
                    "agentview_image": agentview_image,
                    "agentview_image_seq": agentview_seq,
                    "eye_in_hand_image": eye_in_hand_image,
                    "eye_in_hand_image_seq": eye_in_hand_seq,
                    # Map to existing keys for compatibility
                    "top_image": agentview_image,
                    "top_image_seq": agentview_seq,
                    "wrist_image": eye_in_hand_image,
                    "wrist_image_seq": eye_in_hand_seq,
                    "state": state,
                },
                "action": actions,
                "language": language,
            }

    def _get_image_sequence(
        self,
        demo,
        key: str,
        t: int,
    ) -> torch.Tensor:
        """Get image sequence from demo."""
        obs = demo.get("obs", demo)

        # Find the correct image key - try various naming conventions
        # LIBERO uses: agentview_rgb, eye_in_hand_rgb
        # Some datasets use: robot0_eye_in_hand_rgb, agentview_image, etc.
        base_key = key.replace("robot0_", "")  # Remove robot0_ prefix if present
        image_keys = [
            key,  # Original key
            base_key,  # Without robot0_ prefix
            key.replace("_rgb", "_image"),  # _image variant
            base_key.replace("_rgb", "_image"),  # _image without prefix
            f"obs/{key}",  # Nested in obs/
            f"obs/{base_key}",  # Nested without prefix
        ]
        image_data = None

        for k in image_keys:
            if k in obs:
                image_data = obs[k]
                break

        if image_data is None:
            # Return dummy images with correct shape (T, C, H, W)
            seq_len = self.past_img_num + self.future_img_num + 1
            return torch.zeros(seq_len, 3, self.image_size, self.image_size)

        # Get sequence
        start_idx = max(0, t - self.past_img_num)
        end_idx = min(len(image_data), t + self.future_img_num + 1)

        images = image_data[start_idx:end_idx]
        images = torch.from_numpy(images[:]).float()

        # Handle different image formats
        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Rearrange from (T, H, W, C) to (C, T, H, W) or (T, C, H, W)
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)  # (T, C, H, W)

        # Normalize to [0, 1]
        if images.max() > 1.0:
            images = images / 255.0

        # Resize if needed
        if images.shape[-2] != self.image_size or images.shape[-1] != self.image_size:
            images = torch.nn.functional.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Return as (T, C, H, W) for compatibility with model expecting [B, T, C, H, W]
        return images  # (T, C, H, W)


# ============================================================================
# DataLoader Factory
# ============================================================================


def load_libero_dataloader(
    task_suite: str = "libero_10",
    local_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    horizon: int = 50,
    past_img_num: int = 5,
    future_img_num: int = 5,
    image_size: int = 112,
    train_ratio: float = 0.9,
    normalize: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, LiberoNormalizer]:
    """
    Load LIBERO dataset and create train/val dataloaders.

    This function will:
    1. Check if dataset exists locally
    2. Download from HuggingFace if not exists
    3. Create train/val splits
    4. Return dataloaders and normalizer

    Args:
        task_suite: Name of the task suite
        local_dir: Local directory for dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        horizon: Action prediction horizon
        past_img_num: Number of past images
        future_img_num: Number of future images
        image_size: Target image size
        train_ratio: Ratio for train split
        normalize: Whether to normalize data
        seed: Random seed for splitting

    Returns:
        Tuple of (train_dataloader, val_dataloader, normalizer)
    """
    # Setup paths
    if local_dir is None:
        local_dir = Path("./datasets/libero") / task_suite
    else:
        local_dir = Path(local_dir)

    # Download if not exists
    local_dir = download_libero_dataset(
        task_suite=task_suite,
        local_dir=local_dir,
    )

    # Create dataset
    full_dataset = LiberoDataset(
        data_dir=local_dir,
        task_suite=task_suite,
        horizon=horizon,
        past_img_num=past_img_num,
        future_img_num=future_img_num,
        image_size=image_size,
        normalize=normalize,
    )

    # Split dataset
    np.random.seed(seed)
    indices = np.random.permutation(len(full_dataset))
    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset.normalizer


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LIBERO Dataset Loader")
    parser.add_argument(
        "--task-suite", default="libero_10", choices=list(LIBERO_DATASETS.keys())
    )
    parser.add_argument("--local-dir", default=None)
    parser.add_argument(
        "--download-only", action="store_true", help="Only download dataset, don't load"
    )
    parser.add_argument("--batch-size", type=int, default=2)

    args = parser.parse_args()

    if args.download_only:
        download_libero_dataset(args.task_suite, args.local_dir)
    else:
        train_loader, val_loader, normalizer = load_libero_dataloader(
            task_suite=args.task_suite,
            local_dir=args.local_dir,
            batch_size=args.batch_size,
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Test loading a batch
        batch = next(iter(train_loader))
        print(f"Agentview image shape: {batch['observation']['agentview_image'].shape}")
        print(f"Action shape: {batch['action'].shape}")
        print(f"Language: {batch['language'][:2]}")

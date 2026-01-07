#!/usr/bin/env python3
"""
t-SNE Visualization for Latent Space Analysis

This script visualizes the latent space (z_mix) of the H-AIF model
to demonstrate that different tasks/language instructions produce
distinct clusters in the latent space.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from model.models import ActNet
from load_libero_dataset import load_libero_dataloader


def setup_matplotlib_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.figsize": (10, 8),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    sns.set_style("whitegrid")


def load_model(checkpoint_path: str, config_path: str, device: str):
    """Load the trained model from checkpoint."""
    config = OmegaConf.load(config_path)

    model = ActNet(config, device=device)
    model = model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    model.eval()
    return model, config


def collect_latent_vectors(model, dataloader, device, n_samples=500):
    """
    Collect latent vectors (z_mix) and corresponding task labels from the model.

    Returns:
        z_all: numpy array of shape [N, D] - latent vectors
        labels: list of task names
    """
    z_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break

            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for k in batch[key]:
                        if isinstance(batch[key][k], torch.Tensor):
                            batch[key][k] = batch[key][k].to(device)

            # Forward pass
            try:
                output = model(batch, phase="eval")

                # Get z_mix from output
                if hasattr(output, "z_mix") and output.z_mix is not None:
                    z = output.z_mix.cpu().numpy()
                    # Average over sequence dimension if present
                    if len(z.shape) == 3:
                        z = z.mean(axis=1)
                    z_list.append(z)

                    # Get task labels from language
                    if "language" in batch:
                        labels = batch["language"]
                        if isinstance(labels, list):
                            label_list.extend(labels)
                        else:
                            label_list.extend([str(l) for l in labels])
                    else:
                        label_list.extend([f"task_{i}" for i in range(z.shape[0])])

                    count += z.shape[0]
                    print(f"Collected {count}/{n_samples} samples", end="\r")
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    print()

    if not z_list:
        raise ValueError("No latent vectors collected. Check model output.")

    z_all = np.concatenate(z_list, axis=0)[:n_samples]
    labels = label_list[:n_samples]

    return z_all, labels


def create_task_color_map(labels):
    """Create a color mapping for unique tasks."""
    unique_tasks = list(set(labels))
    colors = sns.color_palette("husl", len(unique_tasks))
    color_map = {task: colors[i] for i, task in enumerate(unique_tasks)}
    return color_map, unique_tasks


def visualize_tsne(z_all, labels, output_path, method="tsne", perplexity=30):
    """
    Create t-SNE (or PCA) visualization of latent space.
    """
    print(f"Running {method.upper()} dimensionality reduction...")

    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(z_all) - 1),
            random_state=42,
            init="pca",
        )
    else:
        reducer = PCA(n_components=2)

    z_2d = reducer.fit_transform(z_all)

    # Create color mapping
    color_map, unique_tasks = create_task_color_map(labels)
    colors = [color_map[l] for l in labels]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each task with different color
    for task in unique_tasks:
        mask = np.array([l == task for l in labels])
        ax.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            c=[color_map[task]],
            label=task[:50],
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=14)
    ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=14)
    ax.set_title(
        f"Latent Space Visualization ({method.upper()})\nDifferent colors represent different tasks",
        fontsize=16,
    )

    # Add legend with smaller font for many tasks
    if len(unique_tasks) <= 10:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    else:
        # Put legend outside for many tasks
        ax.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, framealpha=0.9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")

    return z_2d


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE Visualization for H-AIF Latent Space"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint/model_best.pth.tar",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="config_libero.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tsne_visualization.png",
        help="Output path for visualization",
    )
    parser.add_argument(
        "--n-samples", type=int, default=500, help="Number of samples to collect"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tsne", "pca"],
        default="tsne",
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--perplexity", type=int, default=30, help="t-SNE perplexity parameter"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Setup
    setup_matplotlib_style()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model(args.checkpoint, args.config, device)

    # Load data
    print("Loading dataset...")
    train_loader, val_loader, normalizer = load_libero_dataloader(
        task_suite=config.task_suite,
        local_dir=config.datasets_path,
        batch_size=config.batchsize,
        horizon=config.horizon,
        past_img_num=config.past_img_num,
        future_img_num=config.future_img_num,
    )

    # Collect latent vectors
    print("Collecting latent vectors...")
    z_all, labels = collect_latent_vectors(model, val_loader, device, args.n_samples)
    print(f"Collected {len(z_all)} samples with {len(set(labels))} unique tasks")

    # Visualize
    visualize_tsne(
        z_all, labels, args.output, method=args.method, perplexity=args.perplexity
    )

    # Also save PCA if using t-SNE
    if args.method == "tsne":
        pca_output = args.output.replace(".png", "_pca.png")
        visualize_tsne(z_all, labels, pca_output, method="pca")


if __name__ == "__main__":
    main()

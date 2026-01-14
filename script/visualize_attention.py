#!/usr/bin/env python3
"""
Attention Weights Visualization for Multi-Modal Analysis

This script visualizes the attention weights from the MultiModalAttention module
to analyze how different modalities (Vision, Proprioception, Language) contribute
to the action prediction at different stages of task execution.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
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
            "figure.figsize": (12, 8),
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
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

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

        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    model.eval()
    return model, config


def collect_attention_weights(model, dataloader, device, n_samples=100):
    """
    Collect attention weights from the model.

    Returns:
        attention_weights_list: list of attention weight tensors
        languages: list of task descriptions
        timesteps: list of timestep indices
    """
    attention_weights_list = []
    languages = []

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

                # Get attention weights from output
                if (
                    hasattr(output, "attention_weights")
                    and output.attention_weights is not None
                ):
                    attn = output.attention_weights.cpu().numpy()
                    attention_weights_list.append(attn)

                    # Get language/task descriptions
                    if "language" in batch:
                        labels = batch["language"]
                        if isinstance(labels, list):
                            languages.extend(labels)
                        else:
                            languages.extend([str(l) for l in labels])

                    count += attn.shape[0]
                    print(f"Collected {count}/{n_samples} samples", end="\r")
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    print()

    if not attention_weights_list:
        raise ValueError("No attention weights collected. Check model output.")

    return attention_weights_list, languages


def analyze_modality_weights(attention_weights_list, config):
    """
    Analyze the attention weights to determine modality contributions.

    The attention weights shape is typically [B, num_modalities, 1]
    We need to map these to: Vision (Grip + Side), Action/Proprioception, Language
    """
    # Concatenate all attention weights
    all_weights = np.concatenate(attention_weights_list, axis=0)

    # Get statistics
    mean_weights = all_weights.mean(axis=0)
    std_weights = all_weights.std(axis=0)

    # Determine modality assignment based on config
    use_language = getattr(config, "use_language", False)

    # The modalities are typically concatenated as:
    # [grip_visual (120), side_visual (120), action (120), language (1 if enabled)]
    # So the attention is applied on the sequence dimension

    # For simplicity, we'll analyze the distribution across the sequence positions
    # and compute aggregate statistics

    return mean_weights, std_weights, all_weights


def visualize_attention_distribution(attention_weights_list, output_path, config):
    """
    Visualize the distribution of attention weights across modalities.
    """
    all_weights = np.concatenate(attention_weights_list, axis=0)

    # Squeeze extra dimensions
    while all_weights.ndim > 2 and all_weights.shape[-1] == 1:
        all_weights = all_weights.squeeze(-1)

    if all_weights.ndim == 1:
        all_weights = all_weights.reshape(-1, 1)

    # Determine modality boundaries based on config
    use_language = getattr(config, "use_language", False)
    embedding_dim = getattr(config.model, "embedding_dim", 120)

    # Compute mean attention per segment (approximate modality regions)
    n_positions = all_weights.shape[1]

    # Define modality regions (approximate)
    if use_language and n_positions > 3:
        # With language: [visual_grip, visual_side, action, language]
        segment_size = (n_positions - 1) // 3
        segments = {
            "Vision (Grip)": (0, segment_size),
            "Vision (Wrist)": (segment_size, 2 * segment_size),
            "Action/Proprioception": (2 * segment_size, n_positions - 1),
            "Language": (n_positions - 1, n_positions),
        }
    else:
        # Without language: [visual_grip, visual_side, action]
        segment_size = n_positions // 3
        segments = {
            "Vision (Grip)": (0, segment_size),
            "Vision (Wrist)": (segment_size, 2 * segment_size),
            "Action/Proprioception": (2 * segment_size, n_positions),
        }

    # Calculate mean attention for each segment
    modality_weights = {}
    modality_stds = {}
    for name, (start, end) in segments.items():
        if end > start:
            segment_weights = all_weights[:, start:end].mean(axis=1)
            modality_weights[name] = segment_weights.mean()
            modality_stds[name] = segment_weights.std()

    # Create bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot of mean weights
    ax1 = axes[0]
    modalities = list(modality_weights.keys())
    means = [modality_weights[m] for m in modalities]
    stds = [modality_stds[m] for m in modalities]

    colors = sns.color_palette("husl", len(modalities))
    bars = ax1.bar(
        modalities,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    ax1.set_ylabel("Mean Attention Weight", fontsize=12)
    ax1.set_xlabel("Modality", fontsize=12)
    ax1.set_title("Attention Distribution Across Modalities", fontsize=14)
    ax1.tick_params(axis="x", rotation=15)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax1.annotate(
            f"{mean:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Heatmap of attention across positions
    ax2 = axes[1]
    # Show a sample of attention patterns
    n_show = min(50, all_weights.shape[0])
    sample_weights = all_weights[:n_show]

    im = ax2.imshow(sample_weights, aspect="auto", cmap="viridis")
    ax2.set_xlabel("Sequence Position", fontsize=12)
    ax2.set_ylabel("Sample Index", fontsize=12)
    ax2.set_title("Attention Weight Heatmap", fontsize=14)

    # Add modality region indicators
    for name, (start, end) in segments.items():
        mid = (start + end) / 2
        ax2.axvline(x=start - 0.5, color="white", linestyle="--", alpha=0.5)

    plt.colorbar(im, ax=ax2, label="Attention Weight")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved attention visualization to {output_path}")

    return modality_weights, modality_stds


def visualize_temporal_attention(attention_weights_list, output_path, config):
    """
    Visualize how attention changes over time/stages of task execution.
    """
    # This would require collecting attention weights at different timesteps
    # For now, we show the distribution across samples

    all_weights = np.concatenate(attention_weights_list, axis=0)

    # Squeeze extra dimensions
    while all_weights.ndim > 2 and all_weights.shape[-1] == 1:
        all_weights = all_weights.squeeze(-1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot attention evolution (treating sample index as a proxy for time)
    n_positions = all_weights.shape[1]

    # Divide into early, middle, late samples
    n_samples = all_weights.shape[0]
    early = all_weights[: n_samples // 3].mean(axis=0)
    middle = all_weights[n_samples // 3 : 2 * n_samples // 3].mean(axis=0)
    late = all_weights[2 * n_samples // 3 :].mean(axis=0)

    x = np.arange(n_positions)
    ax.plot(x, early, label="Early Stage", linewidth=2, marker="o", markersize=3)
    ax.plot(x, middle, label="Middle Stage", linewidth=2, marker="s", markersize=3)
    ax.plot(x, late, label="Late Stage", linewidth=2, marker="^", markersize=3)

    ax.set_xlabel("Sequence Position", fontsize=12)
    ax.set_ylabel("Mean Attention Weight", fontsize=12)
    ax.set_title("Temporal Evolution of Attention Weights", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    temporal_output = output_path.replace(".png", "_temporal.png")
    plt.savefig(temporal_output, dpi=300, bbox_inches="tight")
    print(f"Saved temporal attention visualization to {temporal_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Attention Weights Visualization for H-AIF"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./results/model_best.pth.tar",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="config_libero.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/attention_visualization.png",
        help="Output path for visualization",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of samples to collect"
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

    # Collect attention weights
    print("Collecting attention weights...")
    attention_weights_list, languages = collect_attention_weights(
        model, val_loader, device, args.n_samples
    )
    print(f"Collected {len(attention_weights_list)} batches")

    # Visualize
    modality_weights, modality_stds = visualize_attention_distribution(
        attention_weights_list, args.output, config
    )

    # Print summary
    print("\n=== Attention Weight Summary ===")
    for modality, weight in modality_weights.items():
        std = modality_stds[modality]
        print(f"{modality}: {weight:.4f} ± {std:.4f}")

    # Temporal visualization
    visualize_temporal_attention(attention_weights_list, args.output, config)


if __name__ == "__main__":
    main()

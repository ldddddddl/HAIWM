#!/usr/bin/env python3
"""
Enhanced Attention Weights Visualization for Multi-Modal Analysis

This script provides comprehensive visualization of attention weights from the
MultiModalAttention module, including:
1. Cross-modal attention matrix visualization
2. Modality-aggregated attention heatmap
3. Attention entropy analysis
4. Task-semantic attention patterns
5. Temporal attention dynamics

Author: Enhanced from original visualize_attention.py
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from model.models import ActNet
from load_libero_dataset import load_libero_dataloader


# =============================================================================
# Configuration
# =============================================================================

MODALITY_COLORS = {
    "Vision (Grip)": "#E74C3C",  # Red
    "Vision (Wrist)": "#3498DB",  # Blue
    "Action/Proprio": "#2ECC71",  # Green
    "Language": "#9B59B6",  # Purple
}

MODALITY_NAMES = ["Vision (Grip)", "Vision (Wrist)", "Action/Proprio", "Language"]


def setup_matplotlib_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.figsize": (14, 10),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_style("whitegrid")


def get_modality_boundaries(config):
    """
    Calculate modality boundaries in the sequence dimension.

    Returns:
        dict with modality names as keys and (start, end) tuples as values
    """
    enc_out_dim = getattr(config, "enc_out_dim", 120)
    act_embedding_dim = getattr(config.act_model_enc, "embedding_dim", 120)
    use_language = getattr(config, "use_language", False)

    boundaries = {
        "Vision (Grip)": (0, enc_out_dim),
        "Vision (Wrist)": (enc_out_dim, enc_out_dim * 2),
        "Action/Proprio": (enc_out_dim * 2, enc_out_dim * 2 + act_embedding_dim),
    }

    if use_language:
        lang_start = enc_out_dim * 2 + act_embedding_dim
        boundaries["Language"] = (lang_start, lang_start + 1)

    return boundaries


# =============================================================================
# Model Loading
# =============================================================================


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


# =============================================================================
# Data Collection with Full Attention Matrix
# =============================================================================


def collect_detailed_attention(model, dataloader, device, n_samples=100):
    """
    Collect detailed attention weights from the model.

    Returns:
        dict containing:
        - full_attention: list of [S, S] attention matrices
        - languages: list of task descriptions
        - per_position_weights: list of [S] per-position weights
    """
    full_attention_list = []
    per_position_weights_list = []
    languages = []

    model.eval()

    # Get the attention module for extracting full attention matrix
    if hasattr(model, "modal_fusion_model") and hasattr(
        model.modal_fusion_model, "attention"
    ):
        attention_module = model.modal_fusion_model.attention
    else:
        attention_module = None

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

            try:
                # Forward pass
                output = model(batch, phase="eval")

                # Get attention weights from output
                if (
                    hasattr(output, "attention_weights")
                    and output.attention_weights is not None
                ):
                    attn = output.attention_weights.cpu().numpy()
                    per_position_weights_list.append(attn)

                # Try to get full attention matrix from the attention module
                # This requires hooking into the attention module
                if attention_module is not None:
                    # The attention weights are stored during forward pass
                    # We need to capture them via a hook or modify the model
                    pass

                # Get language/task descriptions
                if "language" in batch:
                    labels = batch["language"]
                    if isinstance(labels, list):
                        languages.extend(labels)
                    else:
                        languages.extend([str(l) for l in labels])

                count += attn.shape[0] if len(per_position_weights_list) > 0 else 1
                print(f"Collected {count}/{n_samples} samples", end="\r")

            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    print()

    if not per_position_weights_list:
        raise ValueError("No attention weights collected. Check model output.")

    return {
        "per_position_weights": per_position_weights_list,
        "languages": languages,
    }


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_cross_modal_attention_matrix(attention_data, config, output_path):
    """
    Visualize the cross-modal attention as a heatmap with modality boundaries.

    Since we have per-position weights [B, S, 1], we create a synthetic
    attention matrix by computing outer products or correlation.
    """
    all_weights = np.concatenate(attention_data["per_position_weights"], axis=0)

    # Squeeze extra dimensions
    while all_weights.ndim > 2 and all_weights.shape[-1] == 1:
        all_weights = all_weights.squeeze(-1)

    if all_weights.ndim == 1:
        all_weights = all_weights.reshape(1, -1)

    # Average across samples
    mean_weights = all_weights.mean(axis=0)

    # Get modality boundaries
    boundaries = get_modality_boundaries(config)
    n_positions = len(mean_weights)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Attention weights per position with modality regions
    ax1 = axes[0]
    x = np.arange(n_positions)

    # Color by modality
    colors = []
    for i in range(n_positions):
        color = "#CCCCCC"  # Default gray
        for name, (start, end) in boundaries.items():
            if start <= i < end:
                color = MODALITY_COLORS.get(name, "#CCCCCC")
                break
        colors.append(color)

    bars = ax1.bar(x, mean_weights, color=colors, edgecolor="none", width=1.0)

    # Add modality region labels
    for name, (start, end) in boundaries.items():
        mid = (start + end) / 2
        ax1.axvline(
            x=start - 0.5, color="black", linestyle="--", alpha=0.5, linewidth=0.8
        )
        # Add shaded region
        ax1.axvspan(
            start - 0.5,
            end - 0.5,
            alpha=0.1,
            color=MODALITY_COLORS.get(name, "#CCCCCC"),
        )
        ax1.text(
            mid,
            ax1.get_ylim()[1] * 0.95,
            name.split()[0],
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xlabel("Sequence Position", fontsize=11)
    ax1.set_ylabel("Mean Attention Weight", fontsize=11)
    ax1.set_title("Attention Distribution Across Sequence Positions", fontsize=13)
    ax1.set_xlim(-0.5, n_positions - 0.5)

    # Right: Modality-level aggregated bar chart
    ax2 = axes[1]

    modality_weights = {}
    modality_stds = {}
    for name, (start, end) in boundaries.items():
        if end > len(mean_weights):
            end = len(mean_weights)
        if start < end:
            segment_weights = all_weights[:, start:end].mean(axis=1)
            modality_weights[name] = segment_weights.mean()
            modality_stds[name] = segment_weights.std()

    names = list(modality_weights.keys())
    values = [modality_weights[n] for n in names]
    stds = [modality_stds[n] for n in names]
    colors_bar = [MODALITY_COLORS.get(n, "#888888") for n in names]

    bars2 = ax2.bar(
        names,
        values,
        yerr=stds,
        capsize=5,
        color=colors_bar,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
    )

    # Add value labels
    for bar, val in zip(bars2, values):
        ax2.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_ylabel("Mean Attention Weight", fontsize=11)
    ax2.set_title("Modality-Aggregated Attention Weights", fontsize=13)
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved cross-modal attention to {output_path}")
    plt.close()

    return modality_weights, modality_stds


def plot_attention_heatmap_samples(attention_data, config, output_path, n_show=30):
    """
    Create a detailed heatmap of attention weights across samples.
    """
    all_weights = np.concatenate(attention_data["per_position_weights"], axis=0)

    # Squeeze extra dimensions
    while all_weights.ndim > 2 and all_weights.shape[-1] == 1:
        all_weights = all_weights.squeeze(-1)

    n_samples = min(n_show, all_weights.shape[0])
    sample_weights = all_weights[:n_samples]

    boundaries = get_modality_boundaries(config)
    n_positions = sample_weights.shape[1]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create heatmap
    im = ax.imshow(
        sample_weights, aspect="auto", cmap="viridis", interpolation="nearest"
    )

    # Add modality boundary lines and labels
    for name, (start, end) in boundaries.items():
        ax.axvline(x=start - 0.5, color="white", linestyle="-", linewidth=2, alpha=0.8)
        ax.axvline(x=end - 0.5, color="white", linestyle="-", linewidth=2, alpha=0.8)

    # Add modality labels at top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = []
    tick_labels = []
    for name, (start, end) in boundaries.items():
        mid = (start + end) / 2
        ticks.append(mid)
        tick_labels.append(name)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels, fontsize=10, fontweight="bold")
    ax2.tick_params(length=0)

    ax.set_xlabel("Sequence Position", fontsize=11)
    ax.set_ylabel("Sample Index", fontsize=11)
    ax.set_title("Attention Weight Heatmap Across Samples", fontsize=13, pad=30)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved attention heatmap to {output_path}")
    plt.close()


def plot_attention_entropy(attention_data, config, output_path):
    """
    Analyze and visualize attention entropy per modality.

    Lower entropy = more focused attention
    Higher entropy = more distributed attention
    """
    all_weights = np.concatenate(attention_data["per_position_weights"], axis=0)

    while all_weights.ndim > 2 and all_weights.shape[-1] == 1:
        all_weights = all_weights.squeeze(-1)

    boundaries = get_modality_boundaries(config)

    # Calculate entropy for each sample's attention distribution
    # Normalize weights per sample
    eps = 1e-10
    normalized_weights = all_weights / (all_weights.sum(axis=1, keepdims=True) + eps)
    sample_entropies = -np.sum(
        normalized_weights * np.log(normalized_weights + eps), axis=1
    )

    # Calculate entropy per modality region
    modality_entropies = {}
    for name, (start, end) in boundaries.items():
        if end > all_weights.shape[1]:
            end = all_weights.shape[1]
        if start < end:
            region_weights = all_weights[:, start:end]
            region_norm = region_weights / (
                region_weights.sum(axis=1, keepdims=True) + eps
            )
            region_entropy = -np.sum(region_norm * np.log(region_norm + eps), axis=1)
            modality_entropies[name] = region_entropy

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Overall entropy distribution
    ax1 = axes[0]
    ax1.hist(sample_entropies, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(
        sample_entropies.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {sample_entropies.mean():.3f}",
    )
    ax1.set_xlabel("Attention Entropy", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of Attention Entropy Across Samples", fontsize=13)
    ax1.legend()

    # Right: Violin plot of entropy per modality
    ax2 = axes[1]

    modality_names = list(modality_entropies.keys())
    modality_values = [modality_entropies[n] for n in modality_names]
    colors = [MODALITY_COLORS.get(n, "#888888") for n in modality_names]

    parts = ax2.violinplot(
        modality_values,
        positions=range(len(modality_names)),
        showmeans=True,
        showmedians=True,
    )

    # Color violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax2.set_xticks(range(len(modality_names)))
    ax2.set_xticklabels(modality_names, rotation=15)
    ax2.set_xlabel("Modality", fontsize=11)
    ax2.set_ylabel("Entropy", fontsize=11)
    ax2.set_title("Attention Entropy by Modality Region", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved entropy analysis to {output_path}")
    plt.close()


def plot_task_specific_attention(attention_data, config, output_path):
    """
    Visualize attention patterns grouped by task/language instruction.
    """
    all_weights = np.concatenate(attention_data["per_position_weights"], axis=0)
    languages = attention_data.get("languages", [])

    while all_weights.ndim > 2 and all_weights.shape[-1] == 1:
        all_weights = all_weights.squeeze(-1)

    if len(languages) < len(all_weights):
        # Pad with "Unknown" if needed
        languages = languages + ["Unknown"] * (len(all_weights) - len(languages))
    elif len(languages) > len(all_weights):
        languages = languages[: len(all_weights)]

    # Group by task
    task_weights = defaultdict(list)
    for weight, lang in zip(all_weights, languages):
        task_weights[lang].append(weight)

    # Get unique tasks (limit to top 6 for visualization)
    unique_tasks = list(task_weights.keys())[:6]

    if len(unique_tasks) == 0:
        print("No task information available for task-specific visualization")
        return

    boundaries = get_modality_boundaries(config)

    n_tasks = len(unique_tasks)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, task in enumerate(unique_tasks):
        if idx >= 6:
            break

        ax = axes[idx]
        task_data = np.array(task_weights[task])
        mean_weights = task_data.mean(axis=0)
        std_weights = task_data.std(axis=0)

        x = np.arange(len(mean_weights))
        ax.fill_between(
            x,
            mean_weights - std_weights,
            mean_weights + std_weights,
            alpha=0.3,
            color="steelblue",
        )
        ax.plot(x, mean_weights, color="steelblue", linewidth=1.5)

        # Add modality regions
        for name, (start, end) in boundaries.items():
            ax.axvline(x=start, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

        # Truncate long task names
        title = task if len(task) <= 35 else task[:32] + "..."
        ax.set_title(f'"{title}"', fontsize=9, style="italic")
        ax.set_xlabel("Position", fontsize=9)
        ax.set_ylabel("Attention", fontsize=9)

    # Hide unused subplots
    for idx in range(len(unique_tasks), 6):
        axes[idx].set_visible(False)

    plt.suptitle("Task-Specific Attention Patterns", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved task-specific attention to {output_path}")
    plt.close()


def plot_temporal_dynamics(attention_data, config, output_path):
    """
    Improved temporal analysis showing attention evolution across trajectory.
    """
    all_weights = np.concatenate(attention_data["per_position_weights"], axis=0)

    while all_weights.ndim > 2 and all_weights.shape[-1] == 1:
        all_weights = all_weights.squeeze(-1)

    boundaries = get_modality_boundaries(config)
    n_samples = all_weights.shape[0]

    # Divide into quintiles (5 stages: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
    n_stages = 5
    stage_names = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    stage_colors = plt.cm.viridis(np.linspace(0, 1, n_stages))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Attention curves per stage
    ax1 = axes[0]
    stage_means = []

    for i in range(n_stages):
        start_idx = int(n_samples * i / n_stages)
        end_idx = int(n_samples * (i + 1) / n_stages)
        stage_weights = all_weights[start_idx:end_idx].mean(axis=0)
        stage_means.append(stage_weights)

        x = np.arange(len(stage_weights))
        ax1.plot(
            x,
            stage_weights,
            label=stage_names[i],
            color=stage_colors[i],
            linewidth=2,
            alpha=0.8,
        )

    # Add modality boundaries
    for name, (start, end) in boundaries.items():
        ax1.axvline(x=start, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    ax1.set_xlabel("Sequence Position", fontsize=11)
    ax1.set_ylabel("Mean Attention Weight", fontsize=11)
    ax1.set_title("Attention Evolution Across Trajectory Progress", fontsize=13)
    ax1.legend(title="Progress", loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Modality attention change over stages
    ax2 = axes[1]

    modality_stage_weights = {name: [] for name in boundaries.keys()}

    for i in range(n_stages):
        start_idx = int(n_samples * i / n_stages)
        end_idx = int(n_samples * (i + 1) / n_stages)
        stage_weights = all_weights[start_idx:end_idx]

        for name, (s, e) in boundaries.items():
            if e > stage_weights.shape[1]:
                e = stage_weights.shape[1]
            if s < e:
                modality_mean = stage_weights[:, s:e].mean()
                modality_stage_weights[name].append(modality_mean)

    x = np.arange(n_stages)
    width = 0.18

    for i, (name, values) in enumerate(modality_stage_weights.items()):
        offset = (i - len(modality_stage_weights) / 2 + 0.5) * width
        bars = ax2.bar(
            x + offset,
            values,
            width,
            label=name,
            color=MODALITY_COLORS.get(name, "#888888"),
            alpha=0.85,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_names)
    ax2.set_xlabel("Trajectory Progress", fontsize=11)
    ax2.set_ylabel("Mean Attention Weight", fontsize=11)
    ax2.set_title("Modality Attention Across Trajectory Stages", fontsize=13)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved temporal dynamics to {output_path}")
    plt.close()


def create_comprehensive_report(attention_data, config, output_dir):
    """
    Generate a comprehensive multi-panel visualization report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all individual plots
    print("\n=== Generating Enhanced Attention Visualizations ===\n")

    # 1. Cross-modal attention
    modality_weights, modality_stds = plot_cross_modal_attention_matrix(
        attention_data, config, output_dir / "attention_cross_modal.png"
    )

    # 2. Attention heatmap
    plot_attention_heatmap_samples(
        attention_data, config, output_dir / "attention_heatmap.png"
    )

    # 3. Entropy analysis
    plot_attention_entropy(attention_data, config, output_dir / "attention_entropy.png")

    # 4. Task-specific patterns
    plot_task_specific_attention(
        attention_data, config, output_dir / "attention_task_specific.png"
    )

    # 5. Temporal dynamics
    plot_temporal_dynamics(
        attention_data, config, output_dir / "attention_temporal_enhanced.png"
    )

    # Print summary
    print("\n" + "=" * 50)
    print("ATTENTION WEIGHT SUMMARY")
    print("=" * 50)
    for modality, weight in modality_weights.items():
        std = modality_stds[modality]
        print(f"  {modality:20s}: {weight:.6f} ± {std:.6f}")
    print("=" * 50 + "\n")

    print(f"All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Attention Weights Visualization for H-AIF"
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
        "--output-dir",
        type=str,
        default="results/attention_enhanced",
        help="Output directory for visualizations",
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
    output_dir = Path(args.output_dir)
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
    attention_data = collect_detailed_attention(
        model, val_loader, device, args.n_samples
    )
    print(f"Collected {len(attention_data['per_position_weights'])} batches")

    # Generate comprehensive report
    create_comprehensive_report(attention_data, config, output_dir)


if __name__ == "__main__":
    main()

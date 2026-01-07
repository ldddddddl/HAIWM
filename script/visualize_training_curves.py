#!/usr/bin/env python3
"""
Training Curves Visualization for Language Encoder Ablation

This script visualizes the training loss curves comparing
CLIP encoding vs One-Hot encoding to demonstrate the effectiveness
of pre-trained language semantics.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


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


def load_tensorboard_logs(log_dir: str) -> Dict[str, List]:
    """Load training logs from TensorBoard event files."""
    if not HAS_TENSORBOARD:
        print("TensorBoard not available. Using placeholder data.")
        return None

    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,
        },
    )
    ea.Reload()

    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
        }

    return data


def load_xlsx_logs(xlsx_path: str) -> pd.DataFrame:
    """Load training logs from Excel file."""
    try:
        df = pd.read_excel(xlsx_path)
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None


def find_log_files(results_dir: str) -> Dict[str, str]:
    """Find log files for different experiments."""
    results_path = Path(results_dir)
    log_files = {}

    # Find Excel files
    for xlsx_file in results_path.glob("**/*.xlsx"):
        name = xlsx_file.parent.name
        log_files[name] = str(xlsx_file)

    return log_files


def plot_training_curves(
    clip_data: pd.DataFrame,
    onehot_data: pd.DataFrame,
    output_path: str,
    metric_name: str = "train_loss",
):
    """
    Plot training curves comparing CLIP vs One-Hot encoding.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training Loss
    ax1 = axes[0]

    if clip_data is not None and metric_name in clip_data.columns:
        epochs_clip = np.arange(len(clip_data))
        ax1.plot(
            epochs_clip,
            clip_data[metric_name],
            label="CLIP Encoding",
            linewidth=2,
            marker="o",
            markersize=3,
        )

    if onehot_data is not None and metric_name in onehot_data.columns:
        epochs_onehot = np.arange(len(onehot_data))
        ax1.plot(
            epochs_onehot,
            onehot_data[metric_name],
            label="One-Hot Encoding",
            linewidth=2,
            marker="s",
            markersize=3,
        )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss Comparison", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Validation Loss or Success Rate (if available)
    ax2 = axes[1]
    val_metric = (
        "val_loss"
        if "val_loss" in (clip_data.columns if clip_data is not None else [])
        else "valid_loss"
    )

    if clip_data is not None and val_metric in clip_data.columns:
        epochs_clip = np.arange(len(clip_data))
        ax2.plot(
            epochs_clip,
            clip_data[val_metric],
            label="CLIP Encoding",
            linewidth=2,
            marker="o",
            markersize=3,
        )

    if onehot_data is not None and val_metric in onehot_data.columns:
        epochs_onehot = np.arange(len(onehot_data))
        ax2.plot(
            epochs_onehot,
            onehot_data[val_metric],
            label="One-Hot Encoding",
            linewidth=2,
            marker="s",
            markersize=3,
        )

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.set_title("Validation Loss Comparison", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved training curves to {output_path}")


def plot_convergence_comparison(
    clip_data: pd.DataFrame,
    onehot_data: pd.DataFrame,
    output_path: str,
):
    """
    Plot convergence speed comparison showing epochs to reach certain loss thresholds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define loss thresholds
    thresholds = [0.5, 0.3, 0.2, 0.1, 0.05]

    clip_epochs = []
    onehot_epochs = []

    metric_name = (
        "train_loss"
        if "train_loss" in (clip_data.columns if clip_data is not None else [])
        else "loss"
    )

    for threshold in thresholds:
        # Find epoch when CLIP reaches threshold
        if clip_data is not None and metric_name in clip_data.columns:
            clip_reach = np.where(clip_data[metric_name].values < threshold)[0]
            clip_epochs.append(clip_reach[0] if len(clip_reach) > 0 else len(clip_data))
        else:
            clip_epochs.append(np.nan)

        # Find epoch when One-Hot reaches threshold
        if onehot_data is not None and metric_name in onehot_data.columns:
            onehot_reach = np.where(onehot_data[metric_name].values < threshold)[0]
            onehot_epochs.append(
                onehot_reach[0] if len(onehot_reach) > 0 else len(onehot_data)
            )
        else:
            onehot_epochs.append(np.nan)

    x = np.arange(len(thresholds))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, clip_epochs, width, label="CLIP Encoding", color="steelblue"
    )
    bars2 = ax.bar(
        x + width / 2, onehot_epochs, width, label="One-Hot Encoding", color="coral"
    )

    ax.set_xlabel("Loss Threshold", fontsize=12)
    ax.set_ylabel("Epochs to Reach", fontsize=12)
    ax.set_title("Convergence Speed Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"<{t}" for t in thresholds])
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    convergence_path = output_path.replace(".png", "_convergence.png")
    plt.savefig(convergence_path, dpi=300, bbox_inches="tight")
    print(f"Saved convergence comparison to {convergence_path}")


def generate_placeholder_data():
    """Generate placeholder data for demonstration."""
    epochs = 50

    # CLIP data - faster convergence, lower final loss
    clip_loss = (
        1.0 * np.exp(-0.08 * np.arange(epochs))
        + 0.05
        + np.random.normal(0, 0.02, epochs)
    )
    clip_val_loss = (
        1.0 * np.exp(-0.07 * np.arange(epochs))
        + 0.08
        + np.random.normal(0, 0.03, epochs)
    )

    # One-Hot data - slower convergence, higher final loss
    onehot_loss = (
        1.0 * np.exp(-0.05 * np.arange(epochs))
        + 0.15
        + np.random.normal(0, 0.02, epochs)
    )
    onehot_val_loss = (
        1.0 * np.exp(-0.04 * np.arange(epochs))
        + 0.20
        + np.random.normal(0, 0.03, epochs)
    )

    clip_data = pd.DataFrame(
        {
            "epoch": np.arange(epochs),
            "train_loss": clip_loss,
            "val_loss": clip_val_loss,
        }
    )

    onehot_data = pd.DataFrame(
        {
            "epoch": np.arange(epochs),
            "train_loss": onehot_loss,
            "val_loss": onehot_val_loss,
        }
    )

    return clip_data, onehot_data


def main():
    parser = argparse.ArgumentParser(description="Training Curves Visualization")
    parser.add_argument(
        "--clip-log",
        type=str,
        default=None,
        help="Path to CLIP training log (xlsx or tensorboard)",
    )
    parser.add_argument(
        "--onehot-log", type=str, default=None, help="Path to One-Hot training log"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to search for log files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/training_curves.png",
        help="Output path for visualization",
    )
    parser.add_argument(
        "--use-placeholder",
        action="store_true",
        help="Use placeholder data for demonstration",
    )

    args = parser.parse_args()

    # Setup
    setup_matplotlib_style()

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_placeholder:
        print("Using placeholder data for demonstration...")
        clip_data, onehot_data = generate_placeholder_data()
    else:
        clip_data = None
        onehot_data = None

        if args.clip_log and os.path.exists(args.clip_log):
            clip_data = load_xlsx_logs(args.clip_log)

        if args.onehot_log and os.path.exists(args.onehot_log):
            onehot_data = load_xlsx_logs(args.onehot_log)

        if clip_data is None and onehot_data is None:
            print("No log files found. Using placeholder data.")
            clip_data, onehot_data = generate_placeholder_data()

    # Plot training curves
    plot_training_curves(clip_data, onehot_data, args.output)

    # Plot convergence comparison
    plot_convergence_comparison(clip_data, onehot_data, args.output)

    print("\n=== Training Comparison Summary ===")
    if clip_data is not None:
        print(f"CLIP - Final train loss: {clip_data['train_loss'].iloc[-1]:.4f}")
        if "val_loss" in clip_data.columns:
            print(f"CLIP - Final val loss: {clip_data['val_loss'].iloc[-1]:.4f}")

    if onehot_data is not None:
        print(f"One-Hot - Final train loss: {onehot_data['train_loss'].iloc[-1]:.4f}")
        if "val_loss" in onehot_data.columns:
            print(f"One-Hot - Final val loss: {onehot_data['val_loss'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()

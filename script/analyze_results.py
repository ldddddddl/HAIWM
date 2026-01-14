#!/usr/bin/env python3
"""
Quantitative Analysis for H-AIF Results
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualize_tsne import load_model, collect_latent_vectors
from visualize_attention import collect_attention_weights, analyze_modality_weights


def analyze_clustering(z_all, labels):
    """Compute clustering metrics."""
    print("\n=== Latent Space Clustering Analysis ===")

    # Check if we have enough classes
    n_labels = len(set(labels))
    if n_labels < 2:
        print("Not enough classes to compute clustering metrics.")
        return

    # Silhouette Score (-1 to 1, higher is better)
    sil_score = silhouette_score(z_all, labels)
    print(f"Silhouette Score: {sil_score:.4f} (Higher is better, >0.5 is good)")

    # Davies-Bouldin Index (lower is better)
    db_score = davies_bouldin_score(z_all, labels)
    print(f"Davies-Bouldin Index: {db_score:.4f} (Lower is better)")

    print("-" * 30)
    print("Interpretation:")
    if sil_score > 0.5:
        print("Excellent separation. The model clearly distinguishes between tasks.")
    elif sil_score > 0.2:
        print("Moderate separation. Some overlap, but structure is visible.")
    else:
        print("Poor separation. Tasks are significantly overlapping.")


def analyze_attention(attention_weights_list, config):
    """Compute detailed attention statistics."""
    print("\n=== Attention Mechanism Analysis ===")

    mean_weights, std_weights, all_weights = analyze_modality_weights(
        attention_weights_list, config
    )

    # Determine modality names same as in visualize_attention.py
    use_language = getattr(config, "use_language", False)
    n_positions = all_weights.shape[1]

    if use_language and n_positions > 3:
        segment_size = (n_positions - 1) // 3
        segments = {
            "Vision (Grip)": (0, segment_size),
            "Vision (Wrist)": (segment_size, 2 * segment_size),
            "Action": (2 * segment_size, n_positions - 1),
            "Language": (n_positions - 1, n_positions),
        }
    else:
        segment_size = n_positions // 3
        segments = {
            "Vision (Grip)": (0, segment_size),
            "Vision (Wrist)": (segment_size, 2 * segment_size),
            "Action": (2 * segment_size, n_positions),
        }

    print(f"{'Modality':<20} | {'Mean':<10} | {'Std':<10}")
    print("-" * 46)

    results = {}
    for name, (start, end) in segments.items():
        if end > start:
            # Average over the segment, then over samples
            segment_weights = all_weights[:, start:end].mean(axis=1)
            mean_val = segment_weights.mean()
            std_val = segment_weights.std()
            results[name] = mean_val
            print(f"{name:<20} | {mean_val:.4f}     | {std_val:.4f}")

    print("-" * 30)
    # Heuristics for "Goodness"
    max_modality = max(results, key=results.get)
    print(f"Dominant Modality: {max_modality} ({results[max_modality]:.4f})")

    if "Language" in results:
        lang_weight = results["Language"]
        if lang_weight < 0.05:
            print(
                "WARNING: Language weight is very low (<0.05). Model may be ignoring language."
            )
        elif lang_weight > 0.15:
            print("Language has significant influence. Good for instruction following.")

    # Sparsity Check
    # Ideal attention should be somewhat sparse (focused) rather than uniform
    # Uniform weight would be roughly 1/len(results)
    uniform_baseline = 1.0 / len(results)
    if all(abs(v - uniform_baseline) < 0.05 for v in results.values()):
        print(
            "WARNING: Attention distribution is near-uniform. Attention mechanism may not be selective enough."
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze H-AIF Results")
    parser.add_argument(
        "--checkpoint", type=str, default="../results/model_best.pth.tar"
    )
    parser.add_argument("--config", type=str, default="../config_libero.yaml")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, args.config, device)

    # Load data
    from load_libero_dataset import load_libero_dataloader

    train_loader, val_loader, normalizer = load_libero_dataloader(
        task_suite=config.task_suite,
        local_dir=config.datasets_path,
        batch_size=config.batchsize,
        horizon=config.horizon,
        past_img_num=config.past_img_num,
        future_img_num=config.future_img_num,
    )

    # 1. Analyze Latent Space (t-SNE/Clustering)
    print("\nCollecting latent vectors for clustering analysis...")
    z_all, labels = collect_latent_vectors(model, val_loader, device, args.n_samples)
    analyze_clustering(z_all, labels)

    # 2. Analyze Attention
    print("\nCollecting attention weights for analysis...")
    attention_weights_list, _ = collect_attention_weights(
        model, val_loader, device, args.n_samples
    )
    analyze_attention(attention_weights_list, config)


if __name__ == "__main__":
    main()

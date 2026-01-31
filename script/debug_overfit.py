#!/usr/bin/env python3
"""
Enhanced Overfit Debug Script

深入调试过拟合验证问题：
1. 对比训练时和验证脚本的数据处理流程
2. 打印详细的预测值和真实值
3. 验证标准化统计量是否一致

Usage:
    uv run python script/debug_overfit.py \
        --config config_libero.yaml \
        --checkpoint results/26-01-29-18-14-31/check_point/model_best.pth.tar
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from accelerate.utils import send_to_device

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.models import ActNet
from script.load_libero_dataset import LiberoDataset, LiberoNormalizer


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"[INFO] Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    print("[INFO] Creating model...")
    model = ActNet(config, is_use_cuda=device.type == "cuda", device=device)
    model = model.to(device)

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle state dict
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
                    f"[WARNING] Shape mismatch for {k}: checkpoint {v.shape} != model {model_state[k].shape}"
                )
                keys_to_remove.append(k)

    for k in keys_to_remove:
        del state_dict[k]

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[WARNING] Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys: {len(unexpected_keys)}")

    model.eval()
    return model, config


def detailed_comparison(model, dataset, config, device, num_samples=3):
    """逐样本详细对比预测值和真实值"""
    print("\n" + "=" * 60)
    print("  详细预测对比分析")
    print("=" * 60)

    model.eval()

    # 使用与训练相同的batch size
    batch_size = min(config.batchsize, len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Horizon: {config.horizon}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break

            print(f"\n{'=' * 60}")
            print(f"  Batch {batch_idx + 1}")
            print("=" * 60)

            # Move to device
            batch = send_to_device(batch, device)

            # Forward pass
            output = model(batch, phase="eval")

            # Get predictions and targets
            pred_actions = output.actions  # [B, T_pred, 7]
            target_actions = batch["action"]  # [B, T_target, 7]

            print(f"\n[Shape Info]")
            print(f"  Model output shape: {pred_actions.shape}")
            print(f"  Target action shape: {target_actions.shape}")

            # Match dimensions
            horizon = target_actions.shape[1]
            pred_matched = pred_actions[:, :horizon, :]

            print(f"  Matched pred shape: {pred_matched.shape}")

            # 第一个样本的详细分析
            pred_sample = pred_matched[0].cpu().numpy()
            target_sample = target_actions[0].cpu().numpy()

            print(f"\n[Sample 0 Statistics - First 5 timesteps]")
            print("-" * 60)
            dim_names = ["X", "Y", "Z", "RX", "RY", "RZ", "Gripper"]

            header = "  Dim   | Target Range     | Pred Range       | Mean Diff | MSE"
            print(header)
            print("-" * len(header))

            for d, name in enumerate(dim_names):
                t_min, t_max = target_sample[:, d].min(), target_sample[:, d].max()
                p_min, p_max = pred_sample[:, d].min(), pred_sample[:, d].max()
                mean_diff = np.abs(target_sample[:, d] - pred_sample[:, d]).mean()
                mse = np.mean((target_sample[:, d] - pred_sample[:, d]) ** 2)
                print(
                    f"  {name:7s}| [{t_min:+.3f}, {t_max:+.3f}] | [{p_min:+.3f}, {p_max:+.3f}] | {mean_diff:.4f}   | {mse:.4f}"
                )

            print(f"\n[First 5 Timesteps - X dimension]")
            print("  Time  | Target    | Pred      | Diff")
            print("-" * 45)
            for t in range(min(5, horizon)):
                diff = abs(target_sample[t, 0] - pred_sample[t, 0])
                print(
                    f"  {t:5d} | {target_sample[t, 0]:+.5f} | {pred_sample[t, 0]:+.5f} | {diff:.5f}"
                )

            # Check if predictions are mostly constant
            pred_std = pred_sample.std(axis=0)
            target_std = target_sample.std(axis=0)

            print(f"\n[Temporal Variation (std across time)]")
            print("  Dim     | Target Std | Pred Std | Ratio")
            print("-" * 50)
            for d, name in enumerate(dim_names):
                ratio = pred_std[d] / (target_std[d] + 1e-8)
                status = "✓" if 0.5 < ratio < 2.0 else "⚠"
                print(
                    f"  {name:8s}| {target_std[d]:.5f}   | {pred_std[d]:.5f} | {ratio:.3f} {status}"
                )

            # Overall metrics for this batch
            mse_batch = np.mean((pred_sample - target_sample) ** 2)
            mae_batch = np.mean(np.abs(pred_sample - target_sample))

            print(f"\n[Batch Metrics]")
            print(f"  MSE: {mse_batch:.6f}")
            print(f"  MAE: {mae_batch:.6f}")

    return pred_sample, target_sample


def check_normalization_consistency(dataset, config):
    """检查标准化统计量"""
    print("\n" + "=" * 60)
    print("  标准化统计量检查")
    print("=" * 60)

    normalizer = dataset.normalizer

    if "action" not in normalizer.norm_stats:
        print("[ERROR] No action normalization stats!")
        return

    stats = normalizer.norm_stats["action"]
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])

    print(f"\n[Action Normalization Stats]")
    print(f"  Shape: {mean.shape}")
    dim_names = ["X", "Y", "Z", "RX", "RY", "RZ", "Gripper"]

    print("\n  Dim     | Mean       | Std")
    print("-" * 40)
    for d, name in enumerate(dim_names):
        print(f"  {name:8s}| {mean[d]:+.6f} | {std[d]:.6f}")

    # Check for problematic std values
    print("\n[Potential Issues]")
    for d, name in enumerate(dim_names):
        if std[d] < 1e-6:
            print(
                f"  ⚠ {name}: Std is very small ({std[d]:.2e}), may cause numerical issues"
            )
        if std[d] < 1e-3:
            print(
                f"  △ {name}: Std is small ({std[d]:.4f}), normalized values may be large"
            )


def visualize_trajectories(pred, target, output_path):
    """可视化预测和真实轨迹"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    dim_names = ["X", "Y", "Z", "RX", "RY", "RZ", "Gripper"]

    for i, (ax, name) in enumerate(zip(axes.flat[:7], dim_names)):
        ax.plot(target[:, i], "b-", label="Ground Truth", linewidth=2, alpha=0.8)
        ax.plot(pred[:, i], "r--", label="Prediction", linewidth=2, alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value (Normalized)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # R²
        ss_res = np.sum((target[:, i] - pred[:, i]) ** 2)
        ss_tot = np.sum((target[:, i] - np.mean(target[:, i])) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        ax.set_title(f"{name} (R²={r2:.4f})", fontsize=10)

    # Summary
    ax_sum = axes.flat[7]
    ax_sum.axis("off")

    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    summary = f"Overall Metrics:\nMSE: {mse:.6f}\nMAE: {mae:.6f}"
    ax_sum.text(
        0.5,
        0.5,
        summary,
        ha="center",
        va="center",
        fontsize=12,
        transform=ax_sum.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.suptitle("Debug: Prediction vs Ground Truth", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Debug plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Debug Overfit Verification")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", "-p", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", "-o", type=str, default="./results")

    args = parser.parse_args()

    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    model, config = load_model(args.config, args.checkpoint, device)

    # Load dataset
    data_dir = Path(config.datasets_path)
    print(f"\n[INFO] Loading dataset from: {data_dir}")
    dataset = LiberoDataset(
        data_dir=data_dir,
        task_suite=getattr(config, "task_suite", "libero_10"),
        horizon=config.horizon,
        past_img_num=config.past_img_num,
        future_img_num=config.future_img_num,
        image_size=config.cropWidth,
        normalize=True,
    )

    print(f"[INFO] Dataset: {len(dataset)} samples")

    # Check normalization
    check_normalization_consistency(dataset, config)

    # Detailed comparison
    pred, target = detailed_comparison(model, dataset, config, device, num_samples=2)

    # Save visualization
    output_path = Path(args.output) / "debug_overfit.png"
    visualize_trajectories(pred, target, output_path)

    print("\n" + "=" * 60)
    print("  调试完成")
    print("=" * 60)
    print("\n诊断建议:")
    print("  1. 如果 Pred Std 很小但 Target Std 正常 → 模型预测接近常数")
    print("  2. 如果 Mean Diff 很大 → 可能是偏置问题或数据不匹配")
    print("  3. 如果 R² 为负 → 预测比均值更差，需要检查模型或数据")


if __name__ == "__main__":
    main()

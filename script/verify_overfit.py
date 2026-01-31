#!/usr/bin/env python3
"""
Overfit Verification Script

验证模型在训练数据上是否能过拟合，以确认数据处理和模型框架的正确性。

如果MSE接近0 → 框架正确，问题在online评估
如果MSE很大 → 数据处理或模型有问题

Usage:
    uv run python script/verify_overfit.py \
        --config config_libero.yaml \
        --checkpoint results/model_best_test.pth.tar \
        --data-dir datasets/libero/libero_test
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
from tqdm import tqdm

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
                    f"[WARNING] Shape mismatch for {k}: checkpoint {v.shape} != model {model_state[k].shape}. Ignoring."
                )
                keys_to_remove.append(k)

    for k in keys_to_remove:
        del state_dict[k]

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[WARNING] Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys: {unexpected_keys[:5]}...")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model loaded. Total parameters: {total_params:,}")

    return model, config


def verify_normalization(normalizer: LiberoNormalizer, sample_action: torch.Tensor):
    """验证标准化和反标准化的一致性."""
    print("\n" + "=" * 50)
    print("  标准化一致性检验")
    print("=" * 50)

    if "action" not in normalizer.norm_stats:
        print("[WARNING] No action normalization stats found!")
        return False

    # 反标准化后再标准化应该得到原始值
    denorm = normalizer.denormalize(sample_action, "action")
    renorm = normalizer.normalize(denorm, "action")

    diff = torch.abs(sample_action - renorm).max().item()
    print(f"  原始值范围: [{sample_action.min():.4f}, {sample_action.max():.4f}]")
    print(f"  反标准化后范围: [{denorm.min():.4f}, {denorm.max():.4f}]")
    print(f"  重新标准化后差异: {diff:.8f}")

    # 打印标准化统计
    stats = normalizer.norm_stats["action"]
    print(f"  均值: {np.array(stats['mean'])}")
    print(f"  标准差: {np.array(stats['std'])}")

    is_ok = diff < 1e-5
    print(f"  结果: {'✓ 通过' if is_ok else '✗ 失败'}")
    return is_ok


@torch.no_grad()
def verify_overfit(
    model: ActNet,
    dataset: LiberoDataset,
    config,
    device: torch.device,
    output_dir: Path,
):
    """验证模型在训练数据上的过拟合情况."""
    print("\n" + "=" * 50)
    print("  过拟合验证")
    print("=" * 50)

    model.eval()

    # Metrics
    all_mse = []
    all_mae = []
    all_pred = []
    all_target = []
    dim_mse = [[] for _ in range(7)]  # 7D action

    # Create dataloader with batch size = 1 for detailed analysis
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.batchsize, len(dataset)),
        shuffle=False,
        num_workers=0,
    )

    print(f"[INFO] Dataset size: {len(dataset)} samples")
    print(f"[INFO] Number of batches: {len(dataloader)}")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        # Move to device
        batch = send_to_device(batch, device)

        # Forward pass
        output = model(batch, phase="eval")

        # Get predictions and targets
        pred_actions = output.actions  # [B, T, 7]
        target_actions = batch["action"]  # [B, T, 7]

        # Match dimensions
        min_t = min(pred_actions.shape[1], target_actions.shape[1])
        pred = pred_actions[:, :min_t, :].cpu().numpy()
        target = target_actions[:, :min_t, :].cpu().numpy()

        # Compute metrics
        mse = np.mean((pred - target) ** 2)
        mae = np.mean(np.abs(pred - target))

        all_mse.append(mse)
        all_mae.append(mae)

        # Collect for plotting (first sample of each batch)
        all_pred.append(pred[0])
        all_target.append(target[0])

        # Per-dimension MSE
        for d in range(7):
            dim_mse[d].append(np.mean((pred[..., d] - target[..., d]) ** 2))

    # Compute final metrics
    final_mse = np.mean(all_mse)
    final_mae = np.mean(all_mae)
    final_rmse = np.sqrt(final_mse)

    print("\n" + "-" * 40)
    print("过拟合验证结果:")
    print("-" * 40)
    print(f"  MSE:  {final_mse:.6f}")
    print(f"  MAE:  {final_mae:.6f}")
    print(f"  RMSE: {final_rmse:.6f}")

    # Per-dimension metrics
    dim_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    print("\n  每维度MSE:")
    for i, name in enumerate(dim_names):
        print(f"    {name}: {np.mean(dim_mse[i]):.6f}")

    # Gripper accuracy
    all_pred_cat = np.concatenate(all_pred, axis=0)
    all_target_cat = np.concatenate(all_target, axis=0)
    gripper_pred = (all_pred_cat[..., -1] > 0).astype(float)
    gripper_target = (all_target_cat[..., -1] > 0).astype(float)
    gripper_acc = np.mean(gripper_pred == gripper_target)
    print(f"\n  Gripper Accuracy: {gripper_acc:.4f}")

    # Diagnosis
    print("\n" + "-" * 40)
    print("诊断结论:")
    print("-" * 40)
    if final_mse < 0.01:
        print("  ✓ MSE < 0.01: 模型能够过拟合训练数据")
        print("  → 数据处理和模型框架正确")
        print("  → 问题可能在online评估或动作反标准化")
    elif final_mse < 0.1:
        print("  △ 0.01 < MSE < 0.1: 模型部分拟合")
        print("  → 可能需要更多训练或检查数据")
    else:
        print("  ✗ MSE > 0.1: 模型无法拟合训练数据")
        print("  → 数据处理或模型配置可能有问题")

    # Create visualization
    _plot_predictions(all_pred, all_target, output_dir / "overfit_check.png")

    return {
        "mse": final_mse,
        "mae": final_mae,
        "rmse": final_rmse,
        "gripper_accuracy": gripper_acc,
    }


def _plot_predictions(all_pred, all_target, output_path: Path):
    """绘制预测值vs真实值对比图."""
    # Configure matplotlib for Chinese fonts
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 8  # Reduce default font size
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7

    # Concatenate all predictions
    pred = np.concatenate(all_pred, axis=0)  # [T_total, 7]
    target = np.concatenate(all_target, axis=0)  # [T_total, 7]

    # Only plot first 200 timesteps for clarity
    max_steps = min(200, len(pred))
    pred = pred[:max_steps]
    target = target[:max_steps]

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    dim_names = [
        "X (Position)",
        "Y (Position)",
        "Z (Position)",
        "RX (Rotation)",
        "RY (Rotation)",
        "RZ (Rotation)",
        "Gripper",
    ]

    for i, (ax, name) in enumerate(zip(axes.flat[:7], dim_names)):
        ax.plot(target[:, i], "b-", label="Ground Truth", linewidth=1.2, alpha=0.8)
        ax.plot(pred[:, i], "r--", label="Prediction", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Timestep", fontsize=7)
        ax.set_ylabel("Value (Normalized)", fontsize=7)
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Compute R² for this dimension
        ss_res = np.sum((target[:, i] - pred[:, i]) ** 2)
        ss_tot = np.sum((target[:, i] - np.mean(target[:, i])) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        ax.set_title(f"{name} (R²={r2:.4f})", fontsize=9, fontweight="bold")

    # Hide the 8th subplot and add summary info
    ax_summary = axes.flat[7]
    ax_summary.axis("off")

    # Add summary text
    overall_mse = np.mean((pred - target) ** 2)
    overall_mae = np.mean(np.abs(pred - target))
    summary_text = f"Overall Metrics:\nMSE: {overall_mse:.6f}\nMAE: {overall_mae:.6f}\nSamples: {len(pred)}"
    ax_summary.text(
        0.5,
        0.5,
        summary_text,
        ha="center",
        va="center",
        fontsize=10,
        transform=ax_summary.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.suptitle(
        "Overfit Verification: Prediction vs Ground Truth",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Prediction comparison saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="验证模型过拟合能力")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to config yaml file"
    )
    parser.add_argument(
        "--checkpoint", "-p", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir", "-d", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--output", "-o", type=str, default="./results", help="Output directory"
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    model, config = load_model(args.config, args.checkpoint, device)

    # Load dataset (same as training, no split)
    print(f"\n[INFO] Loading dataset from: {args.data_dir}")
    dataset = LiberoDataset(
        data_dir=Path(args.data_dir),
        task_suite=getattr(config, "task_suite", "libero_10"),
        horizon=config.horizon,
        past_img_num=config.past_img_num,
        future_img_num=config.future_img_num,
        image_size=config.cropWidth,
        normalize=True,  # 必须使用标准化
    )

    print(f"[INFO] Loaded {len(dataset)} samples from {len(dataset.hdf5_files)} files")

    # Verify normalization
    sample = dataset[0]
    verify_normalization(dataset.normalizer, sample["action"])

    # Verify overfit
    output_dir = Path(args.output)
    results = verify_overfit(model, dataset, config, device, output_dir)

    print("\n" + "=" * 50)
    print("  验证完成")
    print("=" * 50)


if __name__ == "__main__":
    main()

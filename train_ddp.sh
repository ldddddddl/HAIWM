#!/bin/bash
# HAIWM 多卡分布式训练启动脚本
# 使用方法: ./train_ddp.sh [--config config_libero.yaml] [其他参数...]
#
# 示例:
#   ./train_ddp.sh --config config_libero.yaml
#   NUM_GPUS=2 ./train_ddp.sh --config config.yaml
#
# 环境变量:
#   NUM_GPUS: 指定使用的GPU数量（默认自动检测）
#   MASTER_PORT: 分布式训练主端口（默认29500）
#   TORCH_CUDA_ARCH_LIST: 指定GPU架构（如A100设置为"8.0"）

set -e

# 自动检测GPU数量（若未指定）
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "错误: 未检测到GPU，请确保CUDA可用" >&2
        exit 1
    fi
fi

echo "=================================================="
echo "HAIWM 多卡分布式训练"
echo "检测到 ${NUM_GPUS} 个GPU"
if [ -n "$TORCH_CUDA_ARCH_LIST" ]; then
    echo "CUDA架构: ${TORCH_CUDA_ARCH_LIST}"
fi
echo "=================================================="

# 设置主端口
MASTER_PORT=${MASTER_PORT:-29500}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 使用torchrun启动分布式训练
# --nproc_per_node: 每个节点的进程数（即GPU数）
# --master_port: 分布式通信端口
# --use_ddp: 启用DDP模式
exec torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train.py \
    "$@" \
    --use_ddp

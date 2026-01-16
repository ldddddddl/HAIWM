#!/bin/bash
# HAIWM 多卡分布式训练启动脚本
# 使用方法: ./train_ddp.sh [--config config_libero.yaml] [其他参数...]
#
# 示例:
#   ./train_ddp.sh --config config_libero.yaml
#   NUM_GPUS=2 ./train_ddp.sh --config config_libero.yaml
#   GPU_IDS="0,1" ./train_ddp.sh --config config_libero.yaml
#   GPU_IDS="2,3,4,5" ./train_ddp.sh --config config_libero.yaml
#
# 环境变量:
#   GPU_IDS: 指定使用的GPU ID（如"0,1"或"2,3,4,5"），默认使用所有GPU
#   NUM_GPUS: 指定使用的GPU数量（如果设置了GPU_IDS则自动计算）
#   MASTER_PORT: 分布式训练主端口（默认29500）
#   TORCH_CUDA_ARCH_LIST: 指定GPU架构（如A100设置为"8.0"）

set -e

# 处理GPU选择
if [ -n "$GPU_IDS" ]; then
    # 如果指定了GPU_IDS，设置CUDA_VISIBLE_DEVICES并计算GPU数量
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    # 计算逗号分隔的GPU数量
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
else
    # 自动检测GPU数量（若未指定）
    if [ -z "$NUM_GPUS" ]; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
        if [ "$NUM_GPUS" -eq 0 ]; then
            echo "错误: 未检测到GPU，请确保CUDA可用" >&2
            exit 1
        fi
    fi
fi

echo "=================================================="
echo "HAIWM 多卡分布式训练"
if [ -n "$GPU_IDS" ]; then
    echo "使用GPU: ${GPU_IDS} (共 ${NUM_GPUS} 个)"
else
    echo "使用全部 ${NUM_GPUS} 个GPU"
fi
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
# 使用 uv run 确保在虚拟环境中正确运行
exec uv run python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train.py \
    "$@" \
    --use_ddp

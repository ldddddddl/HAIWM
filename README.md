# HAIWM: Hierarchical Active Inference World Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

HAIWM是一个基于主动推理理论的具身智能机器人世界模型，结合了 xLSTM 架构和多模态融合技术，用于机器人操作任务的学习和执行。

## 📔 To do list

- **[·]**: 图像生成
- **[·]**: More, like attention...

## 🌟 主要特性

- **主动推理架构**: 基于自由能最小化原理的层级决策框架
- **多模态融合**: 整合视觉、本体感觉和语言模态的注意力机制
- **语言条件控制**: 基于 CLIP 的语言编码器，支持自然语言任务指令
- **xLSTM 骨干网络**: 使用扩展 LSTM 进行时序建模
- **LIBERO 基准支持**: 完整支持 LIBERO 机器人操作基准测试

## 📁 项目结构

```
H-AIF/
├── model/
│   ├── models.py          # ActNet 主模型
│   ├── language_encoder.py # CLIP/OneHot 语言编码器
│   ├── baseline_bc.py     # Baseline 模型 (BC-RNN, BC-Transformer)
│   └── ...
├── script/
│   ├── visualize_tsne.py      # t-SNE 潜在空间可视化
│   ├── visualize_attention.py # 注意力权重可视化
│   └── evaluate_success_rate.py # 成功率评估
├── train.py               # 训练入口
├── losses.py              # 损失函数
├── config_libero.yaml     # LIBERO 数据集配置
├── config.yaml            # JetMax 数据集配置
└── README.md
```

## 🚀 快速开始

### 1. 环境安装

推荐使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或
wget -qO- https://astral.sh/uv/install.sh | sh
```

### 2. 克隆仓库

```bash
# 克隆时同步子模块
git clone --recurse-submodules https://github.com/ldddddddl/H-AIF.git
cd HAIWM

# 如果已克隆，更新子模块
git submodule update --init --recursive
```

### 3. 安装依赖

```bash
# 同步依赖 (跳过 LFS 大文件)
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 可选: 以可编辑模式安装
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

> **注意**: `GIT_LFS_SKIP_SMUDGE=1` 用于跳过 LeRobot 的大文件下载。

### 4. 下载数据集

#### LIBERO 数据集

数据集会在首次运行时自动从 HuggingFace 下载：

```bash
# 数据集将下载到 datasets/libero/ 目录
uv run python train.py --config config_libero.yaml
```

支持的数据集套件：
- `libero_10`: 10 个任务 (推荐入门)
- `libero_90`: 90 个任务
- `libero_spatial`: 空间推理任务
- `libero_object`: 物体操作任务
- `libero_goal`: 目标导向任务

#### 手动下载 LIBERO-100 完整数据集

如需下载完整的 LIBERO-100 数据集，请使用以下命令：

```bash
# 安装 huggingface_hub
uv pip install huggingface_hub
```

```bash
# 下载 LIBERO-100 数据集到指定目录
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='libero-project/LIBERO',
    repo_type='dataset',
    local_dir='./datasets/libero',
    local_dir_use_symlinks=False,
    allow_patterns=['libero_10/*', 'libero_90/*'],
)
print('下载完成！')
"
```

下载后的目录结构：
```
datasets/libero/
├── libero_10/
├── libero_90/
├── libero_spatial/
├── libero_object/
└── libero_goal/
```

## ⚙️ 配置说明

### config_libero.yaml 主要参数

```yaml
# 基础配置
name: xlstm_libero
epochs: 200
lr: 3.0e-5
batchsize: 8

# 数据集配置
use_libero: True
task_suite: libero_10  # 数据集套件
datasets_path: ./datasets/libero/libero_10

# 语言模态配置
use_language: True
language_encoder_type: "clip"  # "clip" 或 "onehot" (消融实验)
clip_model: "ViT-B/32"

# 模型配置
horizon: 50           # 动作预测时域
action_dim: 7         # 动作维度 (6D位姿 + 1D夹爪)
past_img_num: 5       # 历史图像帧数
future_img_num: 5     # 预测图像帧数

# 损失权重
alpha_loss:
    actions: 2500.0   # 动作预测损失
    sucker: 500.0     # 夹爪动作损失 (末端执行器)
    kl: 500.0         # KL 散度损失
    frames: 6.0       # 图像预测损失
```

### 消融实验配置

```yaml
# 1. 不使用语言模态
use_language: False

# 2. 使用 One-Hot 编码 (对比 CLIP)
use_language: True
language_encoder_type: "onehot"
```

## 🏃 训练

### 基础训练

```bash
uv run python train.py --config config_libero.yaml
```

### 使用 GPU

```bash
# 指定 GPU
CUDA_VISIBLE_DEVICES=0 uv run python train.py --config config_libero.yaml
```

### 多卡分布式训练 (DDP)

```bash
# 使用启动脚本（自动检测GPU数量）
./train_ddp.sh --config config_libero.yaml

# 或指定GPU数量
NUM_GPUS=2 ./train_ddp.sh --config config_libero.yaml
```

### ⚠️ 服务器运行常见问题

#### CUDA 架构编译错误

如果遇到以下错误：
```
nvcc fatal: Unsupported gpu architecture 'compute_89'
```

这是因为 xLSTM 的 sLSTM CUDA 扩展在编译时需要正确的 GPU 架构。请按以下步骤解决：

**1. 查看服务器 GPU 的 Compute Capability：**
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# 或
python -c "import torch; print(torch.cuda.get_device_capability())"
```

**2. 查看 nvcc 支持的最高架构版本：**
```bash
nvcc --version
```

| CUDA Toolkit 版本 | 支持的最高架构 |
|------------------|---------------|
| CUDA 11.1-11.7   | 8.6 (sm_86)   |
| CUDA 11.8+       | 8.9 (sm_89)   |
| CUDA 12.0+       | 9.0 (sm_90)   |

**3. 选择正确的 `TORCH_CUDA_ARCH_LIST` 值：**

> ⚠️ **重要**: 设置值应取 **GPU Compute Capability** 和 **nvcc 支持的最高版本** 中的 **较小值**。

例如：RTX 4090 (8.9) + CUDA 11.7 (最高支持 8.6) → 使用 `8.0` 或 `8.6`

| GPU 型号 | Compute Capability | CUDA 11.7 设置 | CUDA 11.8+ 设置 |
|----------|-------------------|----------------|----------------|
| A100     | 8.0               | `8.0`          | `8.0`          |
| RTX 3090 | 8.6               | `8.6`          | `8.6`          |
| RTX 4090 | 8.9               | `8.0`          | `8.9`          |
| H100     | 9.0               | `8.0`          | `9.0`          |

**4. 清除缓存并重新运行：**
```bash
# 清除 PyTorch 扩展缓存
rm -rf ~/.cache/torch_extensions/

# 设置环境变量并运行
TORCH_CUDA_ARCH_LIST="8.0" uv run python train.py --config config_libero.yaml

# 或添加到 ~/.bashrc 永久生效
echo 'export TORCH_CUDA_ARCH_LIST="8.0"' >> ~/.bashrc
source ~/.bashrc
```

> **提示**: 如果服务器有多种 GPU，可以设置多个架构：`TORCH_CUDA_ARCH_LIST="8.0;8.6"`
>
> **注意**: 使用较低架构编译（如在 RTX 4090 上使用 8.0）会通过 PTX JIT 编译运行，功能正常但可能略有性能损失。

### 训练输出

训练结果保存在 `results/` 目录：
```
results/
└── 26-01-07-15-30-00/
    ├── config.yaml           # 训练配置
    ├── 26-01-07-15-30-00.xlsx # 训练日志
    └── check_point/
        └── model_*.pth.tar   # 模型权重
```

## 📊 可视化与评估

### t-SNE 潜在空间可视化

```bash
uv run python script/visualize_tsne.py \
    --checkpoint results/*/check_point/model_*.pth.tar \
    --config config_libero.yaml \
    --output results/tsne.png
```

### 注意力权重可视化

```bash
uv run python script/visualize_attention.py \
    --checkpoint results/*/check_point/model_*.pth.tar \
    --output results/attention.png
```

### 成功率评估

```bash
uv run python script/evaluate_success_rate.py \
    --output results/success_rate.png \
    --use-placeholder  # 使用示例数据演示
```

## 🔬 模型架构

```
输入
 ├── 视觉: Top Camera + Wrist Camera [B, T, 3, 112, 112]
 ├── 本体感觉: 关节状态 [B, 7]
 └── 语言: 任务指令 "pick up the red cube"
      ↓
┌─────────────────────────────────────────┐
│           CLIPLanguageEncoder           │
│     (ViT-B/32 → 512D → 120D 投影)       │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│         MultiModalFusionModel           │
│  ┌────────┬────────┬────────┬────────┐ │
│  │ Vision │ Vision │ Action │  Lang  │ │
│  │ (Grip) │ (Side) │        │        │ │
│  └────────┴────────┴────────┴────────┘ │
│              ↓ Attention                │
│         加权融合 + 残差连接             │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│              xLSTM Backbone             │
│         (mLSTM + sLSTM blocks)          │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│           主动推理层 (Critic)           │
│       State Loss → Weights + Bias       │
└─────────────────────────────────────────┘
      ↓
输出
 ├── 动作序列: [B, horizon, 7]
 ├── 夹爪动作: [B, T, 2]
 └── (可选) 未来图像帧预测
```

## 📝 变量命名说明

| 变量名 | 含义 |
|-------|------|
| `gripper` / `sucker` | 末端执行器动作 (夹爪/吸盘) |
| `acc_gripper` | 末端执行器动作分类准确率 |
| `actions` | 机器人关节/末端位姿动作 |
| `z_mix` | 融合后的潜在变量 |
| `attention_weights` | 多模态注意力权重 |

## 📚 参考文献

- [Active Inference](https://en.wikipedia.org/wiki/Free_energy_principle)
- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- [LIBERO Benchmark](https://libero-project.github.io/)
- [CLIP](https://openai.com/research/clip)

## 📄 License

MIT License

## 🙏 致谢

感谢以下开源项目：
- [LeRobot](https://github.com/huggingface/lerobot)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [xlstm](https://github.com/NX-AI/xlstm)
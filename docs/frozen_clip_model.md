# Frozen CLIP 模型文档

基于冻结 CLIP ViT-L/14 的媒体质量评分模型。

---

## 模型介绍

### 架构概述

Frozen CLIP 模型是一个轻量级的媒体质量评分系统，使用冻结的 CLIP 视觉编码器提取特征，配合可训练的时间注意力机制和评分头预测 0-9 区间的质量分数。

```
┌─────────────────────────────────────────────────────────────┐
│                     Frozen CLIP 模型                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入媒体 (图像/视频)                                        │
│       │                                                     │
│       ▼                                                     │
│  ┌───────────────────────────────────────┐                  │
│  │  FrozenCLIPEncoder (冻结)              │                  │
│  │  - 模型: openai/clip-vit-large-patch14 │                  │
│  │  - 输出: 768-dim 特征向量               │                  │
│  │  - 参数: ~400M (冻结)                  │                  │
│  └───────────────────────────────────────┘                  │
│       │                                                     │
│       ▼                                                     │
│  ┌───────────────────────────────────────┐                  │
│  │  TemporalAttention (可训练)            │                  │
│  │  - 多头自注意力聚合帧特征               │                  │
│  │  - 可学习位置编码                       │                  │
│  │  - 参数: ~800K                         │                  │
│  └───────────────────────────────────────┘                  │
│       │                                                     │
│       ▼                                                     │
│  ┌───────────────────────────────────────┐                  │
│  │  ScoreHead (可训练)                    │                  │
│  │  - MLP: 768 → 256 → 64 → 1            │                  │
│  │  - 输出: 0-9 范围评分                 │                  │
│  │  - 参数: ~220K                         │                  │
│  └───────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 说明 | 参数量 |
|------|------|--------|
| FrozenCLIPEncoder | CLIP ViT-L/14 视觉编码器（冻结） | ~400M (冻结) |
| TemporalAttention | 多头时间注意力聚合 | ~800K |
| ScoreHead | MLP 评分预测头 | ~220K |
| **总可训练参数** | | **~1M** |

### 支持的媒体格式

**图像**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`

**视频**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

### CLIP 主干模型候选（理论可用）

基于当前实现（`src/training/frozen_clip_encoder.py` 使用 `transformers.CLIPModel`）：

当前仓库默认使用 `openai/clip-vit-large-patch14`，以下模型按当前代码路径属于理论可用项（可通过 `--clip_model_name` 或配置项 `clip_model_name` 指定）：

- `openai/clip-vit-base-patch16`
- `openai/clip-vit-base-patch32`
- `openai/clip-vit-large-patch14`
- `openai/clip-vit-large-patch14-336`
- `laion/CLIP-ViT-B-16-laion2b-s34b-b88-k-m`
- `laion/CLIP-ViT-B-32-laion2b-s34b-b88-k-m`
- `laion/CLIP-ViT-L-14-laion2b-s32b-b82k`
- `laion/CLIP-ViT-H-14-laion2b-s32b-b79k`

说明：

- 上述模型为理论可用模型名；实际可用性取决于当前环境对该 checkpoint 的下载/缓存与加载成功情况。
- 大模型（如 `ViT-L`、`ViT-H`）通常显著提高显存消耗，请结合 `clip_batch_size`、`precision` 与 DeepSpeed 进行资源规划。
- 若加载失败，请参考“如何使用预训练 CLIP 模型”使用镜像或代理重试。

### 评分范围

输出分数范围：**[0.0, 9.0]**

- 0-3: 低质量
- 4-6: 中等质量
- 7-9: 高质量

---

## 安装

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)

### 依赖安装

```bash
# 创建虚拟环境
conda create -n frozen_clip python=3.10
conda activate frozen_clip

# 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install transformers>=4.30.0
pip install pillow
pip install numpy
pip install scipy  # 用于 Pearson 相关性计算

# 可选：DeepSpeed 分布式训练
pip install deepspeed>=0.14.0
```

### 验证安装

```bash
python -c "
from src.training.frozen_clip_encoder import FrozenCLIPEncoder
from src.training.temporal_attention import TemporalAttention
from src.training.score_head import ScoreHead
print('All components imported successfully!')
"
```

---

## 训练

### 数据格式

训练数据使用 JSON Lines 格式 (`labels.json`)：

```json
{"media_path": "/path/to/image.jpg", "score": 7.5, "prompt": ""}
{"media_path": "/path/to/video.mp4", "score": 5.2, "prompt": ""}
```

### 单 GPU 训练

```bash
# 基础训练
python scripts/train_frozen_clip.py \
    --labels_path labels.json \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --output_dir checkpoints/frozen_clip

# 自定义配置
python scripts/train_frozen_clip.py \
    --labels_path labels.json \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_frames 12 \
    --num_heads 8 \
    --temporal_dim 256 \
    --dropout 0.1 \
    --loss_type mse \
    --output_dir checkpoints/frozen_clip_v2
```

### 多 GPU 分布式训练 (DeepSpeed)

```bash
# 4 GPU DeepSpeed ZeRO-2 训练
torchrun --nproc_per_node=4 scripts/train_frozen_clip.py \
    --labels_path labels.json \
    --deepspeed_config configs/ds_frozen_clip.json \
    --batch_size 8 \
    --epochs 10 \
    --output_dir checkpoints/frozen_clip_ds

# 2 GPU + CPU Offload（显存有限时）
torchrun --nproc_per_node=2 scripts/train_frozen_clip.py \
    --labels_path labels.json \
    --deepspeed_config configs/ds_frozen_clip_v100_fp16.json \
    --batch_size 4 \
    --epochs 10
```

### 当前推荐配置（2026-03-27）

当前仓库保留的推荐基线为：

- `checkpoints/checkpoint_best.pt`
- `epoch = 3`
- `val_loss = 1.8671458218548749`

推荐训练命令（4x V100 16GB，使用仓库内保留的 DeepSpeed 配置）：

```bash
torchrun --nproc_per_node=4 scripts/train_frozen_clip.py \
    --labels_path labels.json \
    --deepspeed_config configs/ds_frozen_clip.json \
    --val_ratio 0.2 \
    --num_frames 16 \
    --long_video_strategy expand \
    --max_long_frames 32 \
    --score_min 0 \
    --score_max 9 \
    --epochs 8 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 2 \
    --seed 42 \
    --precision fp16 \
    --clip_batch_size 8 \
    --clip_model_name openai/clip-vit-large-patch14-336 \
    --unfreeze_last_n_vision_layers 2 \
    --num_heads 8 \
    --temporal_dim 256 \
    --dropout 0.1 \
    --max_frames 32 \
    --loss_type mse \
    --num_workers 2 \
    --prefetch_factor 2 \
    --output_dir outputs/frozen_clip_v100_4gpu_clip336_u2_nf16_mf32_b2_acc2_fp16_e8
```

继续训练时优先从 `checkpoint_best.pt` 恢复，而不是 `checkpoint_latest.pt`：

```bash
torchrun --nproc_per_node=4 scripts/train_frozen_clip.py \
    --labels_path labels.json \
    --deepspeed_config configs/ds_frozen_clip.json \
    --val_ratio 0.2 \
    --num_frames 16 \
    --long_video_strategy expand \
    --max_long_frames 32 \
    --score_min 0 \
    --score_max 9 \
    --epochs 8 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 2 \
    --seed 42 \
    --precision fp16 \
    --clip_batch_size 8 \
    --clip_model_name openai/clip-vit-large-patch14-336 \
    --unfreeze_last_n_vision_layers 2 \
    --num_heads 8 \
    --temporal_dim 256 \
    --dropout 0.1 \
    --max_frames 32 \
    --loss_type mse \
    --num_workers 2 \
    --prefetch_factor 2 \
    --output_dir outputs/frozen_clip_v100_4gpu_clip336_u2_nf16_mf32_b2_acc2_fp16_e8_resume \
    --resume outputs/frozen_clip_v100_4gpu_clip336_u2_nf16_mf32_b2_acc2_fp16_e8_20260326/checkpoint_best.pt
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--labels_path` | `labels.json` | 标签文件路径 |
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 16 | 每GPU批次大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--gradient_accumulation_steps` | 1 | 梯度累积步数 |
| `--precision` | fp32 | 训练精度 (`fp32/fp16/bf16`) |
| `--clip_batch_size` | 64 | CLIP 编码阶段的分块 batch size |
| `--clip_model_name` | `openai/clip-vit-large-patch14` | CLIP 主干模型 |
| `--unfreeze_last_n_vision_layers` | 0 | 解冻最后 N 层视觉层 |
| `--num_frames` | 8 | 视频采样帧数 |
| `--long_video_strategy` | expand | 长视频处理策略 |
| `--max_long_frames` | 32 | 长视频最大展开帧数 |
| `--num_heads` | 8 | 注意力头数 |
| `--temporal_dim` | 256 | 时间投影维度 |
| `--dropout` | 0.1 | Dropout率 |
| `--max_frames` | 32 | 最大帧数（位置编码） |
| `--loss_type` | mse | 损失函数 (mse/smooth_l1) |
| `--val_ratio` | 0.2 | 验证集比例 |
| `--output_dir` | checkpoints/frozen_clip | 输出目录 |
| `--deepspeed_config` | None | DeepSpeed配置文件 |

### 从检查点恢复训练

```bash
python scripts/train_frozen_clip.py \
    --labels_path labels.json \
    --resume checkpoints/frozen_clip/checkpoint_epoch_5.pt
```

---

## 推理

### 基本用法

```python
from src.models.frozen_clip_engine import FrozenClipEngine, create_engine

# 方式1: 分步加载
engine = FrozenClipEngine(device="cuda", num_frames=12)
engine.load_model("checkpoints/frozen_clip/checkpoint_best.pt")

# 方式2: 便捷函数
engine = create_engine("checkpoints/frozen_clip/checkpoint_best.pt", device="cuda")

# 评分图像
score = engine.score_image("path/to/image.jpg")
print(f"Image score: {score:.2f}")

# 评分视频
score = engine.score_video("path/to/video.mp4")
print(f"Video score: {score:.2f}")

# 自动检测媒体类型
result = engine.score("path/to/media.mp4")
print(f"Type: {result['media_type']}, Score: {result['score']:.2f}")
```

### 批量推理

```python
# 批量评分多个文件
media_paths = [
    "path/to/image1.jpg",
    "path/to/image2.png",
    "path/to/video1.mp4",
]

results = engine.score_batch(media_paths)

for result in results:
    if result.get("error"):
        print(f"Error: {result['media_path']} - {result['error']}")
    else:
        print(f"{result['media_path']}: {result['score']:.2f}")
```

### 获取模型信息

```python
info = engine.get_model_info()
print(f"Device: {info['device']}")
print(f"CLIP Model: {info['clip_model']}")
print(f"Feature Dim: {info['clip_feature_dim']}")
print(f"Trainable Params: {info['total_trainable_params']:,}")
```

### 命令行推理

```bash
# 快速测试
python src/models/frozen_clip_engine.py checkpoints/frozen_clip/checkpoint_best.pt path/to/media.jpg
```

---

## 模型架构详解

### FrozenCLIPEncoder

冻结的 CLIP 视觉编码器，用于提取 768 维特征向量。

```python
from src.training.frozen_clip_encoder import FrozenCLIPEncoder
from PIL import Image

encoder = FrozenCLIPEncoder(device="cuda")

# 单图特征提取
image = Image.open("photo.jpg")
features = encoder.extract_features(image)  # (1, 768)

# 多帧特征提取（视频）
frames = [Image.open(f"frame_{i}.jpg") for i in range(12)]
features = encoder.extract_features(frames)  # (12, 768)
```

### TemporalAttention

时间注意力模块，用于聚合多帧特征。

```python
from src.training.temporal_attention import TemporalAttention
import torch

attention = TemporalAttention(
    feature_dim=768,
    num_heads=8,
    temporal_dim=256,
    max_frames=32,
)

# 视频帧特征 (batch_size=4, frames=12, dim=768)
video_features = torch.randn(4, 12, 768)
attended = attention(video_features)  # (4, 12, 768)

# 聚合为单个向量
aggregated = attention.aggregate_temporal(attended, method="mean")  # (4, 768)
```

### ScoreHead

MLP 评分头，预测 0-9 范围的质量分数。

```python
from src.training.score_head import ScoreHead
import torch

score_head = ScoreHead(
    input_dim=768,
    hidden_dims=(256, 64),
    dropout=0.1,
)

# 输入聚合后的特征
features = torch.randn(4, 768)
scores = score_head(features)  # (4, 1) in [0, 9]

# 推理模式
scores = score_head.predict(features)  # 无梯度计算
```

---

## 检查点格式

保存的检查点结构：

```python
checkpoint = {
    "epoch": 10,
    "val_loss": 0.1234,
    "config": {...},
    "temporal_attention": temporal_attention.state_dict(),
    "score_head": score_head.state_dict(),
}
```

---

## 性能优化

### 显存优化

| 场景 | 推荐 batch_size | 说明 |
|------|----------------|------|
| 纯图像 | 16-32 | CLIP 特征提取轻量 |
| 视频 (8帧) | 8-16 | 根据分辨率调整 |
| 视频 (12帧) | 4-8 | 较高显存占用 |
| 混合数据 | 1-4 | 最安全配置 |

### DeepSpeed ZeRO-2 配置

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "bf16": { "enabled": true }
}
```

---

## 常见问题

### Q: 训练时出现 OOM 怎么办？

A: 尝试以下方法：
1. 减少 `batch_size`
2. 减少 `num_frames`（从 12 降到 8）
3. 使用 DeepSpeed CPU offload
4. 降低 `max_frames` 限制

### Q: 如何使用预训练的 CLIP 模型？

A: 模型会自动从 HuggingFace 下载。如果网络受限：

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### Q: 支持哪些 CLIP 模型变体？

A: 当前默认使用 `openai/clip-vit-large-patch14` (768-dim)。如需更换：

1. 修改 `FrozenCLIPEncoder.MODEL_NAME`
2. 调整 `feature_dim` 参数
3. 相应调整 `temporal_dim`

---

## 文件结构

```
xpfilter/
├── src/
│   ├── models/
│   │   └── frozen_clip_engine.py    # 推理引擎
│   └── training/
│       ├── frozen_clip_encoder.py   # CLIP 编码器
│       ├── frozen_clip_dataset.py   # 数据集
│       ├── temporal_attention.py    # 时间注意力
│       └── score_head.py            # 评分头
├── scripts/
│   └── train_frozen_clip.py         # 训练脚本
├── configs/
│   ├── ds_frozen_clip.json          # 通用 DeepSpeed 配置
│   └── ds_frozen_clip_v100_fp16.json # V100 FP16 配置
└── docs/
    └── frozen_clip_model.md         # 本文档
```

---

## 更新历史

- **2026-03-10**: 初始版本，包含完整的模型介绍、训练和推理文档

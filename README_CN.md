# XPfilter

XPfilter 是一个**带评分器筛选的 Telegram（可能会有更多平台） / 媒体下载管线**。

它的原理是：**尽可能广地下载候选媒体，用评分器逐个打分，只把达到阈值的内容物化到目标媒体库里**。

## 本项目真正要做什么

本项目提供了一套完整的工具链路，涵盖了标注，训练，推理和下载器。

完整链路是：

1. 使用webui标注工具，对指定文件夹中的媒体进行导入和预览，并且可以直接标注，标注结果将会存入数据库中，并且随时可以导出。
2. 当标注数量到达总数1000条以上，并且自认为高分低分分布差不多时，可以导出为label.json，并且运行训练，训练将会得到一个对齐自己的标注的评分器。
3. 项目中提供了多种推理脚本，可以为本地的文件夹进行基于分数的软链接分桶，帮你直接分层，“这个文件夹里哪些我爱看，哪些我不爱看。”也可以使用这种办法评估自己的打分模型是否符合预期。
4. 项目继承了telegram下载器，下载器会持续下载账号上的所有媒体文件，并且在本地经过评分器筛选，只保留大于特定阈值的媒体，从而实现高质量偏好媒体的筛选下载。

核心是：

> **用评分器驱动下载筛选，自动构建更干净、更高质量的媒体集合。**


## 使用方法

**本项目最合适的使用方法是接入一个agent，让agent代劳给出各个阶段的命令和说明，本项目的文档对agent来说非常充分。**


## 仓库结构

```text
configs/        运行与训练配置
docs/           详细文档
scripts/        训练 / 下载 / 重分桶 / 清理脚本
src/            API、模型、训练、存储、服务
tg_downloader/  Telegram 门控下载实现
tests/          测试
webui/          前端标注和管线界面
```

## 快速开始

### 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

首次启动前端前，安装前端依赖：

```bash
cd webui
npm install
cd ..
```

### 启动后端 + WebUI

```bash
./start.sh
```

可选模式：

- `./start.sh api`
- `./start.sh frontend`
- `./start.sh --build-webui`
- `./start.sh --del-cache`

默认端口：

- 后端：`31211`
- 前端：`31212`

## 典型使用路径

### 1. 标注数据

打开：

- `http://localhost:31212/label`

导出 `labels.json`。

### 2. 训练评分器

```bash
python scripts/train_frozen_clip.py \
  --labels_path labels.json \
  --output_dir checkpoints/frozen_clip \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --clip_model_name openai/clip-vit-large-patch14
```

### 3. 执行门控下载

```bash
python scripts/run_tg_gated_download.py --min-score 7.0
```

### 4. 执行完整 Telegram pipeline

```bash
python scripts/run_telegram_global_pipeline.py --min-score 7.0
```

完整编排可依次执行：

1. Telegram 门控下载
2. 可选补推理
3. 按分数重分桶
4. 可选清理阈值以下文件

## 需要注意的点

- 这个仓库公开的是 **Frozen CLIP 评分 + Telegram 门控下载 + API/WebUI** 的基线实现。
- 模型 checkpoint、Telegram session、本地数据库和下载数据本身不随仓库分发。
- 当前筛选机制是**下载后筛选**，因为模型必须先拿到完整文件才能评分。
- 要区分 `cache_root` 和 `target_root`：低分媒体仍可能留在缓存层，但默认只有达标结果会被物化到目标媒体库。
- 虽然仓库包含训练代码，但系统层面的最终目标仍然是：

> **做一个由评分器驱动的下载筛选器，而不是单独的训练实验仓库。**

## 相关文档

- `docs/frozen_clip_model.md`
- `docs/telegram_global_pipeline.md`
- `progress.md`
- `SKILL.md`

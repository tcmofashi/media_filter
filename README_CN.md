# MediaFilter 中文

[English](./README.md) | 中文

## 项目说明

MediaFilter 是一个公开发布的基线仓库，提供：

- 冻结 CLIP 训练与推理
- Telegram 门控下载 + 评分 + 重分桶 + 低分清理
- 简化后的 API 与 WebUI，支持标注、训练、下载与全链路任务编排

本仓库对外只保留当前主线可复用方案，已移除早期实验内容（如 heretic、LoRA、大模型推理扩展链路）。

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/tcmofashi/media_filter.git
cd media_filter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

WebUI 依赖：

```bash
cd webui
npm install
cd ..
```

### 2. 一键启动

```bash
./start.sh
```

- `./start.sh`：启动后端 + 前端
- `./start.sh api`：只启动后端
- `./start.sh frontend`：只启动前端
- `./start.sh --build-webui`：使用前端 build/preview 模式
- `./start.sh --del-cache`：启动前清理缓存目录

默认端口：

- 后端：`31211`
- 前端：`31212`

### 3. 快速验证

- API 文档：`http://localhost:31211/docs`
- 前端主页：`http://localhost:31212`
- 标注页：`http://localhost:31212/label`
- 任务页：`http://localhost:31212/pipeline`

## 核心特性

- WebUI 标注与导出（`labels.json`）
- CLIP 训练脚本：`scripts/train_frozen_clip.py`
- Telegram 下载门控：`scripts/run_tg_gated_download.py`
- 全链路编排脚本：`scripts/run_telegram_global_pipeline.py`
- 重排/重分桶/清理：`scripts/bulk_infer_telegram.py`、`scripts/rebucket_telegram_by_score.py`、`scripts/prune_telegram_below_score.py`
- API 和数据库入口用于模型结果查询

## 使用流程

1. 在 `/label` 标注并导出 `labels.json`
2. 训练或恢复模型：`scripts/train_frozen_clip.py`
3. 运行 Telegram 下载：`scripts/run_tg_gated_download.py`
4. 运行全链路任务：`scripts/run_telegram_global_pipeline.py`（下载 → 重打分 → 重分桶 → 清理）

详细流程与参数见 [`SKILL.md`](./SKILL.md)。

## 配置

- 运行时配置：`src/config.py`
- 默认公开配置：`configs/config.yaml`
- 本地覆盖：`configs/config.local.yaml`
- 环境变量前缀：`MF_`

## 目录结构

- `src/`：API、服务、模型、数据库与训练核心
- `scripts/`：脚本入口与任务编排
- `tg_downloader/`：Telegram 下载实现
- `webui/`：前端标注与任务面板
- `docs/`：模型和 pipeline 文档
- `data/`：运行时数据库/缓存（不提交）
- `checkpoints/`：本地产物/示例目录（运行时生成）

## 不发布内容

仓库不分发：

- 模型权重
- Telegram 会话文件
- 本地数据库和下载媒体
- 本地覆盖配置

## 文档索引

- 操作手册：[`SKILL.md`](./SKILL.md)
- Telegram 流水线：`docs/telegram_global_pipeline.md`
- 模型说明：`docs/frozen_clip_model.md`

## 协议

MIT 许可证，见 [`LICENSE`](./LICENSE)。

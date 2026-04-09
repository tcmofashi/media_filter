# MediaFilter 使用 Skill

这份 Skill 说明如何从标注到部署，完成仓库公开基线的全链路跑通。

## 1. 仓库定位

当前公开基线包含：

- Frozen CLIP 评分模型（训练 + 推理）
- Telegram 门控下载 + 分数过滤 + 重分桶 + 清理
- 最小 API + WebUI（单入口启动）

不包含分发：

- 模型 checkpoint
- Telegram session
- 本地数据库/缓存/下载文件

## 2. 启动前准备

1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 安装并准备前端依赖（首次启动）

```bash
cd webui
npm install
cd ..
```

3. 复制可选环境文件（如需要）

```bash
cp configs/config.yaml configs/config.local.yaml
```

## 3. Linux 一键启动

```bash
./start.sh
```

- `./start.sh api`：仅后端（API）
- `./start.sh frontend`：仅 WebUI
- `./start.sh --build-webui`：启动前端 build/preview 模式
- `./start.sh --del-cache`：启动前清理本地缓存目录

默认端口：

- 后端：`31211`
- 前端：`31212`

后端 API 文档：`http://localhost:31211/docs`

## 4. 标注网页（Label）

1. 打开 WebUI：`http://localhost:31212/label`
2. 上传媒体、填写 `score` 与可选 `prompt`
3. 提交后点击导出（导出 `labels.json`）

说明：

- 导出文件可直接用于训练脚本输入
- 分数范围默认建议为 `[0, 9]`

## 5. 数据导出 / 导入数据库

- 导出：WebUI 的导出入口默认产物为 `labels.json`
- 数据导入到本地 sqlite（可选）：

```bash
python scripts/import_to_db.py --train data/train.json --val data/val.json --db data/media_filter.db
```

如果你已经有 `labels.json`，可按项目既有脚本做切分后再入库。

## 6. 训练 Frozen CLIP

```bash
python scripts/train_frozen_clip.py \
  --labels_path labels.json \
  --output_dir checkpoints/frozen_clip \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --clip_model_name openai/clip-vit-large-patch14
```

推荐参数：

- 批量训练：`--clip_batch_size`（CLIP 前向切片）
- 视频策略：`--num_frames`
- 注意力：`--num_heads` / `--temporal_dim`

产物默认使用 `--model_checkpoint` 供后续推理引用（项目默认为 `checkpoints/checkpoint_best.pt`）。

## 7. 下载器流程（Telegram）

### 7.1 全链路编排（推荐）

```bash
python scripts/run_telegram_global_pipeline.py
```

会按配置依次执行：

1. Telegram 读取并过滤 `chat`（按 `chat_batch_size`）
2. 门控下载到缓存区
3. 可选重打分
4. 按分数重分桶到目标目录
5. 可选清理低分项

### 7.2 仅执行下载门控

```bash
python scripts/run_tg_gated_download.py
```

## 8. 部署与推理服务

### API 方式推理

- 上传单文件：`POST /score`
- 批量路径打分：`POST /batch-score`
- 拉取推理结果：`GET /media/inference`

### 任务调度方式（WebUI Pipeline）

在 `/pipeline` 页面可直接创建：

- 训练任务
- 下载任务
- 全链路任务
- 模型部署参数更新

## 9. 关键配置项速查

- `configs/config.yaml`：运行期核心配置（数据库、session、目录、模型路径）
- `src/config.py`：冻结 CLIP 运行时配置（含 `clip_model_name`、`model_checkpoint`）
- `.env`：环境变量入口（`MF_` 前缀）

## 10. 常见问题

- 下载失败/连接中断：优先确认代理/网络到 Telegram、HuggingFace 可达
- OOM（显存不足）：调低 `clip_batch_size`、`batch_size`，减小视频采样参数或使用 deepspeed
- 文件引用过期：重取源消息上下文后重试下载（Telegram 限制导致）
- 代理端口变更：后端脚本默认未硬编码代理，按脚本参数或系统代理环境变量设置

# MediaFlusher

MediaFlusher is a public baseline for a Frozen CLIP scoring stack and a Telegram media curation pipeline.

## Public scope

This repository intentionally keeps only the parts that are in active use:

- Frozen CLIP training and inference
- Telegram gated download, scoring, rebucketing, and flat-link materialization
- Minimal API and storage code needed by the current pipeline

Removed from the public baseline:

- Early `heretic` abliteration experiments
- Qwen / large-model inference routes
- LoRA training flows and related scripts

## Pipeline

The main published flow is:

1. Build/label data in webui (`/label`), export `labels.json`.
2. Import to sqlite if needed via `scripts/import_to_db.py`.
3. Train or resume the Frozen CLIP scorer with `scripts/train_frozen_clip.py`.
4. Run Telegram gated download with `scripts/run_tg_gated_download.py`.
5. Orchestrate full end-to-end via `scripts/run_telegram_global_pipeline.py` (download, optional bulk re-score, bucket, prune).
6. Materialize high-score outputs under the configured `target_root` and `flat_links_root`.

Detailed Telegram pipeline behavior is documented in `docs/telegram_global_pipeline.md`.
The Frozen CLIP model details and training reference is in `docs/frozen_clip_model.md`.

## Command Routing Matrix

- 标注入口（WebUI）: `./start.sh frontend` -> `webui` -> `/label` (标注) / `/pipeline` (作业入口)
- WebAPI: `./start.sh api` -> `src/main.py`
- 训练: `scripts/train_frozen_clip.py`
- 训练数据导入: `scripts/import_to_db.py`（配合 `--train/--val`）
- 标签导出: `webui` `/label` 页
- 单次下载门控: `scripts/run_tg_gated_download.py`
- 全链路编排: `scripts/run_telegram_global_pipeline.py`
- 全量重推理: `scripts/bulk_infer_telegram.py`
- 按分数重分桶: `scripts/rebucket_telegram_by_score.py`
- 低分清理: `scripts/prune_telegram_below_score.py`
- 数据切分: `scripts/split_dataset.py`

## Main entry points

- Frozen CLIP training: `scripts/train_frozen_clip.py`
- Telegram gated download: `scripts/run_tg_gated_download.py`
- Telegram global pipeline: `scripts/run_telegram_global_pipeline.py`
- Frozen CLIP runtime config: `src/config.py`
- Public project config: `configs/config.yaml`

## Linux one-stop startup

Use one command as the single Linux entry:

```bash
./start.sh
```

This starts both API and WebUI in the foreground with shared shutdown.
Other options:

- `./start.sh api` start backend API only.
- `./start.sh frontend` start WebUI only.
- `./start.sh --del-cache` clear cache dirs before startup.
- `./start.sh --build-webui` build frontend and preview it instead of dev mode.

## Not distributed

The public repository does not ship:

- Model checkpoints or pretrained weights
- Telegram sessions, caches, local databases, or downloaded media
- Local override config such as `configs/config.local.yaml`

Those artifacts are intentionally gitignored.

# MediaFilter

[中文文档](./README_CN.md)

**MediaFilter** is a public baseline for a frozen CLIP scoring system and Telegram media curation pipeline.  
Repository: `https://github.com/tcmofashi/media_filter`

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [Pipeline](#pipeline)
- [One-Command Entry](#one-command-entry)
- [CLI Entrypoints](#cli-entrypoints)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Security & Distribution Scope](#security--distribution-scope)
- [Documentation](#documentation)
- [License](#license)

## Project Overview

MediaFilter focuses on a minimal, production-style baseline:

- Frozen CLIP training and inference
- Telegram gated media discovery + download + scoring pipeline
- Re-bucketing and pruning by score
- Lightweight API and WebUI for labeling and pipeline orchestration

Historical experiment artifacts have been removed from the open-source package (such as early Heretic trials, LoRA pipelines, and large-model inference routes).

## Quick Start

### 1) Environment

```bash
git clone https://github.com/tcmofashi/media_filter.git
cd media_filter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

WebUI dependencies:

```bash
cd webui
npm install
cd ..
```

### 2) Start services

```bash
./start.sh
```

- `./start.sh` starts backend API + WebUI
- `./start.sh api` starts backend only
- `./start.sh frontend` starts WebUI only
- `./start.sh --build-webui` runs frontend build + preview mode
- `./start.sh --del-cache` clears local cache before start

Default ports:

- API: `31211`
- WebUI: `31212`

### 3) Verify

- Open API docs: `http://localhost:31211/docs`
- Open pipeline UI: `http://localhost:31212`
- Labeling: `http://localhost:31212/label`
- Job panel: `http://localhost:31212/pipeline`

## Core Features

- Labeling and score export in WebUI (`SKILL.md` for step-by-step workflow)
- Local SQLite media scoring/tracking metadata
- Frozen CLIP trainer: `scripts/train_frozen_clip.py`
- Telegram gated media download: `scripts/run_tg_gated_download.py`
- Full end-to-end orchestration: `scripts/run_telegram_global_pipeline.py`
- Re-score/rebucket/prune utilities for ranked media

## Pipeline

Typical pipeline flow:

1. Label data in WebUI and export `labels.json`.
2. Train/resume Frozen CLIP with `scripts/train_frozen_clip.py`.
3. Execute Telegram gated download: `scripts/run_tg_gated_download.py`.
4. Optional full orchestration:
   `scripts/run_telegram_global_pipeline.py` (download → score → rebucket → prune).
5. Materialize scored outputs to `target_root` and browse via media endpoints.

Score threshold and batch behaviors are configured in `configs/config.yaml` and command arguments.

## One-Command Entry

Use `start.sh` as the unified Linux entrypoint:

- Backend: `src/main.py` via FastAPI
- Frontend: `webui` (Vite dev server)
- Shared shutdown handler for all processes

This project intentionally uses a single shell entry for running local demo stack.

## CLI Entrypoints

- Training: `scripts/train_frozen_clip.py`
- Import labels to DB (optional): `scripts/import_to_db.py`
- Single pass download: `scripts/run_tg_gated_download.py`
- Full Telegram pipeline: `scripts/run_telegram_global_pipeline.py`
- Bulk re-score: `scripts/bulk_infer_telegram.py`
- Re-bucket: `scripts/rebucket_telegram_by_score.py`
- Prune low-score: `scripts/prune_telegram_below_score.py`
- Data split helper: `scripts/split_dataset.py`
- Label export/API: `src/api/routes/label.py`, `src/api/routes/export.py`

## Configuration

- Runtime config: `src/config.py`
- Public defaults: `configs/config.yaml`
- Local override: `configs/config.local.yaml` (gitignored or optional)
- Environment variables: prefix `MF_` (`MF_MODEL_CHECKPOINT`, etc.)

## Project Structure

- `src/`: API, services, storage, models, training code
- `scripts/`: CLI orchestration and pipeline scripts
- `tg_downloader/`: Telegram downloader implementation
- `webui/`: Frontend for labeling and pipeline operations
- `docs/`: Pipeline and model documentation
- `data/`: Runtime data dirs (gitignored)
- `configs/`: Runtime and deepspeed configurations

## Security & Distribution Scope

Not distributed in this repository:

- Pretrained model checkpoints
- Telegram session files
- Local DB/cache/download outputs
- Local override config (`configs/config.local.yaml`)

These paths are ignored via `.gitignore`.

## Documentation

- Operational Guide: [`SKILL.md`](./SKILL.md)
- Telegram Pipeline: `docs/telegram_global_pipeline.md`
- Model and training details: `docs/frozen_clip_model.md`

## License

This project is licensed under the [MIT License](./LICENSE).

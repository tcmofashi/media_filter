# XPfilter

XPfilter is a **Telegram (and potentially more platforms in the future) / media downloading pipeline with scorer-based filtering**.

Its core idea is: **download candidate media as broadly as possible, score each item one by one with a scorer, and only materialize content that reaches the threshold into the target media library**.

## What this project is really for

This project provides a complete toolchain covering labeling, training, inference, and the downloader.

The full workflow is:

1. Use the WebUI labeling tool to import and preview media from a specified folder, label them directly, store the labeling results in the database, and export them at any time.
2. Once the number of labeled items exceeds 1000 and you believe the high-score / low-score distribution is reasonably balanced, you can export to `labels.json` and run training. Training will produce a scorer aligned with your own labels.
3. The project provides multiple inference scripts that can create score-based soft-link buckets for local folders, helping you directly stratify content into layers like “which files in this folder I like” and “which files I do not like.” This can also be used to evaluate whether your scoring model matches your expectations.
4. The project inherits a Telegram downloader. The downloader continuously downloads all media files available to the account, then filters them locally through the scorer and keeps only media above a specific threshold, enabling preference-based high-quality media downloading.

The core is:

> **Use a scorer to drive download filtering and automatically build a cleaner, higher-quality media collection.**


## How to use it

**The most suitable way to use this project is to connect it to an agent and let the agent provide the commands and instructions for each stage. The documentation in this project is very sufficient for agents.**


## Repository structure

```text
configs/        runtime and training config
docs/           detailed documentation
scripts/        training / download / rebucket / cleanup scripts
src/            API, model, training, storage, services
tg_downloader/  Telegram gated download implementation
tests/          tests
webui/          frontend labeling and pipeline UI
```

## Quick start

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Before starting the frontend for the first time, install frontend dependencies:

```bash
cd webui
npm install
cd ..
```

### Start backend + WebUI

```bash
./start.sh
```

Optional modes:

- `./start.sh api`
- `./start.sh frontend`
- `./start.sh --build-webui`
- `./start.sh --del-cache`

Default ports:

- backend: `31211`
- frontend: `31212`

## Typical usage path

### 1. Label data

Open:

- `http://localhost:31212/label`

Export `labels.json`.

### 2. Train the scorer

```bash
python scripts/train_frozen_clip.py \
  --labels_path labels.json \
  --output_dir checkpoints/frozen_clip \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --clip_model_name openai/clip-vit-large-patch14
```

### 3. Run gated download

```bash
python scripts/run_tg_gated_download.py --min-score 7.0
```

### 4. Run the full Telegram pipeline

```bash
python scripts/run_telegram_global_pipeline.py --min-score 7.0
```

The full orchestration can run, in order:

1. Telegram gated download
2. Optional backfill inference
3. Score-based rebucketing
4. Optional cleanup of files below the threshold

## Notes

- This repository publishes the baseline implementation of **Frozen CLIP scoring + Telegram gated download + API/WebUI**.
- Model checkpoints, Telegram sessions, local databases, and downloaded data are not distributed with the repository.
- The current filtering mechanism is **post-download filtering**, because the model must first receive the complete file before it can score it.
- You should distinguish `cache_root` and `target_root`: low-score media may still remain in the cache layer, but by default only passing results are materialized into the target media library.
- Although the repository includes training code, the final system-level goal is still:

> **A scorer-driven download filter, rather than a standalone training experiment repository.**

## Related docs

- `docs/frozen_clip_model.md`
- `docs/telegram_global_pipeline.md`
- `progress.md`
- `SKILL.md`

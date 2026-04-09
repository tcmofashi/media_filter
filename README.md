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

1. Train or resume the Frozen CLIP scorer with `scripts/train_frozen_clip.py`.
2. Run Telegram gated download with `scripts/run_tg_gated_download.py`.
3. Orchestrate the full Telegram pipeline with `scripts/run_telegram_global_pipeline.py`.
4. Materialize high-score outputs under the configured `target_root` and `flat_links_root`.

Detailed Telegram pipeline behavior is documented in `docs/telegram_global_pipeline.md`.

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

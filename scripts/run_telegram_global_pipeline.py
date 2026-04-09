#!/usr/bin/env python
"""Run the local Telegram gated download pipeline inside media_filter."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tg_downloader.logging_utils import install_timestamped_output

LEGACY_ROOT = Path.home() / "telegram_media_downloader"
DEFAULT_MEDIA_ROOT = PROJECT_ROOT / "data"
DEFAULT_TARGET_ROOT = DEFAULT_MEDIA_ROOT / "tg_target"
DEFAULT_CACHE_ROOT = DEFAULT_MEDIA_ROOT / "tg_cache"
DEFAULT_FLAT_LINKS_ROOT = DEFAULT_MEDIA_ROOT / "tg_flat_links"
DEFAULT_DB_PATH = PROJECT_ROOT / "data/media_filter.db"
DEFAULT_STATE_PATH = PROJECT_ROOT / "data/tg_downloader_state.json"
DEFAULT_SESSION_DIR = PROJECT_ROOT / "data/tg_session"
DEFAULT_LEGACY_CONFIG = LEGACY_ROOT / "config.yaml"
DEFAULT_LEGACY_DATA = LEGACY_ROOT / "data.yaml"
DEFAULT_LEGACY_SESSION = LEGACY_ROOT / "sessions/media_downloader.session"
DEFAULT_CHECKPOINT = PROJECT_ROOT / (
    "outputs/"
    "frozen_clip_v100_4gpu_clip336_u2_nf16_mf32_b2_acc2_fp16_e8_resume_best_20260327/"
    "checkpoint_best.pt"
)

install_timestamped_output()


def threshold_label(score: float) -> str:
    """Build a filesystem-friendly threshold label."""
    text = f"{score:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def default_bucket_root(target_root: Path, min_score: float, mode: str) -> Path:
    """Resolve the default bucket output directory."""
    if mode == "move":
        return target_root
    return target_root / "score_links" / f"gte_{threshold_label(min_score)}"


def run_command(command: list[str], cwd: Path) -> None:
    """Run a child command and echo it first."""
    print(f"[run] cwd={cwd} cmd={shlex.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def env_first(*names: str) -> str | None:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local Telegram global-walk gated download, then optionally "
            "re-score existing files, materialize high-score buckets, and prune."
        )
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=None,
        help=(
            "Optional project YAML config. When omitted, loads configs/config.yaml "
            "and applies configs/config.local.yaml on top when present."
        ),
    )
    parser.add_argument(
        "--api-id",
        type=int,
        default=None,
        help="Telegram api_id. Falls back to TG_API_ID / TELEGRAM_API_ID.",
    )
    parser.add_argument(
        "--api-hash",
        type=str,
        default=None,
        help="Telegram api_hash. Falls back to TG_API_HASH / TELEGRAM_API_HASH.",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Pyrogram session name. Defaults to the legacy session stem when available.",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Directory used to store the local Pyrogram session.",
    )
    parser.add_argument(
        "--session-string",
        type=str,
        default=None,
        help="Optional Pyrogram session string. Falls back to TG_SESSION_STRING / TELEGRAM_SESSION_STRING.",
    )
    parser.add_argument(
        "--target-root",
        "--save-root",
        dest="target_root",
        type=Path,
        default=None,
        help="Directory where passing Telegram media will be materialized.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Directory where raw Telegram downloads are cached before gating.",
    )
    parser.add_argument(
        "--flat-links-root",
        type=Path,
        default=None,
        help="Optional cross-chat flat directory of score-prefixed symlinks to kept target files.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Incremental state JSON written by the local downloader.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite database used by media_filter.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Frozen CLIP checkpoint used for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Inference device, for example "auto", "cuda", "cuda:1", or "cpu".',
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Gate threshold. Files below this score will be deleted during download unless --keep-below-threshold is set.",
    )
    parser.add_argument(
        "--keep-below-threshold",
        action="store_true",
        help="Also materialize low-score files into target-root. Cache-root always keeps raw files.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["hardlink", "symlink", "copy"],
        default=None,
        help="How passing cache files are materialized into target-root.",
    )
    parser.add_argument(
        "--discover-chat-types",
        type=str,
        default=None,
        help='Comma-separated dialog types to walk, for example "channel,supergroup".',
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=None,
        help="Optional maximum number of raw messages to walk per chat in this run.",
    )
    parser.add_argument(
        "--max-chats",
        type=int,
        default=None,
        help="Optional cap on how many discovered dialogs to process.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Progress print interval within each chat.",
    )
    parser.add_argument(
        "--chat-batch-size",
        type=int,
        default=None,
        help="How many scoreable media messages to process per chat visit during gated download.",
    )
    parser.add_argument(
        "--message-concurrency",
        type=int,
        default=None,
        help="How many messages may be downloaded concurrently inside one chat batch.",
    )
    parser.add_argument(
        "--score-concurrency",
        type=int,
        default=None,
        help="How many score evaluations may run in parallel.",
    )
    parser.add_argument(
        "--cache-max-items",
        type=int,
        default=None,
        help="Maximum number of files kept under cache-root during gated download.",
    )
    parser.add_argument(
        "--chat-idle-seconds",
        type=float,
        default=None,
        help="Sleep interval between chat batches during gated download.",
    )
    parser.add_argument(
        "--round-idle-seconds",
        type=float,
        default=None,
        help="Sleep interval after idle rounds during gated download.",
    )
    parser.add_argument(
        "--continuous",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether the download stage should keep polling forever.",
    )
    parser.add_argument(
        "--breadth-rounds",
        type=int,
        default=None,
        help="How many breadth-first warm-up rounds to run before focus scheduling.",
    )
    parser.add_argument(
        "--focus-top-chats",
        type=int,
        default=None,
        help="After warm-up, revisit this many high-score chats each round.",
    )
    parser.add_argument(
        "--focus-min-scored",
        type=int,
        default=None,
        help="Minimum scored items before a chat is eligible for focus scheduling.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the gated download stage and operate on existing files/DB only.",
    )
    parser.add_argument(
        "--run-bulk-infer",
        action="store_true",
        help="After gated download, run bulk_infer_telegram.py across the whole save_root to backfill or refresh cached results.",
    )
    parser.add_argument(
        "--bulk-limit",
        type=int,
        default=None,
        help="Optional cap passed to bulk_infer_telegram.py.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress interval passed to bulk_infer_telegram.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-scoring in bulk_infer_telegram.py.",
    )
    parser.add_argument(
        "--skip-rebucket",
        action="store_true",
        help="Skip threshold export after download/scoring.",
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Output directory for thresholded media. Defaults to save_root/score_links/gte_<threshold> for non-move modes.",
    )
    parser.add_argument(
        "--rebucket-mode",
        choices=["move", "symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize thresholded media.",
    )
    parser.add_argument(
        "--prune-below-threshold",
        action="store_true",
        help="Delete completed media whose cached score is below --min-score.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview rebucket/prune actions without changing files. Download itself is not dry-run.",
    )
    parser.add_argument(
        "--legacy-config",
        type=Path,
        default=None,
        help="Path to the original telegram_media_downloader config.yaml.",
    )
    parser.add_argument(
        "--legacy-data",
        type=Path,
        default=None,
        help="Path to the original telegram_media_downloader data.yaml.",
    )
    parser.add_argument(
        "--legacy-session",
        type=Path,
        default=None,
        help="Path to the original telegram_media_downloader .session file.",
    )
    parser.add_argument(
        "--skip-legacy-session-copy",
        action="store_true",
        help="Do not copy the original .session into the local session dir. Use this for a fresh interactive login.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from tg_downloader.project_config import (
        get_config_bool,
        get_config_path,
        get_config_value,
        load_project_telegram_config,
        resolve_path,
    )

    project_config = load_project_telegram_config(args.project_config)
    target_root = get_config_path(args.target_root, project_config, "target_root", DEFAULT_TARGET_ROOT)
    cache_root = get_config_path(args.cache_root, project_config, "cache_root", DEFAULT_CACHE_ROOT)
    flat_links_root = resolve_path(
        get_config_value(args.flat_links_root, project_config, "flat_links_root"),
        default=DEFAULT_FLAT_LINKS_ROOT,
        project_root=PROJECT_ROOT,
    )
    state_path = get_config_path(args.state_path, project_config, "state_path", DEFAULT_STATE_PATH)
    checkpoint = get_config_path(args.checkpoint, project_config, "checkpoint", DEFAULT_CHECKPOINT)
    db_path = get_config_path(args.db, project_config, "db", DEFAULT_DB_PATH)
    session_dir = get_config_path(args.session_dir, project_config, "session_dir", DEFAULT_SESSION_DIR)
    device = str(get_config_value(args.device, project_config, "device", "auto"))
    target_mode = get_config_value(args.target_mode, project_config, "target_mode", "hardlink")
    discover_chat_types = get_config_value(
        args.discover_chat_types,
        project_config,
        "discover_chat_types",
        "channel,supergroup,group,private",
    )
    log_every = int(get_config_value(args.log_every, project_config, "log_every", 25))
    chat_batch_size = int(get_config_value(args.chat_batch_size, project_config, "chat_batch_size", 150))
    message_concurrency = int(
        get_config_value(args.message_concurrency, project_config, "message_concurrency", 3)
    )
    score_concurrency = int(
        get_config_value(args.score_concurrency, project_config, "score_concurrency", 2)
    )
    cache_max_items = int(get_config_value(args.cache_max_items, project_config, "cache_max_items", 100))
    chat_idle_seconds = float(
        get_config_value(args.chat_idle_seconds, project_config, "chat_idle_seconds", 2.0)
    )
    round_idle_seconds = float(
        get_config_value(args.round_idle_seconds, project_config, "round_idle_seconds", 30.0)
    )
    breadth_rounds = int(get_config_value(args.breadth_rounds, project_config, "breadth_rounds", 3))
    focus_top_chats = int(
        get_config_value(args.focus_top_chats, project_config, "focus_top_chats", 5)
    )
    focus_min_scored = int(
        get_config_value(args.focus_min_scored, project_config, "focus_min_scored", 5)
    )
    min_score = float(get_config_value(args.min_score, project_config, "min_score", 7.0))
    session_name = get_config_value(args.session_name, project_config, "session_name")
    legacy_config = get_config_path(
        args.legacy_config, project_config, "legacy_config", DEFAULT_LEGACY_CONFIG
    )
    legacy_data = get_config_path(
        args.legacy_data, project_config, "legacy_data", DEFAULT_LEGACY_DATA
    )
    legacy_session = get_config_path(
        args.legacy_session, project_config, "legacy_session", DEFAULT_LEGACY_SESSION
    )
    skip_legacy_session_copy = get_config_bool(
        args.skip_legacy_session_copy,
        project_config,
        "skip_legacy_session_copy",
        default=False,
    )
    bucket_root = (
        args.bucket_root.expanduser().resolve()
        if args.bucket_root is not None
        else default_bucket_root(target_root, min_score, args.rebucket_mode)
    )

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"target_root={target_root}")
    print(f"cache_root={cache_root}")
    print(f"flat_links_root={flat_links_root}")
    print(f"state_path={state_path}")
    print(f"db={db_path}")
    print(f"checkpoint={checkpoint}")
    print(f"device={device}")
    print(
        "project_config="
        f"{args.project_config or 'auto(configs/config.yaml + configs/config.local.yaml)'}"
    )
    print(f"min_score={min_score}")
    print(f"keep_below_threshold={int(args.keep_below_threshold)}")
    print(f"target_mode={target_mode}")
    print(f"message_concurrency={message_concurrency}")
    print(f"score_concurrency={score_concurrency}")
    print(f"cache_max_items={cache_max_items}")
    print(f"bucket_root={bucket_root}")
    print(f"rebucket_mode={args.rebucket_mode}")
    print(f"run_bulk_infer={int(args.run_bulk_infer)}")

    if not args.skip_download:
        download_command = [
            sys.executable,
            "scripts/run_tg_gated_download.py",
            "--target-root",
            str(target_root),
            "--cache-root",
            str(cache_root),
            "--flat-links-root",
            str(flat_links_root),
            "--state-path",
            str(state_path),
            "--db",
            str(db_path),
            "--checkpoint",
            str(checkpoint),
            "--device",
            device,
            "--min-score",
            str(min_score),
            "--session-dir",
            str(session_dir),
            "--discover-chat-types",
            discover_chat_types,
            "--log-every",
            str(log_every),
            "--chat-batch-size",
            str(chat_batch_size),
            "--message-concurrency",
            str(message_concurrency),
            "--score-concurrency",
            str(score_concurrency),
            "--cache-max-items",
            str(cache_max_items),
            "--chat-idle-seconds",
            str(chat_idle_seconds),
            "--round-idle-seconds",
            str(round_idle_seconds),
            "--breadth-rounds",
            str(breadth_rounds),
            "--focus-top-chats",
            str(focus_top_chats),
            "--focus-min-scored",
            str(focus_min_scored),
            "--target-mode",
            target_mode,
            "--legacy-config",
            str(legacy_config),
            "--legacy-data",
            str(legacy_data),
            "--legacy-session",
            str(legacy_session),
        ]
        if flat_links_root is None:
            download_command[download_command.index("--flat-links-root") : download_command.index("--flat-links-root") + 2] = []
        if session_name is not None:
            download_command.extend(["--session-name", session_name])
        if args.api_id is not None:
            download_command.extend(["--api-id", str(args.api_id)])
        elif env_first("TG_API_ID", "TELEGRAM_API_ID"):
            pass
        if args.api_hash is not None:
            download_command.extend(["--api-hash", args.api_hash])
        if args.session_string is not None:
            download_command.extend(["--session-string", args.session_string])
        if skip_legacy_session_copy:
            download_command.append("--skip-legacy-session-copy")
        if args.history_limit is not None:
            download_command.extend(["--history-limit", str(args.history_limit)])
        if args.max_chats is not None:
            download_command.extend(["--max-chats", str(args.max_chats)])
        if args.keep_below_threshold:
            download_command.append("--keep-below-threshold")
        if args.continuous is True:
            download_command.append("--continuous")
        else:
            download_command.append("--no-continuous")
        run_command(download_command, PROJECT_ROOT)

    if args.run_bulk_infer:
        infer_command = [
            sys.executable,
            "scripts/bulk_infer_telegram.py",
            "--root",
            str(target_root),
            "--db",
            str(db_path),
            "--checkpoint",
            str(checkpoint),
            "--device",
            device,
            "--progress-every",
            str(args.progress_every),
            "--skip-sort",
        ]
        if args.bulk_limit is not None:
            infer_command.extend(["--limit", str(args.bulk_limit)])
        if args.force:
            infer_command.append("--force")
        run_command(infer_command, PROJECT_ROOT)

    if not args.skip_rebucket:
        rebucket_command = [
            sys.executable,
            "scripts/rebucket_telegram_by_score.py",
            "--root",
            str(target_root),
            "--db",
            str(db_path),
            "--checkpoint",
            str(checkpoint),
            "--mode",
            args.rebucket_mode,
            "--bucket-root",
            str(bucket_root),
            "--min-score",
            str(min_score),
        ]
        if args.dry_run:
            rebucket_command.append("--dry-run")
        run_command(rebucket_command, PROJECT_ROOT)

    if args.prune_below_threshold:
        prune_command = [
            sys.executable,
            "scripts/prune_telegram_below_score.py",
            "--root",
            str(target_root),
            "--db",
            str(db_path),
            "--checkpoint",
            str(checkpoint),
            "--min-score",
            str(min_score),
        ]
        if args.dry_run:
            prune_command.append("--dry-run")
        run_command(prune_command, PROJECT_ROOT)

    print("[done] telegram global pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

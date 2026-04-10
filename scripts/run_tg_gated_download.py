#!/usr/bin/env python
"""Run the local Telegram global-walk gated downloader."""

from __future__ import annotations

import argparse
import os
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
DEFAULT_DB_PATH = PROJECT_ROOT / "data/xpfilter.db"
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


def env_first(*names: str) -> str | None:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def resolve_api_id(cli_value: int | None, legacy_value: int | None) -> int | None:
    """Resolve Telegram API ID from CLI, environment, or legacy config."""
    if cli_value is not None:
        return cli_value
    raw_value = env_first("TG_API_ID", "TELEGRAM_API_ID")
    if raw_value:
        return int(raw_value)
    return legacy_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover Telegram dialogs globally, walk message history, download "
            "scoreable media into a local cache, score it with the current Frozen "
            "CLIP model, and materialize only passing files into a target directory."
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
        help="Telegram api_id. Falls back to TG_API_ID / TELEGRAM_API_ID or legacy config.",
    )
    parser.add_argument(
        "--api-hash",
        type=str,
        default=None,
        help="Telegram api_hash. Falls back to TG_API_HASH / TELEGRAM_API_HASH or legacy config.",
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
        help="Incremental state JSON that tracks per-chat progress and retries.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite database used for inference cache and media index.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Frozen CLIP checkpoint used for scoring.",
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
        help="Only materialize media whose predicted score reaches this threshold.",
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
        help="How many scoreable media messages to process per chat visit.",
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
        help="Maximum number of media files kept under cache-root.",
    )
    parser.add_argument(
        "--chat-idle-seconds",
        type=float,
        default=None,
        help="Sleep interval between chat batches.",
    )
    parser.add_argument(
        "--round-idle-seconds",
        type=float,
        default=None,
        help="Sleep interval after an idle round when continuous mode is enabled.",
    )
    parser.add_argument(
        "--continuous",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep polling dialogs forever instead of exiting after one breadth round.",
    )
    parser.add_argument(
        "--breadth-rounds",
        type=int,
        default=None,
        help="How many flat breadth-first rounds to run before focusing on high-score chats.",
    )
    parser.add_argument(
        "--focus-top-chats",
        type=int,
        default=None,
        help="After warm-up, revisit this many top average-score chats each round.",
    )
    parser.add_argument(
        "--focus-min-scored",
        type=int,
        default=None,
        help="Minimum scored items before one chat is eligible for focus scheduling.",
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
        normalize_proxy_config,
        proxy_to_url,
        resolve_path,
    )

    project_config = load_project_telegram_config(args.project_config)
    checkpoint = get_config_path(
        args.checkpoint, project_config, "checkpoint", DEFAULT_CHECKPOINT
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    from tg_downloader.downloader import (
        DownloadConfig,
        parse_chat_types,
        run_gated_download,
    )
    from tg_downloader.legacy import (
        bootstrap_session_from_legacy,
        bootstrap_state_from_legacy,
        load_legacy_runtime,
    )

    legacy_config = get_config_path(
        args.legacy_config, project_config, "legacy_config", DEFAULT_LEGACY_CONFIG
    )
    legacy_data = get_config_path(
        args.legacy_data, project_config, "legacy_data", DEFAULT_LEGACY_DATA
    )
    legacy_session = get_config_path(
        args.legacy_session, project_config, "legacy_session", DEFAULT_LEGACY_SESSION
    )
    legacy = load_legacy_runtime(
        config_path=legacy_config,
        data_path=legacy_data,
        session_path=legacy_session,
    )
    api_id = resolve_api_id(args.api_id, legacy.api_id)
    api_hash = (
        args.api_hash
        or env_first("TG_API_HASH", "TELEGRAM_API_HASH")
        or legacy.api_hash
    )
    session_string = args.session_string or env_first(
        "TG_SESSION_STRING",
        "TELEGRAM_SESSION_STRING",
    )
    session_name = get_config_value(
        args.session_name, project_config, "session_name"
    ) or (
        legacy.session_path.stem if legacy.session_path is not None else "xpfilter_tg"
    )

    if api_id is None:
        raise SystemExit(
            "Missing Telegram api_id. Pass --api-id, set TG_API_ID, or provide legacy config."
        )
    if not api_hash:
        raise SystemExit(
            "Missing Telegram api_hash. Pass --api-hash, set TG_API_HASH, or provide legacy config."
        )

    target_root = get_config_path(
        args.target_root, project_config, "target_root", DEFAULT_TARGET_ROOT
    )
    cache_root = get_config_path(
        args.cache_root, project_config, "cache_root", DEFAULT_CACHE_ROOT
    )
    flat_links_root = resolve_path(
        get_config_value(args.flat_links_root, project_config, "flat_links_root"),
        default=DEFAULT_FLAT_LINKS_ROOT,
        project_root=PROJECT_ROOT,
    )
    state_path = get_config_path(
        args.state_path, project_config, "state_path", DEFAULT_STATE_PATH
    )
    db_path = get_config_path(args.db, project_config, "db", DEFAULT_DB_PATH)
    session_dir = get_config_path(
        args.session_dir, project_config, "session_dir", DEFAULT_SESSION_DIR
    )
    device = str(get_config_value(args.device, project_config, "device", "auto"))
    target_mode = get_config_value(
        args.target_mode, project_config, "target_mode", "hardlink"
    )
    discover_chat_types = get_config_value(
        args.discover_chat_types,
        project_config,
        "discover_chat_types",
        "channel,supergroup,group,private",
    )
    log_every = int(get_config_value(args.log_every, project_config, "log_every", 25))
    chat_batch_size = int(
        get_config_value(args.chat_batch_size, project_config, "chat_batch_size", 150)
    )
    message_concurrency = int(
        get_config_value(
            args.message_concurrency, project_config, "message_concurrency", 3
        )
    )
    score_concurrency = int(
        get_config_value(args.score_concurrency, project_config, "score_concurrency", 2)
    )
    cache_max_items = int(
        get_config_value(args.cache_max_items, project_config, "cache_max_items", 100)
    )
    chat_idle_seconds = float(
        get_config_value(
            args.chat_idle_seconds, project_config, "chat_idle_seconds", 2.0
        )
    )
    round_idle_seconds = float(
        get_config_value(
            args.round_idle_seconds, project_config, "round_idle_seconds", 30.0
        )
    )
    continuous = bool(
        get_config_value(args.continuous, project_config, "continuous", True)
    )
    breadth_rounds = int(
        get_config_value(args.breadth_rounds, project_config, "breadth_rounds", 3)
    )
    focus_top_chats = int(
        get_config_value(args.focus_top_chats, project_config, "focus_top_chats", 5)
    )
    focus_min_scored = int(
        get_config_value(args.focus_min_scored, project_config, "focus_min_scored", 5)
    )
    min_score = float(
        get_config_value(args.min_score, project_config, "min_score", 7.0)
    )
    skip_legacy_session_copy = get_config_bool(
        args.skip_legacy_session_copy,
        project_config,
        "skip_legacy_session_copy",
        default=False,
    )
    proxy = normalize_proxy_config(project_config.get("proxy")) or legacy.proxy
    proxy_url = proxy_to_url(proxy)
    if proxy_url:
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

    bootstrap_state_from_legacy(state_path, legacy)
    local_session_path = None
    if not session_string and not skip_legacy_session_copy:
        local_session_path = bootstrap_session_from_legacy(
            session_dir=session_dir,
            session_name=session_name,
            legacy_session_path=legacy.session_path,
        )

    config = DownloadConfig(
        api_id=api_id,
        api_hash=api_hash,
        session_name=session_name,
        save_root=target_root,
        cache_root=cache_root,
        flat_links_root=flat_links_root,
        state_path=state_path,
        db_path=db_path,
        checkpoint_path=checkpoint,
        device=device,
        session_dir=session_dir,
        session_string=session_string,
        proxy=proxy,
        discover_chat_types=parse_chat_types(discover_chat_types),
        history_limit=args.history_limit,
        max_chats=args.max_chats,
        min_score=min_score,
        keep_below_threshold=args.keep_below_threshold,
        target_mode=target_mode,
        log_every=log_every,
        chat_batch_size=chat_batch_size,
        message_concurrency=message_concurrency,
        score_concurrency=score_concurrency,
        cache_max_items=cache_max_items,
        chat_idle_seconds=chat_idle_seconds,
        round_idle_seconds=round_idle_seconds,
        continuous=continuous,
        breadth_rounds=breadth_rounds,
        focus_top_chats=focus_top_chats,
        focus_min_scored=focus_min_scored,
    )

    print(f"target_root={config.save_root}")
    print(f"cache_root={config.cache_root}")
    print(f"flat_links_root={config.flat_links_root}")
    print(f"state_path={config.state_path}")
    print(f"db={config.db_path}")
    print(f"checkpoint={config.checkpoint_path}")
    print(f"device={config.device}")
    print(f"session_dir={config.session_dir}")
    print(f"session_name={config.session_name}")
    print(
        "project_config="
        f"{args.project_config or 'auto(configs/config.yaml + configs/config.local.yaml)'}"
    )
    print(f"legacy_config={legacy.config_path}")
    print(f"legacy_data={legacy.data_path}")
    print(f"legacy_session={legacy.session_path}")
    print(f"local_session={local_session_path}")
    print(f"skip_legacy_session_copy={int(skip_legacy_session_copy)}")
    print(f"proxy_enabled={int(bool(config.proxy))}")
    print(f"proxy_url={proxy_to_url(config.proxy, include_auth=False)}")
    print(f"min_score={config.min_score}")
    print(f"keep_below_threshold={int(config.keep_below_threshold)}")
    print(f"target_mode={config.target_mode}")
    print(f"discover_chat_types={','.join(config.discover_chat_types)}")
    print(f"history_limit={config.history_limit}")
    print(f"max_chats={config.max_chats}")
    print(f"chat_batch_size={config.chat_batch_size}")
    print(f"message_concurrency={config.message_concurrency}")
    print(f"score_concurrency={config.score_concurrency}")
    print(f"cache_max_items={config.cache_max_items}")
    print(f"chat_idle_seconds={config.chat_idle_seconds}")
    print(f"round_idle_seconds={config.round_idle_seconds}")
    print(f"continuous={int(config.continuous)}")
    print(f"breadth_rounds={config.breadth_rounds}")
    print(f"focus_top_chats={config.focus_top_chats}")
    print(f"focus_min_scored={config.focus_min_scored}")

    try:
        run_gated_download(config)
    except KeyboardInterrupt:
        raise SystemExit("Interrupted.") from None
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

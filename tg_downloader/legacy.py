"""Helpers to reuse the original telegram_media_downloader runtime state."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ruamel import yaml

from tg_downloader.state import DownloaderState

LEGACY_ROOT = Path.home() / "telegram_media_downloader"
DEFAULT_LEGACY_CONFIG = LEGACY_ROOT / "config.yaml"
DEFAULT_LEGACY_DATA = LEGACY_ROOT / "data.yaml"
DEFAULT_LEGACY_SESSION = LEGACY_ROOT / "sessions" / "media_downloader.session"

_YAML = yaml.YAML(typ="safe")


@dataclass(frozen=True)
class LegacyChatState:
    """Legacy per-chat progress merged from config.yaml and data.yaml."""

    chat_id: str
    last_read_message_id: int = 0
    ids_to_retry: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class LegacyRuntime:
    """Legacy runtime configuration used to bootstrap the local downloader."""

    api_id: int | None = None
    api_hash: str = ""
    proxy: dict[str, Any] | None = None
    chats: dict[str, LegacyChatState] = field(default_factory=dict)
    config_path: Path | None = None
    data_path: Path | None = None
    session_path: Path | None = None


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    payload = _YAML.load(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normalize_proxy(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    proxy = dict(value)
    if not proxy.get("scheme") or not proxy.get("hostname") or not proxy.get("port"):
        return None
    return proxy


def load_legacy_runtime(
    config_path: Path | None = DEFAULT_LEGACY_CONFIG,
    data_path: Path | None = DEFAULT_LEGACY_DATA,
    session_path: Path | None = DEFAULT_LEGACY_SESSION,
) -> LegacyRuntime:
    """Load config/state from the original telegram_media_downloader repo."""
    resolved_config = config_path.expanduser().resolve() if config_path else None
    resolved_data = data_path.expanduser().resolve() if data_path else None
    resolved_session = session_path.expanduser().resolve() if session_path else None

    config = _load_yaml(resolved_config)
    data = _load_yaml(resolved_data)
    chats: dict[str, LegacyChatState] = {}

    for item in config.get("chat") or []:
        if not isinstance(item, dict) or item.get("chat_id") is None:
            continue
        chat_id = str(item["chat_id"])
        chats[chat_id] = LegacyChatState(
            chat_id=chat_id,
            last_read_message_id=int(item.get("last_read_message_id") or 0),
            ids_to_retry=[],
        )

    for item in data.get("chat") or []:
        if not isinstance(item, dict) or item.get("chat_id") is None:
            continue
        chat_id = str(item["chat_id"])
        retry_ids = sorted({int(value) for value in (item.get("ids_to_retry") or [])})
        existing = chats.get(chat_id)
        chats[chat_id] = LegacyChatState(
            chat_id=chat_id,
            last_read_message_id=(
                existing.last_read_message_id if existing is not None else 0
            ),
            ids_to_retry=retry_ids,
        )

    return LegacyRuntime(
        api_id=int(config["api_id"]) if config.get("api_id") is not None else None,
        api_hash=str(config.get("api_hash") or ""),
        proxy=_normalize_proxy(config.get("proxy")),
        chats=chats,
        config_path=resolved_config if resolved_config and resolved_config.exists() else None,
        data_path=resolved_data if resolved_data and resolved_data.exists() else None,
        session_path=(
            resolved_session
            if resolved_session is not None and resolved_session.exists()
            else None
        ),
    )


def bootstrap_state_from_legacy(
    state_path: Path,
    runtime: LegacyRuntime,
) -> DownloaderState:
    """Seed the local state file using progress from the legacy downloader."""
    state = DownloaderState.load(state_path)
    for chat_id, legacy_chat in runtime.chats.items():
        chat_state = state.get_chat(chat_id=chat_id)
        chat_state.last_read_message_id = max(
            chat_state.last_read_message_id,
            legacy_chat.last_read_message_id,
        )
        if legacy_chat.ids_to_retry:
            chat_state.failed_message_ids = sorted(
                set(chat_state.failed_message_ids).union(legacy_chat.ids_to_retry)
            )
    state.save()
    return state


def bootstrap_session_from_legacy(
    session_dir: Path,
    session_name: str,
    legacy_session_path: Path | None,
) -> Path | None:
    """Copy the original .session file into the local project session directory."""
    if legacy_session_path is None:
        return None

    source = legacy_session_path.expanduser().resolve()
    if not source.exists():
        return None

    session_dir = session_dir.expanduser().resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    target = session_dir / f"{session_name}.session"
    if not target.exists():
        shutil.copy2(source, target)
    return target

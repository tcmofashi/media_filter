"""Helpers to read Telegram downloader defaults from the project config."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ruamel import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CONFIG_PATH = PROJECT_ROOT / "configs/config.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs/config.local.yaml"
DEFAULT_CONFIG_PATH = LOCAL_CONFIG_PATH

_YAML = yaml.YAML(typ="safe")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML mapping file if it exists."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    payload = _YAML.load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return dict(payload)


def _extract_telegram_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the telegram subsection from one config mapping."""
    telegram = payload.get("telegram")
    return dict(telegram) if isinstance(telegram, dict) else {}


def load_project_telegram_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load telegram config from an explicit file or the public+local defaults."""
    if config_path is not None:
        return _extract_telegram_config(_load_yaml_mapping(config_path))

    telegram = _extract_telegram_config(_load_yaml_mapping(PUBLIC_CONFIG_PATH))
    telegram.update(_extract_telegram_config(_load_yaml_mapping(LOCAL_CONFIG_PATH)))
    return telegram


def resolve_path(
    value: str | Path | None,
    *,
    default: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path | None:
    """Resolve config paths relative to the project root unless already absolute."""
    if value is None:
        return default

    path = value if isinstance(value, Path) else Path(value)
    path = path.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def get_config_value(
    cli_value: Any,
    config: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Prefer CLI, then project config, then a hardcoded default."""
    if cli_value is not None:
        return cli_value
    value = config.get(key)
    return default if value is None else value


def get_config_path(
    cli_value: Path | None,
    config: dict[str, Any],
    key: str,
    default: Path,
) -> Path:
    """Resolve a path from CLI or project config."""
    resolved = resolve_path(cli_value, project_root=PROJECT_ROOT)
    if resolved is not None:
        return resolved
    resolved = resolve_path(config.get(key), default=default, project_root=PROJECT_ROOT)
    return resolved if resolved is not None else default


def get_config_bool(
    cli_flag: bool,
    config: dict[str, Any],
    key: str,
    default: bool = False,
) -> bool:
    """Treat CLI true as highest priority, otherwise consult project config."""
    if cli_flag:
        return True
    value = config.get(key)
    if value is None:
        return default
    return bool(value)


def normalize_proxy_config(value: Any) -> dict[str, Any] | None:
    """Normalize one proxy config from YAML into the pyrogram proxy shape."""
    if isinstance(value, str):
        parsed = urlparse(value.strip())
        if not parsed.scheme or not parsed.hostname or parsed.port is None:
            return None
        proxy = {
            "scheme": parsed.scheme.lower(),
            "hostname": parsed.hostname,
            "port": int(parsed.port),
        }
        if parsed.username:
            proxy["username"] = parsed.username
        if parsed.password:
            proxy["password"] = parsed.password
        return proxy

    if not isinstance(value, dict):
        return None

    scheme = str(value.get("scheme") or "").strip().lower()
    hostname = str(value.get("hostname") or value.get("host") or "").strip()
    port = value.get("port")
    if not scheme or not hostname or port is None:
        return None

    proxy = {
        "scheme": scheme,
        "hostname": hostname,
        "port": int(port),
    }
    username = value.get("username")
    password = value.get("password")
    if username:
        proxy["username"] = str(username)
    if password:
        proxy["password"] = str(password)
    return proxy


def proxy_to_url(proxy: dict[str, Any] | None, *, include_auth: bool = True) -> str | None:
    """Render one proxy dict into a URL string."""
    if not proxy:
        return None
    scheme = str(proxy.get("scheme") or "").strip().lower()
    hostname = str(proxy.get("hostname") or "").strip()
    port = proxy.get("port")
    if not scheme or not hostname or port is None:
        return None

    auth = ""
    username = proxy.get("username")
    password = proxy.get("password")
    if include_auth and username:
        auth = str(username)
        if password:
            auth += f":{password}"
        auth += "@"
    return f"{scheme}://{auth}{hostname}:{int(port)}"

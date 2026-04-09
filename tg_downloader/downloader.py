"""Telegram global-walk gated downloader implemented inside media_filter."""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.models.frozen_clip_engine import FrozenClipEngine
from tg_downloader.db import InferenceStore, compute_content_hash
from tg_downloader.history import get_chat_history_v2
from tg_downloader.state import DownloaderState

try:
    import pyrogram
    from pyrogram.errors import (
        AuthKeyUnregistered,
        BadRequest,
        FileReferenceExpired,
        FloodWait,
        RPCError,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "pyrogram is required for tg_downloader. Install requirements.txt first."
    ) from exc

SAFE_COMPONENT_RE = re.compile(r"[^\w.-]+", flags=re.UNICODE)
SCORE_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)?__")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PHOTO_DEFAULT_EXTENSION = ".jpg"
MEDIA_RETRY_COUNT = 3
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVER_CHAT_TYPES = ("channel", "supergroup", "group", "private")
DEFAULT_LEGACY_TARGET_ROOT = PROJECT_ROOT / "data/tg_target"
DEFAULT_LEGACY_CACHE_ROOT = PROJECT_ROOT / "data/tg_cache"
DEFAULT_LEGACY_FLAT_LINKS_ROOT = PROJECT_ROOT / "data/tg_flat_links"
MIME_EXTENSION_OVERRIDES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}
NETWORK_RESTART_DELAY_SECONDS = 3.0
NETWORK_RESTART_MAX_DELAY_SECONDS = 60.0


@dataclass(frozen=True)
class DialogInfo:
    """One discovered Telegram dialog."""

    chat_id: int | str
    title: str
    chat_type: str
    has_protected_content: bool = False


@dataclass(frozen=True)
class MediaCandidate:
    """One scoreable media candidate extracted from a message."""

    media_type: str
    file_stem: str
    extension: str
    expected_size: int


@dataclass
class DownloadConfig:
    """Runtime configuration for local gated downloads."""

    api_id: int
    api_hash: str
    session_name: str
    save_root: Path
    cache_root: Path
    flat_links_root: Path | None
    state_path: Path
    db_path: Path
    checkpoint_path: Path
    device: str = "auto"
    session_dir: Path = Path("data/tg_session")
    session_string: str | None = None
    proxy: dict[str, Any] | None = None
    discover_chat_types: tuple[str, ...] = DEFAULT_DISCOVER_CHAT_TYPES
    history_limit: int | None = None
    max_chats: int | None = None
    min_score: float = 7.0
    keep_below_threshold: bool = False
    target_mode: str = "hardlink"
    log_every: int = 25
    chat_batch_size: int = 150
    message_concurrency: int = 3
    score_concurrency: int = 2
    cache_max_items: int = 100
    chat_idle_seconds: float = 2.0
    round_idle_seconds: float = 30.0
    continuous: bool = True
    breadth_rounds: int = 3
    focus_top_chats: int = 5
    focus_min_scored: int = 5


@dataclass
class RunStats:
    """Counters collected during one run."""

    chats_discovered: int = 0
    chat_batches: int = 0
    focus_batches: int = 0
    rounds: int = 0
    messages_seen: int = 0
    downloads: int = 0
    scored: int = 0
    kept: int = 0
    deleted: int = 0
    skipped: int = 0
    failed: int = 0
    cache_hits: int = 0
    target_materialized: int = 0
    flat_materialized: int = 0
    cache_evicted: int = 0
    protected_seen: int = 0
    protected_kept: int = 0
    protected_failed: int = 0


def parse_chat_types(value: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize chat type filters."""
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)
    return tuple(item.strip().lower() for item in items if item and item.strip())


def normalize_chat_type(chat_type: Any) -> str:
    """Normalize a pyrogram chat type value."""
    if chat_type is None:
        return ""

    value = getattr(chat_type, "value", chat_type)
    normalized = str(value).strip().lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", maxsplit=1)[-1]
    return normalized


def is_protected_message(message: pyrogram.types.Message) -> bool:
    """Check whether one message or its chat is marked as protected."""
    if bool(getattr(message, "has_protected_content", False)):
        return True
    chat = getattr(message, "chat", None)
    return bool(getattr(chat, "has_protected_content", False))


def display_chat_name(chat: Any) -> str:
    """Return a human-readable chat title."""
    return (
        getattr(chat, "title", None)
        or getattr(chat, "first_name", None)
        or getattr(chat, "username", None)
        or str(getattr(chat, "id", "unknown"))
    )


def sanitize_component(text: str, fallback: str, max_length: int = 96) -> str:
    """Make a filesystem-safe path component while preserving unicode words."""
    cleaned = re.sub(r"\s+", "_", (text or "").strip())
    cleaned = SAFE_COMPONENT_RE.sub("_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_length]


def classify_extension(extension: str) -> str | None:
    """Resolve a file extension to image or video."""
    suffix = extension.lower()
    if suffix in FrozenClipEngine.IMAGE_EXTENSIONS:
        return "image"
    if suffix in FrozenClipEngine.VIDEO_EXTENSIONS:
        return "video"
    return None


def guess_extension(
    file_name: str | None,
    mime_type: str | None,
    default: str,
) -> str:
    """Guess a file extension from file name and mime type."""
    suffix = Path(file_name).suffix.lower() if file_name else ""
    if suffix:
        return suffix
    if mime_type and mime_type.lower() in MIME_EXTENSION_OVERRIDES:
        return MIME_EXTENSION_OVERRIDES[mime_type.lower()]
    guessed = mimetypes.guess_extension(mime_type or "")
    if guessed == ".jpe":
        return ".jpg"
    return guessed or default


def resolve_media_candidate(message: pyrogram.types.Message) -> MediaCandidate | None:
    """Pick the first scoreable media item from a Telegram message."""
    if message.photo:
        file_stem = getattr(message.photo, "file_unique_id", None) or "photo"
        return MediaCandidate(
            media_type="image",
            file_stem=file_stem,
            extension=PHOTO_DEFAULT_EXTENSION,
            expected_size=getattr(message.photo, "file_size", 0) or 0,
        )

    if message.video:
        extension = guess_extension(
            getattr(message.video, "file_name", None),
            getattr(message.video, "mime_type", None),
            ".mp4",
        )
        return MediaCandidate(
            media_type="video",
            file_stem=Path(getattr(message.video, "file_name", "") or "video").stem
            or "video",
            extension=extension,
            expected_size=getattr(message.video, "file_size", 0) or 0,
        )

    if message.animation:
        extension = guess_extension(
            getattr(message.animation, "file_name", None),
            getattr(message.animation, "mime_type", None),
            ".mp4",
        )
        media_type = classify_extension(extension)
        if media_type is None:
            return None
        return MediaCandidate(
            media_type=media_type,
            file_stem=Path(
                getattr(message.animation, "file_name", "") or "animation"
            ).stem
            or "animation",
            extension=extension,
            expected_size=getattr(message.animation, "file_size", 0) or 0,
        )

    if message.document:
        extension = guess_extension(
            getattr(message.document, "file_name", None),
            getattr(message.document, "mime_type", None),
            "",
        )
        media_type = classify_extension(extension)
        if media_type is None:
            return None
        return MediaCandidate(
            media_type=media_type,
            file_stem=Path(
                getattr(message.document, "file_name", "") or "document"
            ).stem
            or "document",
            extension=extension,
            expected_size=getattr(message.document, "file_size", 0) or 0,
        )

    return None


def is_scoreable_message(message: pyrogram.types.Message) -> bool:
    """Return whether one message contains supported media for download/scoring."""
    return resolve_media_candidate(message) is not None


def resolve_download_input(message: pyrogram.types.Message) -> object:
    """Pick the most specific downloadable object for one message."""
    for attr_name in ("photo", "video", "animation", "document"):
        media = getattr(message, attr_name, None)
        if media is not None:
            return media
    return message


def build_relative_path(
    message: pyrogram.types.Message,
    candidate: MediaCandidate,
    score_prefix: float | None = None,
) -> Path:
    """Build a deterministic relative path for one message media."""
    chat = message.chat
    chat_id = getattr(chat, "id", "unknown")
    chat_name = sanitize_component(
        display_chat_name(chat),
        fallback=f"chat_{chat_id}",
    )
    chat_type = sanitize_component(
        normalize_chat_type(getattr(chat, "type", None)) or "unknown",
        fallback="unknown",
    )
    file_stem = sanitize_component(candidate.file_stem, fallback="media", max_length=72)
    file_name = f"{message.id}_{file_stem}{candidate.extension.lower()}"
    if score_prefix is not None:
        file_name = f"{format_score_prefix(score_prefix)}{file_name}"
    return Path(chat_type) / chat_name / file_name


def build_target_path(
    root: Path,
    message: pyrogram.types.Message,
    candidate: MediaCandidate,
    score_prefix: float | None = None,
) -> Path:
    """Build a deterministic full path under one root."""
    return root / build_relative_path(message, candidate, score_prefix=score_prefix)


def build_flat_relative_path(
    message: pyrogram.types.Message,
    candidate: MediaCandidate,
    score_prefix: float | None = None,
) -> Path:
    """Build a flattened filename without per-chat subdirectories."""
    relative = build_relative_path(message, candidate)
    return build_flat_relative_path_from_relative(relative, score_prefix=score_prefix)


def build_flat_relative_path_from_relative(
    relative: Path,
    score_prefix: float | None = None,
) -> Path:
    """Flatten one target-root relative path into a single score-prefixed filename."""
    flat_name = "__".join((*relative.parts[:-1], strip_score_prefix(relative.name)))
    if score_prefix is not None:
        flat_name = f"{format_score_prefix(score_prefix)}{flat_name}"
    return Path(flat_name)


def build_flat_target_path(
    root: Path,
    message: pyrogram.types.Message,
    candidate: MediaCandidate,
    score_prefix: float | None = None,
) -> Path:
    """Build one flat output path under the cross-chat symlink directory."""
    return root / build_flat_relative_path(message, candidate, score_prefix=score_prefix)


def build_score_prefixed_relative_path(
    relative: Path,
    score_prefix: float,
) -> Path:
    """Attach one score prefix to an existing cache/target relative path."""
    return relative.with_name(f"{format_score_prefix(score_prefix)}{strip_score_prefix(relative.name)}")


def sync_target_root(
    config: DownloadConfig,
    store: InferenceStore,
) -> int:
    """Backfill kept target files from completed cache inference rows."""
    materialized = 0
    for row in store.list_results_under_root(config.cache_root):
        if row["status"] != "completed" or row["score"] is None:
            continue
        score = float(row["score"])
        if score < config.min_score and not config.keep_below_threshold:
            continue

        cache_path = Path(row["media_path"])
        if not cache_path.exists():
            continue

        try:
            relative = cache_path.relative_to(config.cache_root)
        except ValueError:
            continue

        target_path = config.save_root / build_score_prefixed_relative_path(relative, score)
        canonical_target_path = config.save_root / relative
        delete_target_variants(
            canonical_target_path,
            config.save_root,
            store,
            keep_path=target_path,
        )
        if materialize_target_file(cache_path, target_path, config.target_mode):
            materialized += 1
        store.write_result(
            media_path=target_path,
            media_type=row["media_type"],
            content_hash=row["content_hash"],
            checkpoint_path=Path(row["model_checkpoint"]),
            score=score,
            status="completed",
        )

    if materialized > 0:
        print_progress("target-sync", root=config.save_root, materialized=materialized)
    return materialized


def sync_flat_links_root(
    config: DownloadConfig,
    store: InferenceStore,
) -> int:
    """Reconcile flat score-prefixed symlinks and their DB rows from target files."""
    if config.flat_links_root is None:
        return 0

    materialized = 0
    db_backfilled = 0
    removed = 0
    expected_paths: set[Path] = set()
    for target_path in sorted(
        path for path in config.save_root.rglob("*") if path.is_file() or path.is_symlink()
    ):
        relative = target_path.relative_to(config.save_root)
        match = SCORE_PREFIX_RE.match(relative.name)
        if match is None:
            continue
        score = float(match.group(0)[:-2])
        canonical_flat_path = config.flat_links_root / build_flat_relative_path_from_relative(relative)
        flat_target_path = config.flat_links_root / build_flat_relative_path_from_relative(
            relative,
            score_prefix=score,
        )
        expected_paths.add(flat_target_path)
        delete_target_variants(
            canonical_flat_path,
            config.flat_links_root,
            store,
            keep_path=flat_target_path,
        )
        if materialize_target_file(target_path, flat_target_path, "symlink"):
            materialized += 1
        if store.get_result_by_path(flat_target_path) is None:
            db_backfilled += 1
        target_row = store.get_result_by_path(target_path)
        if target_row is not None:
            store.write_result(
                media_path=flat_target_path,
                media_type=target_row["media_type"],
                content_hash=target_row["content_hash"],
                checkpoint_path=Path(target_row["model_checkpoint"]),
                score=target_row["score"],
                status=target_row["status"],
                error=target_row["error"],
            )

    for flat_path in sorted(
        path for path in config.flat_links_root.iterdir() if path.is_file() or path.is_symlink()
    ):
        if flat_path in expected_paths:
            continue
        flat_path.unlink()
        store.delete_path(flat_path)
        removed += 1

    if materialized > 0 or db_backfilled > 0 or removed > 0:
        print_progress(
            "flat-sync",
            root=config.flat_links_root,
            materialized=materialized,
            db_backfilled=db_backfilled,
            removed=removed,
        )
    return materialized


def format_score_prefix(score: float) -> str:
    """Format one score as a stable filename prefix."""
    return f"{score:.4f}__"


def strip_score_prefix(file_name: str) -> str:
    """Remove one score prefix from a filename when present."""
    return SCORE_PREFIX_RE.sub("", file_name, count=1)


def delete_target_variants(
    target_path: Path,
    stop_root: Path,
    store: InferenceStore,
    keep_path: Path | None = None,
) -> int:
    """Delete existing target files and DB rows for one logical basename."""
    parent = target_path.parent
    canonical_name = strip_score_prefix(target_path.name)
    variants = set(store.find_name_variants(parent, canonical_name))

    if parent.exists():
        for sibling in parent.iterdir():
            if sibling.is_dir():
                continue
            if strip_score_prefix(sibling.name) == canonical_name:
                variants.add(sibling)

    deleted = 0
    for variant in sorted(variants, key=str):
        if keep_path is not None and variant == keep_path:
            continue
        if variant.exists() or variant.is_symlink():
            variant.unlink()
        store.delete_path(variant)
        deleted += 1

    if deleted:
        prune_empty_parents(parent, stop_root)
    return deleted


def persist_chat_stats(
    store: InferenceStore,
    state: DownloaderState,
    chat_id: int | str,
) -> None:
    """Persist aggregate stats for one chat into sqlite."""
    chat_state = state.peek_chat(chat_id)
    if chat_state is not None:
        store.write_chat_stats(chat_state)


def order_dialogs_for_breadth(
    dialogs: list[DialogInfo],
    state: DownloaderState,
    max_chats: int | None,
) -> list[DialogInfo]:
    """Breadth-first order: favor chats with fewer processed messages so far."""
    ordered = sorted(
        dialogs,
        key=lambda dialog: (
            (
                state.peek_chat(dialog.chat_id).processed_count
                if state.peek_chat(dialog.chat_id) is not None
                else 0
            ),
            (
                state.peek_chat(dialog.chat_id).last_read_message_id
                if state.peek_chat(dialog.chat_id) is not None
                else 0
            ),
            str(dialog.chat_id),
        ),
    )
    if max_chats is not None:
        return ordered[:max_chats]
    return ordered


def select_focus_dialogs(
    dialogs: list[DialogInfo],
    state: DownloaderState,
    config: DownloadConfig,
) -> list[DialogInfo]:
    """Pick the current high-scoring chats for extra deep batches."""
    scored_dialogs: list[tuple[float, float, int, str, DialogInfo]] = []
    for dialog in dialogs:
        chat_state = state.peek_chat(dialog.chat_id)
        if chat_state is None:
            continue
        if chat_state.scored_count < config.focus_min_scored:
            continue
        priority = chat_state.focus_score(config.focus_min_scored)
        if priority <= 0:
            continue
        scored_dialogs.append(
            (
                priority,
                chat_state.avg_score,
                chat_state.scored_count,
                str(dialog.chat_id),
                dialog,
            )
        )

    scored_dialogs.sort(reverse=True)
    return [item[-1] for item in scored_dialogs[: config.focus_top_chats]]


def cache_evict_sort_key(row: dict[str, object]) -> tuple[float, float, str]:
    """Lower-score rows should be evicted first."""
    status = str(row.get("status") or "")
    score = row.get("score")
    updated_at = str(row.get("updated_at") or "")
    if status != "completed":
        return (-2.0, -2.0, updated_at)
    if score is None:
        return (-1.0, -1.0, updated_at)
    return (0.0, float(score), updated_at)


def list_cache_files(cache_root: Path) -> list[Path]:
    """List all current cache files on disk."""
    return sorted(
        path
        for path in cache_root.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def migrate_date_layout(root: Path, store: InferenceStore) -> int:
    """Flatten old chat/date/file layouts into chat/file."""
    migrated = 0
    for file_path in sorted(
        path for path in root.rglob("*") if path.is_file() or path.is_symlink()
    ):
        date_dir = file_path.parent
        if not DATE_DIR_RE.match(date_dir.name):
            continue
        new_path = date_dir.parent / file_path.name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if new_path.exists() or new_path.is_symlink():
            new_path.unlink()
        file_path.replace(new_path)
        store.rename_path(file_path, new_path)
        prune_empty_parents(date_dir, root)
        migrated += 1

    if migrated > 0:
        print_progress("layout-migrate", root=root, files=migrated)
    return migrated


def _move_tree_with_rename(source: Path, destination: Path) -> int:
    """Move file-like entries from source into destination preserving relative paths."""
    moved = 0
    for source_path in sorted(
        source.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if source_path.is_dir():
            continue
        relative = source_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        source_path.replace(target)
        moved += 1

    for directory in sorted(
        (path for path in source.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass

    try:
        source.rmdir()
    except OSError:
        pass
    return moved


def migrate_root_path(
    old_root: Path,
    new_root: Path,
    store: InferenceStore,
) -> int:
    """Move legacy root media tree to new root and update db rows."""
    old_root = old_root.expanduser().resolve()
    new_root = new_root.expanduser().resolve()
    if old_root == new_root:
        return 0

    updated_rows = store.replace_root_prefix(old_root, new_root)
    moved_files = 0
    if not old_root.exists():
        return updated_rows

    if new_root.exists():
        moved_files = _move_tree_with_rename(old_root, new_root)
    else:
        moved_files = sum(
            1 for path in old_root.rglob("*") if path.is_file() or path.is_symlink()
        )
        if moved_files:
            old_root.replace(new_root)

    if moved_files > 0 or updated_rows > 0:
        print_progress(
            "migrate-root",
            old_root=old_root,
            new_root=new_root,
            moved_files=moved_files,
            updated_rows=updated_rows,
        )

    return moved_files + updated_rows


def migrate_legacy_media_roots(config: DownloadConfig, store: InferenceStore) -> int:
    """Migrate legacy project-root media roots to configured roots."""
    migrated = 0
    migrated += migrate_root_path(
        old_root=DEFAULT_LEGACY_TARGET_ROOT,
        new_root=config.save_root,
        store=store,
    )
    migrated += migrate_root_path(
        old_root=DEFAULT_LEGACY_CACHE_ROOT,
        new_root=config.cache_root,
        store=store,
    )
    if config.flat_links_root is not None:
        migrated += migrate_root_path(
            old_root=DEFAULT_LEGACY_FLAT_LINKS_ROOT,
            new_root=config.flat_links_root,
            store=store,
        )
    return migrated


def prune_empty_parents(path: Path, stop_root: Path) -> None:
    """Remove empty parent directories after a gated delete."""
    current = path
    stop_root = stop_root.resolve()
    while True:
        try:
            resolved = current.resolve()
        except FileNotFoundError:
            resolved = current

        if resolved == stop_root:
            return
        if not current.exists():
            current = current.parent
            continue
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def materialize_target_file(source: Path, target: Path, mode: str) -> bool:
    """Materialize a scored cache file into the target directory."""
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        try:
            if target.samefile(source):
                return False
        except FileNotFoundError:
            pass
        target.unlink()

    if mode == "hardlink":
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        return True
    if mode == "symlink":
        os.symlink(source, target)
        return True
    if mode == "copy":
        shutil.copy2(source, target)
        return True
    raise ValueError(f"Unsupported target_mode: {mode}")


def ensure_complete_file(path: Path, expected_size: int) -> None:
    """Verify a downloaded file size if Telegram reported one."""
    if expected_size <= 0:
        return
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"downloaded size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )


async def refetch_message(
    client: pyrogram.Client,
    message: pyrogram.types.Message,
) -> pyrogram.types.Message:
    """Refetch a message when its file reference expires."""
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        return message

    refreshed = await client.get_messages(chat_id=chat_id, message_ids=message.id)
    if refreshed and not getattr(refreshed, "empty", False):
        return refreshed
    return message


async def download_message_media(
    client: pyrogram.Client,
    message: pyrogram.types.Message,
    target_path: Path,
    expected_size: int,
    *,
    protected: bool = False,
) -> tuple[Path, bool]:
    """Download one Telegram media file with a small retry loop."""
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path, False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.name}.part")
    if temp_path.exists():
        temp_path.unlink()

    current_message = message
    last_error: Exception | None = None
    for attempt in range(MEDIA_RETRY_COUNT):
        if protected or attempt > 0:
            current_message = await refetch_message(client, current_message)

        sleep_seconds = 0.0
        primary_source = resolve_download_input(current_message)
        download_sources = [primary_source]
        refresh_before_retry = False

        for download_source in download_sources:
            try:
                download_path = await client.download_media(
                    download_source,
                    file_name=str(temp_path),
                )
                if not download_path or not isinstance(download_path, str):
                    raise RuntimeError("download_media returned no file path")

                temp_result = Path(download_path)
                ensure_complete_file(temp_result, expected_size)
                temp_result.replace(target_path)
                return target_path, True
            except FloodWait as exc:
                last_error = exc
                sleep_seconds = max(sleep_seconds, float(exc.value))
                break
            except FileReferenceExpired as exc:
                last_error = exc
                sleep_seconds = max(sleep_seconds, 1.0)
                refresh_before_retry = True
                break
            except BadRequest as exc:
                last_error = exc
                sleep_seconds = max(sleep_seconds, 1.0)
                refresh_before_retry = True
                break
            except RPCError as exc:
                last_error = exc
                sleep_seconds = max(sleep_seconds, 1.0)
                break
            except RuntimeError as exc:
                last_error = exc
                sleep_seconds = max(sleep_seconds, 1.0)
                refresh_before_retry = True
                break
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                sleep_seconds = max(sleep_seconds, 1.0)
                break
            finally:
                if temp_path.exists() and not target_path.exists():
                    try:
                        ensure_complete_file(temp_path, expected_size)
                    except Exception:  # pylint: disable=broad-except
                        temp_path.unlink()

        if target_path.exists():
            return target_path, True
        if refresh_before_retry:
            current_message = await refetch_message(client, current_message)
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)
        if attempt == MEDIA_RETRY_COUNT - 1 and last_error is not None:
            break

    if temp_path.exists():
        temp_path.unlink()
    raise RuntimeError(
        f"failed to download message {message.id} after {MEDIA_RETRY_COUNT} attempts: {last_error}"
    )


def print_progress(prefix: str, **fields: object) -> None:
    """Print a compact single-line progress update."""
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[{prefix}] {suffix}".strip(), flush=True)


def format_exception(exc: BaseException) -> str:
    """Render one exception as ClassName: message."""
    return f"{type(exc).__name__}: {exc}"


def is_recoverable_network_error(exc: BaseException) -> bool:
    """Return whether one exception likely came from a broken transport/proxy."""
    return isinstance(exc, (OSError, ConnectionError, TimeoutError))


async def score_path(
    path: Path,
    media_type: str,
    engine: FrozenClipEngine,
    store: InferenceStore,
    checkpoint_path: Path,
    score_semaphore: asyncio.Semaphore | None = None,
) -> tuple[float, str, bool]:
    """Score a media path or reuse an existing cached hash result."""
    content_hash = await asyncio.to_thread(compute_content_hash, path)
    cached = store.get_completed_by_hash(content_hash, str(checkpoint_path))
    if cached is not None and cached["score"] is not None:
        score = float(cached["score"])
        cache_hit = True
    else:
        if score_semaphore is None:
            score = await asyncio.to_thread(
                engine.score_video if media_type == "video" else engine.score_image,
                path,
            )
        else:
            async with score_semaphore:
                score = await asyncio.to_thread(
                    engine.score_video if media_type == "video" else engine.score_image,
                    path,
                )
        cache_hit = False

    return score, content_hash, cache_hit


async def process_one_message(
    client: pyrogram.Client,
    message: pyrogram.types.Message,
    config: DownloadConfig,
    state: DownloaderState,
    store: InferenceStore,
    engine: FrozenClipEngine,
    stats: RunStats,
    score_semaphore: asyncio.Semaphore | None = None,
) -> None:
    """Download, score, and gate one message."""
    chat = message.chat
    chat_id = getattr(chat, "id", None)
    if chat_id is None or getattr(message, "empty", False):
        stats.skipped += 1
        return

    chat_title = display_chat_name(chat)
    chat_type = normalize_chat_type(getattr(chat, "type", None)) or "unknown"
    protected = is_protected_message(message)
    stats.messages_seen += 1
    if protected:
        stats.protected_seen += 1

    candidate = resolve_media_candidate(message)
    if candidate is None:
        state.mark_processed(
            chat_id=chat_id,
            title=chat_title,
            chat_type=chat_type,
            message_id=message.id,
            skipped=True,
            protected=protected,
        )
        persist_chat_stats(store, state, chat_id)
        stats.skipped += 1
        return

    cache_path = build_target_path(config.cache_root, message, candidate)
    downloaded = False
    try:
        _, downloaded = await download_message_media(
            client=client,
            message=message,
            target_path=cache_path,
            expected_size=candidate.expected_size,
            protected=protected,
        )
        if downloaded:
            state.mark_downloaded(
                chat_id=chat_id,
                title=chat_title,
                chat_type=chat_type,
                protected=protected,
            )
            stats.downloads += 1

        score, content_hash, cache_hit = await score_path(
            path=cache_path,
            media_type=candidate.media_type,
            engine=engine,
            store=store,
            checkpoint_path=config.checkpoint_path,
            score_semaphore=score_semaphore,
        )
        stats.scored += 1
        if cache_hit:
            stats.cache_hits += 1
    except Exception as exc:  # pylint: disable=broad-except
        error_text = f"{type(exc).__name__}: {exc}"
        failed_hash = f"failed:{chat_id}:{message.id}"
        if cache_path.exists():
            try:
                failed_hash = compute_content_hash(cache_path)
            except Exception:  # pylint: disable=broad-except
                pass
        store.write_result(
            media_path=cache_path,
            media_type=candidate.media_type,
            content_hash=failed_hash,
            checkpoint_path=config.checkpoint_path,
            score=None,
            status="failed",
            error=error_text,
        )
        state.mark_failed(
            chat_id=chat_id,
            title=chat_title,
            chat_type=chat_type,
            message_id=message.id,
            error=error_text,
            protected=protected,
        )
        persist_chat_stats(store, state, chat_id)
        stats.failed += 1
        if protected:
            stats.protected_failed += 1
        print_progress(
            "failed",
            chat_id=chat_id,
            message_id=message.id,
            protected=int(protected),
            error=error_text,
        )
        return

    store.write_result(
        media_path=cache_path,
        media_type=candidate.media_type,
        content_hash=content_hash,
        checkpoint_path=config.checkpoint_path,
        score=score,
        status="completed",
    )

    canonical_target_path = build_target_path(config.save_root, message, candidate)
    canonical_flat_target_path = (
        build_flat_target_path(config.flat_links_root, message, candidate)
        if config.flat_links_root is not None
        else None
    )
    if score >= config.min_score or config.keep_below_threshold:
        target_path = build_target_path(
            config.save_root,
            message,
            candidate,
            score_prefix=score,
        )
        flat_target_path = (
            build_flat_target_path(
                config.flat_links_root,
                message,
                candidate,
                score_prefix=score,
            )
            if config.flat_links_root is not None
            else None
        )
        try:
            delete_target_variants(
                canonical_target_path,
                config.save_root,
                store,
                keep_path=target_path,
            )
            if materialize_target_file(cache_path, target_path, config.target_mode):
                stats.target_materialized += 1
            store.write_result(
                media_path=target_path,
                media_type=candidate.media_type,
                content_hash=content_hash,
                checkpoint_path=config.checkpoint_path,
                score=score,
                status="completed",
            )
            if flat_target_path is not None and canonical_flat_target_path is not None:
                delete_target_variants(
                    canonical_flat_target_path,
                    config.flat_links_root,
                    store,
                    keep_path=flat_target_path,
                )
                if materialize_target_file(target_path, flat_target_path, "symlink"):
                    stats.flat_materialized += 1
                store.write_result(
                    media_path=flat_target_path,
                    media_type=candidate.media_type,
                    content_hash=content_hash,
                    checkpoint_path=config.checkpoint_path,
                    score=score,
                    status="completed",
                )
        except Exception as exc:  # pylint: disable=broad-except
            error_text = f"{type(exc).__name__}: {exc}"
            failure_path = flat_target_path or target_path
            store.write_result(
                media_path=failure_path,
                media_type=candidate.media_type,
                content_hash=content_hash,
                checkpoint_path=config.checkpoint_path,
                score=None,
                status="failed",
                error=error_text,
            )
            state.mark_failed(
                chat_id=chat_id,
                title=chat_title,
                chat_type=chat_type,
                message_id=message.id,
                error=error_text,
                protected=protected,
            )
            persist_chat_stats(store, state, chat_id)
            stats.failed += 1
            if protected:
                stats.protected_failed += 1
            print_progress(
                "target-failed",
                chat_id=chat_id,
                message_id=message.id,
                protected=int(protected),
                target=failure_path,
                error=error_text,
            )
            return

        state.mark_processed(
            chat_id=chat_id,
            title=chat_title,
            chat_type=chat_type,
            message_id=message.id,
            score=score,
            kept=True,
            protected=protected,
        )
        persist_chat_stats(store, state, chat_id)
        stats.kept += 1
        if protected:
            stats.protected_kept += 1
        print_progress(
            "keep",
            score=f"{score:.4f}",
            cache_hit=int(cache_hit),
            protected=int(protected),
            cache=cache_path,
            target=target_path,
            flat=flat_target_path,
        )
        return

    delete_target_variants(canonical_target_path, config.save_root, store)
    if canonical_flat_target_path is not None and config.flat_links_root is not None:
        delete_target_variants(
            canonical_flat_target_path,
            config.flat_links_root,
            store,
        )
    stats.deleted += 1
    state.mark_processed(
        chat_id=chat_id,
        title=chat_title,
        chat_type=chat_type,
        message_id=message.id,
        score=score,
        deleted=True,
        protected=protected,
    )
    persist_chat_stats(store, state, chat_id)
    print_progress(
        "reject",
        score=f"{score:.4f}",
        cache_hit=int(cache_hit),
        protected=int(protected),
        cache=cache_path,
    )


async def fetch_retry_messages(
    client: pyrogram.Client,
    chat_id: int | str,
    message_ids: list[int],
) -> list[pyrogram.types.Message]:
    """Fetch failed messages that need retrying."""
    if not message_ids:
        return []
    messages = await client.get_messages(chat_id=chat_id, message_ids=message_ids)
    if isinstance(messages, list):
        return [item for item in messages if item and not getattr(item, "empty", False)]
    if messages and not getattr(messages, "empty", False):
        return [messages]
    return []


async def collect_chat_batch(
    client: pyrogram.Client,
    dialog: DialogInfo,
    config: DownloadConfig,
    state: DownloaderState,
) -> list[pyrogram.types.Message]:
    """Collect one chat visit, counting only scoreable media toward the batch size."""
    chat_state = state.get_chat(
        chat_id=dialog.chat_id,
        title=dialog.title,
        chat_type=dialog.chat_type,
    )
    media_batch_limit = max(1, int(config.chat_batch_size))
    history_limit = (
        max(1, int(config.history_limit))
        if config.history_limit is not None
        else None
    )

    messages: list[pyrogram.types.Message] = []
    seen_ids: set[int] = set()
    media_count = 0

    def append_message(message: pyrogram.types.Message) -> bool:
        nonlocal media_count
        message_id = getattr(message, "id", None)
        if message_id in seen_ids:
            return False
        seen_ids.add(message_id)
        messages.append(message)
        if is_scoreable_message(message):
            media_count += 1
        return True

    def reached_batch_limit() -> bool:
        if media_count >= media_batch_limit:
            return True
        if history_limit is not None and len(messages) >= history_limit:
            return True
        return False

    retry_limit = (
        min(media_batch_limit, history_limit)
        if history_limit is not None
        else media_batch_limit
    )
    retry_ids = state.pending_retry_ids(dialog.chat_id)[:retry_limit]
    if retry_ids:
        for message in await fetch_retry_messages(client, dialog.chat_id, retry_ids):
            if not append_message(message):
                continue
            if reached_batch_limit():
                return messages

    if reached_batch_limit():
        return messages

    history_remaining = (
        max(0, history_limit - len(messages))
        if history_limit is not None
        else None
    )
    if history_remaining == 0:
        return messages

    async for message in get_chat_history_v2(
        client,
        dialog.chat_id,
        limit=history_remaining or 0,
        offset_id=chat_state.last_read_message_id,
        reverse=True,
    ):
        if not append_message(message):
            continue
        if reached_batch_limit():
            break

    return messages


def can_evict_cache_entry(
    row: dict[str, object],
    config: DownloadConfig,
    store: InferenceStore,
) -> bool:
    """Avoid evicting cache files that back live symlink targets."""
    if config.target_mode != "symlink":
        return True
    content_hash = str(row.get("content_hash") or "")
    if not content_hash:
        return True
    return not store.has_existing_media_with_hash_under_root(content_hash, config.save_root)


def resolve_cache_max_items(
    config: DownloadConfig,
    store: InferenceStore,
) -> int:
    """Resolve cache limit, scaling automatically by scored media count when 0."""
    configured_limit = int(config.cache_max_items)
    if configured_limit > 0:
        return configured_limit

    scored_count = store.get_completed_count(config.checkpoint_path)
    if scored_count <= 0:
        return 0
    dynamic_limit = max(1, (scored_count * 5 + 99) // 100)
    return min(1500, dynamic_limit)


def enforce_cache_limit(
    config: DownloadConfig,
    store: InferenceStore,
    stats: RunStats,
) -> int:
    """Keep only the top-scoring cache files on disk."""
    max_items = resolve_cache_max_items(config, store)
    if max_items <= 0:
        return 0

    current_files = list_cache_files(config.cache_root)
    overflow = len(current_files) - max_items
    if overflow <= 0:
        return 0

    rows = store.list_results_under_root(config.cache_root)
    row_by_path = {
        Path(row["media_path"]).resolve(): {
            "media_path": row["media_path"],
            "content_hash": row["content_hash"],
            "score": row["score"],
            "status": row["status"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    }

    candidates: list[tuple[tuple[float, float, str], Path, dict[str, object]]] = []
    for file_path in current_files:
        resolved = file_path.resolve()
        row = row_by_path.get(resolved)
        payload = row or {
            "media_path": str(resolved),
            "content_hash": "",
            "score": None,
            "status": "unknown",
            "updated_at": "",
        }
        if not can_evict_cache_entry(payload, config, store):
            continue
        candidates.append((cache_evict_sort_key(payload), file_path, payload))

    candidates.sort(key=lambda item: item[0])

    evicted = 0
    for _, file_path, payload in candidates:
        if evicted >= overflow:
            break
        if file_path.exists() or file_path.is_symlink():
            file_path.unlink()
            prune_empty_parents(file_path.parent, config.cache_root)
            evicted += 1
            stats.cache_evicted += 1
            print_progress(
                "cache-evict",
                score=(
                    "na"
                    if payload.get("score") is None
                    else f"{float(payload['score']):.4f}"
                ),
                path=file_path,
            )

    if evicted < overflow:
        print_progress(
            "cache-cap",
            wanted=max_items,
            current=len(list_cache_files(config.cache_root)),
            blocked=overflow - evicted,
        )
    return evicted


async def process_chat_batch(
    client: pyrogram.Client,
    dialog: DialogInfo,
    config: DownloadConfig,
    state: DownloaderState,
    store: InferenceStore,
    engine: FrozenClipEngine,
    stats: RunStats,
    score_semaphore: asyncio.Semaphore | None = None,
    *,
    stage: str,
) -> None:
    """Process one small batch from one discovered chat."""
    chat_state = state.get_chat(
        chat_id=dialog.chat_id,
        title=dialog.title,
        chat_type=dialog.chat_type,
        has_protected_content=dialog.has_protected_content,
    )
    print_progress(
        "chat",
        id=dialog.chat_id,
        type=dialog.chat_type,
        title=dialog.title,
        protected=int(chat_state.has_protected_content),
        protected_seen=chat_state.protected_processed_count,
        last_read=chat_state.last_read_message_id,
        retry=len(chat_state.failed_message_ids),
        avg=(
            f"{chat_state.avg_score:.4f}"
            if chat_state.scored_count > 0
            else "na"
        ),
        stage=stage,
    )

    batch = await collect_chat_batch(client, dialog, config, state)
    if not batch:
        return

    scored_before = stats.scored
    processed_in_chat = 0
    progress_lock = asyncio.Lock()
    message_concurrency = max(1, int(config.message_concurrency))

    async def process_with_progress(message: pyrogram.types.Message) -> None:
        nonlocal processed_in_chat
        try:
            await process_one_message(
                client=client,
                message=message,
                config=config,
                state=state,
                store=store,
                engine=engine,
                stats=stats,
                score_semaphore=score_semaphore,
            )
        except Exception as exc:  # pylint: disable=broad-except
            chat = getattr(message, "chat", None)
            chat_id = getattr(chat, "id", dialog.chat_id)
            chat_title = display_chat_name(chat) if chat is not None else dialog.title
            chat_type = (
                normalize_chat_type(getattr(chat, "type", None))
                if chat is not None
                else dialog.chat_type
            ) or dialog.chat_type
            error_text = f"{type(exc).__name__}: {exc}"
            state.mark_failed(
                chat_id=chat_id,
                title=chat_title,
                chat_type=chat_type,
                message_id=message.id,
                error=error_text,
                protected=is_protected_message(message),
            )
            persist_chat_stats(store, state, chat_id)
            stats.failed += 1
            print_progress(
                "message-crashed",
                chat_id=chat_id,
                message_id=message.id,
                stage=stage,
                error=error_text,
            )
        finally:
            async with progress_lock:
                processed_in_chat += 1
                if config.log_every > 0 and processed_in_chat % config.log_every == 0:
                    print_progress(
                        "progress",
                        chat_id=dialog.chat_id,
                        processed=processed_in_chat,
                        seen=stats.messages_seen,
                        kept=stats.kept,
                        deleted=stats.deleted,
                        failed=stats.failed,
                    )

    if message_concurrency <= 1 or len(batch) <= 1:
        for message in batch:
            await process_with_progress(message)
    else:
        semaphore = asyncio.Semaphore(message_concurrency)

        async def run_one(message: pyrogram.types.Message) -> None:
            async with semaphore:
                await process_with_progress(message)

        await asyncio.gather(*(run_one(message) for message in batch))

    state.mark_batch(
        chat_id=dialog.chat_id,
        title=dialog.title,
        chat_type=dialog.chat_type,
        message_count=processed_in_chat,
        scored_count=stats.scored - scored_before,
    )
    persist_chat_stats(store, state, dialog.chat_id)
    enforce_cache_limit(config, store, stats)
    stats.chat_batches += 1
    if stage == "focus":
        stats.focus_batches += 1

    chat_state = state.get_chat(
        chat_id=dialog.chat_id,
        title=dialog.title,
        chat_type=dialog.chat_type,
        has_protected_content=dialog.has_protected_content,
    )
    print_progress(
        "batch",
        stage=stage,
        chat_id=dialog.chat_id,
        messages=processed_in_chat,
        scored=stats.scored - scored_before,
        protected=int(chat_state.has_protected_content),
        protected_seen=chat_state.protected_processed_count,
        chat_avg=(
            f"{chat_state.avg_score:.4f}"
            if chat_state.scored_count > 0
            else "na"
        ),
        cache_items=len(list_cache_files(config.cache_root)),
    )


async def discover_dialogs(
    client: pyrogram.Client,
    config: DownloadConfig,
    state: DownloaderState,
    store: InferenceStore,
) -> list[DialogInfo]:
    """Enumerate dialogs visible to the authenticated Telegram account."""
    if not hasattr(client, "get_dialogs"):
        raise AttributeError("Current pyrogram client does not support get_dialogs().")

    include_types = set(config.discover_chat_types)
    dialogs: list[DialogInfo] = []
    protected_count = 0
    async for dialog in client.get_dialogs():
        chat = getattr(dialog, "chat", None)
        chat_id = getattr(chat, "id", None)
        if chat is None or chat_id is None:
            continue

        chat_type = normalize_chat_type(getattr(chat, "type", None))
        if include_types and chat_type not in include_types:
            continue

        title = display_chat_name(chat)
        has_protected_content = bool(getattr(chat, "has_protected_content", False))
        if has_protected_content:
            protected_count += 1
        dialogs.append(
            DialogInfo(
                chat_id=chat_id,
                title=title,
                chat_type=chat_type,
                has_protected_content=has_protected_content,
            )
        )
        state.get_chat(
            chat_id=chat_id,
            title=title,
            chat_type=chat_type,
            has_protected_content=has_protected_content,
        )

    for dialog in dialogs:
        persist_chat_stats(store, state, dialog.chat_id)
    state.save()

    stats_count = len(dialogs)
    print_progress(
        "discover",
        dialogs=stats_count,
        protected=protected_count,
        chat_types=",".join(config.discover_chat_types),
    )
    return dialogs


def build_client(config: DownloadConfig) -> pyrogram.Client:
    """Construct the pyrogram client for this run."""
    config.session_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "name": config.session_name,
        "api_id": config.api_id,
        "api_hash": config.api_hash,
        "workdir": str(config.session_dir),
    }
    if config.session_string:
        kwargs["session_string"] = config.session_string
    if config.proxy:
        kwargs["proxy"] = config.proxy
    return pyrogram.Client(**kwargs)


async def restart_client_session(
    client: pyrogram.Client,
    config: DownloadConfig,
    *,
    context: str,
    error: BaseException,
) -> pyrogram.Client:
    """Recreate the Telegram client after one recoverable network failure."""
    error_text = format_exception(error)
    attempt = 0
    stopped_client = False
    while True:
        attempt += 1
        delay_seconds = min(
            NETWORK_RESTART_DELAY_SECONDS * attempt,
            NETWORK_RESTART_MAX_DELAY_SECONDS,
        )
        print_progress(
            "network-recover",
            context=context,
            attempt=attempt,
            delay=f"{delay_seconds:.1f}s",
            error=error_text,
        )
        if not stopped_client:
            try:
                await client.stop()
            except Exception:  # pylint: disable=broad-except
                pass
            stopped_client = True
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        replacement = build_client(config)
        try:
            await replacement.start()
            return replacement
        except AuthKeyUnregistered as exc:
            session_path = config.session_dir / f"{config.session_name}.session"
            raise RuntimeError(
                "Telegram session is no longer valid. "
                f"Delete or replace {session_path} and login again."
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            error_text = format_exception(exc)
            try:
                await replacement.stop()
            except Exception:  # pylint: disable=broad-except
                pass
            if not is_recoverable_network_error(exc):
                raise


async def _run_async(config: DownloadConfig) -> RunStats:
    """Internal async runner."""
    config.save_root.mkdir(parents=True, exist_ok=True)
    config.cache_root.mkdir(parents=True, exist_ok=True)
    if config.flat_links_root is not None:
        config.flat_links_root.mkdir(parents=True, exist_ok=True)
    config.state_path.parent.mkdir(parents=True, exist_ok=True)

    state = DownloaderState.load(config.state_path)
    store = InferenceStore(config.db_path)
    migrate_legacy_media_roots(config, store)
    migrate_date_layout(config.cache_root, store)
    migrate_date_layout(config.save_root, store)
    sync_target_root(config, store)
    sync_flat_links_root(config, store)
    stats = RunStats()
    score_semaphore = asyncio.Semaphore(max(1, int(config.score_concurrency)))
    client = build_client(config)
    try:
        try:
            await client.start()
        except AuthKeyUnregistered as exc:
            session_path = config.session_dir / f"{config.session_name}.session"
            raise RuntimeError(
                "Telegram session is no longer valid. "
                f"Delete or replace {session_path} and login again."
            ) from exc

        engine = FrozenClipEngine(device=config.device)
        engine.load_model(config.checkpoint_path)
        while True:
            stats.rounds += 1
            try:
                dialogs = await discover_dialogs(client, config, state, store)
            except Exception as exc:  # pylint: disable=broad-except
                if not is_recoverable_network_error(exc):
                    raise
                stats.rounds -= 1
                client = await restart_client_session(
                    client,
                    config,
                    context="discover",
                    error=exc,
                )
                continue
            stats.chats_discovered = len(dialogs)
            breadth_dialogs = order_dialogs_for_breadth(dialogs, state, config.max_chats)
            round_processed = 0

            print_progress(
                "round",
                index=stats.rounds,
                mode="breadth",
                dialogs=len(breadth_dialogs),
            )

            for dialog in breadth_dialogs:
                try:
                    before_batches = stats.chat_batches
                    await process_chat_batch(
                        client=client,
                        dialog=dialog,
                        config=config,
                        state=state,
                        store=store,
                        engine=engine,
                        stats=stats,
                        score_semaphore=score_semaphore,
                        stage="breadth",
                    )
                    if stats.chat_batches > before_batches:
                        round_processed += 1
                except Exception as exc:  # pylint: disable=broad-except
                    error_text = format_exception(exc)
                    chat_state = state.get_chat(dialog.chat_id, dialog.title, dialog.chat_type)
                    chat_state.last_error = error_text
                    persist_chat_stats(store, state, dialog.chat_id)
                    state.save()
                    if is_recoverable_network_error(exc):
                        client = await restart_client_session(
                            client,
                            config,
                            context=f"chat:{dialog.chat_id}:breadth",
                            error=exc,
                        )
                        continue
                    stats.failed += 1
                    print_progress(
                        "chat-failed",
                        id=dialog.chat_id,
                        type=dialog.chat_type,
                        stage="breadth",
                        error=error_text,
                    )
                if config.chat_idle_seconds > 0:
                    await asyncio.sleep(config.chat_idle_seconds)

            if stats.rounds > config.breadth_rounds:
                focus_dialogs = select_focus_dialogs(dialogs, state, config)
                if focus_dialogs:
                    print_progress(
                        "round",
                        index=stats.rounds,
                        mode="focus",
                        dialogs=len(focus_dialogs),
                    )
                for dialog in focus_dialogs:
                    try:
                        before_batches = stats.chat_batches
                        await process_chat_batch(
                            client=client,
                            dialog=dialog,
                            config=config,
                        state=state,
                        store=store,
                        engine=engine,
                        stats=stats,
                        score_semaphore=score_semaphore,
                        stage="focus",
                    )
                        if stats.chat_batches > before_batches:
                            round_processed += 1
                    except Exception as exc:  # pylint: disable=broad-except
                        error_text = format_exception(exc)
                        chat_state = state.get_chat(dialog.chat_id, dialog.title, dialog.chat_type)
                        chat_state.last_error = error_text
                        persist_chat_stats(store, state, dialog.chat_id)
                        state.save()
                        if is_recoverable_network_error(exc):
                            client = await restart_client_session(
                                client,
                                config,
                                context=f"chat:{dialog.chat_id}:focus",
                                error=exc,
                            )
                            continue
                        stats.failed += 1
                        print_progress(
                            "chat-failed",
                            id=dialog.chat_id,
                            type=dialog.chat_type,
                            stage="focus",
                            error=error_text,
                        )
                    if config.chat_idle_seconds > 0:
                        await asyncio.sleep(config.chat_idle_seconds)

            if not config.continuous:
                break

            if round_processed <= 0 and config.round_idle_seconds > 0:
                print_progress(
                    "idle",
                    round=stats.rounds,
                    sleep=f"{config.round_idle_seconds:.1f}s",
                )
                await asyncio.sleep(config.round_idle_seconds)
    finally:
        state.save()
        store.close()
        try:
            await client.stop()
        except Exception:  # pylint: disable=broad-except
            pass

    print_progress(
        "summary",
        rounds=stats.rounds,
        chat_batches=stats.chat_batches,
        focus_batches=stats.focus_batches,
        dialogs=stats.chats_discovered,
        seen=stats.messages_seen,
        downloads=stats.downloads,
        scored=stats.scored,
        kept=stats.kept,
        deleted=stats.deleted,
        skipped=stats.skipped,
        failed=stats.failed,
        cache_hits=stats.cache_hits,
        materialized=stats.target_materialized,
        flat_materialized=stats.flat_materialized,
        cache_evicted=stats.cache_evicted,
        protected_seen=stats.protected_seen,
        protected_kept=stats.protected_kept,
        protected_failed=stats.protected_failed,
    )
    return stats


def run_gated_download(config: DownloadConfig) -> RunStats:
    """Run the local Telegram global-walk gated downloader."""
    return asyncio.run(_run_async(config))

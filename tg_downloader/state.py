"""State persistence for incremental Telegram gated downloads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ChatState:
    """Incremental progress for one chat."""

    chat_id: str
    title: str = ""
    chat_type: str = ""
    has_protected_content: bool = False
    last_read_message_id: int = 0
    failed_message_ids: list[int] = field(default_factory=list)
    downloaded_count: int = 0
    processed_count: int = 0
    kept_count: int = 0
    deleted_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    protected_downloaded_count: int = 0
    protected_processed_count: int = 0
    protected_kept_count: int = 0
    protected_failed_count: int = 0
    batch_count: int = 0
    scored_count: int = 0
    score_sum: float = 0.0
    avg_score: float = 0.0
    min_score: float | None = None
    max_score: float | None = None
    last_batch_message_count: int = 0
    last_batch_scored_count: int = 0
    last_error: str = ""
    updated_at: str = field(default_factory=utcnow_iso)

    @classmethod
    def from_dict(cls, chat_id: str, payload: dict[str, Any]) -> "ChatState":
        """Create a chat state object from a JSON payload."""
        failed_ids = payload.get("failed_message_ids") or []
        return cls(
            chat_id=chat_id,
            title=str(payload.get("title") or ""),
            chat_type=str(payload.get("chat_type") or ""),
            has_protected_content=bool(payload.get("has_protected_content") or False),
            last_read_message_id=int(payload.get("last_read_message_id") or 0),
            failed_message_ids=sorted({int(item) for item in failed_ids}),
            downloaded_count=int(payload.get("downloaded_count") or 0),
            processed_count=int(payload.get("processed_count") or 0),
            kept_count=int(payload.get("kept_count") or 0),
            deleted_count=int(payload.get("deleted_count") or 0),
            skipped_count=int(payload.get("skipped_count") or 0),
            failed_count=int(payload.get("failed_count") or 0),
            protected_downloaded_count=int(
                payload.get("protected_downloaded_count") or 0
            ),
            protected_processed_count=int(
                payload.get("protected_processed_count") or 0
            ),
            protected_kept_count=int(payload.get("protected_kept_count") or 0),
            protected_failed_count=int(payload.get("protected_failed_count") or 0),
            batch_count=int(payload.get("batch_count") or 0),
            scored_count=int(payload.get("scored_count") or 0),
            score_sum=float(payload.get("score_sum") or 0.0),
            avg_score=float(payload.get("avg_score") or 0.0),
            min_score=(
                float(payload["min_score"])
                if payload.get("min_score") is not None
                else None
            ),
            max_score=(
                float(payload["max_score"])
                if payload.get("max_score") is not None
                else None
            ),
            last_batch_message_count=int(payload.get("last_batch_message_count") or 0),
            last_batch_scored_count=int(payload.get("last_batch_scored_count") or 0),
            last_error=str(payload.get("last_error") or ""),
            updated_at=str(payload.get("updated_at") or utcnow_iso()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize a chat state object into JSON-compatible data."""
        return {
            "title": self.title,
            "chat_type": self.chat_type,
            "has_protected_content": self.has_protected_content,
            "last_read_message_id": self.last_read_message_id,
            "failed_message_ids": self.failed_message_ids,
            "downloaded_count": self.downloaded_count,
            "processed_count": self.processed_count,
            "kept_count": self.kept_count,
            "deleted_count": self.deleted_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "protected_downloaded_count": self.protected_downloaded_count,
            "protected_processed_count": self.protected_processed_count,
            "protected_kept_count": self.protected_kept_count,
            "protected_failed_count": self.protected_failed_count,
            "batch_count": self.batch_count,
            "scored_count": self.scored_count,
            "score_sum": self.score_sum,
            "avg_score": self.avg_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "last_batch_message_count": self.last_batch_message_count,
            "last_batch_scored_count": self.last_batch_scored_count,
            "last_error": self.last_error,
            "updated_at": self.updated_at,
        }

    def focus_score(self, min_samples: int) -> float:
        """Compute a damped priority score used for focus scheduling."""
        if self.scored_count <= 0:
            return 0.0
        confidence = min(1.0, self.scored_count / max(1, min_samples))
        return self.avg_score * confidence


class DownloaderState:
    """JSON-backed incremental state store."""

    def __init__(self, path: Path, payload: dict[str, Any] | None = None):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.version = 1
        self.updated_at = utcnow_iso()
        self.chats: dict[str, ChatState] = {}

        if payload:
            self.version = int(payload.get("version") or 1)
            self.updated_at = str(payload.get("updated_at") or utcnow_iso())
            for chat_id, chat_payload in (payload.get("chats") or {}).items():
                self.chats[str(chat_id)] = ChatState.from_dict(str(chat_id), chat_payload)

    @classmethod
    def load(cls, path: Path) -> "DownloaderState":
        """Load state from disk if it exists."""
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            return cls(resolved)

        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return cls(resolved, payload)

    def _touch(self) -> None:
        self.updated_at = utcnow_iso()

    def get_chat(
        self,
        chat_id: int | str,
        title: str = "",
        chat_type: str = "",
        has_protected_content: bool | None = None,
    ) -> ChatState:
        """Return an existing chat state or create a new one."""
        key = str(chat_id)
        state = self.chats.get(key)
        if state is None:
            state = ChatState(chat_id=key, title=title, chat_type=chat_type)
            self.chats[key] = state

        if title:
            state.title = title
        if chat_type:
            state.chat_type = chat_type
        if has_protected_content is not None:
            state.has_protected_content = (
                state.has_protected_content or bool(has_protected_content)
            )
        state.updated_at = utcnow_iso()
        self._touch()
        return state

    def peek_chat(self, chat_id: int | str) -> ChatState | None:
        """Return chat state without mutating timestamps."""
        return self.chats.get(str(chat_id))

    def pending_retry_ids(self, chat_id: int | str) -> list[int]:
        """Return failed message ids that should be retried first."""
        return list(self.get_chat(chat_id).failed_message_ids)

    def aggregate_totals(self) -> dict[str, int]:
        """Return one global sum across all chats."""
        totals = {
            "chat_count": len(self.chats),
            "downloaded_count": 0,
            "processed_count": 0,
            "kept_count": 0,
            "deleted_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "scored_count": 0,
            "batch_count": 0,
            "protected_downloaded_count": 0,
            "protected_processed_count": 0,
            "protected_kept_count": 0,
            "protected_failed_count": 0,
        }
        for state in self.chats.values():
            totals["downloaded_count"] += state.downloaded_count
            totals["processed_count"] += state.processed_count
            totals["kept_count"] += state.kept_count
            totals["deleted_count"] += state.deleted_count
            totals["skipped_count"] += state.skipped_count
            totals["failed_count"] += state.failed_count
            totals["scored_count"] += state.scored_count
            totals["batch_count"] += state.batch_count
            totals["protected_downloaded_count"] += state.protected_downloaded_count
            totals["protected_processed_count"] += state.protected_processed_count
            totals["protected_kept_count"] += state.protected_kept_count
            totals["protected_failed_count"] += state.protected_failed_count
        return totals

    def mark_downloaded(
        self,
        chat_id: int | str,
        title: str,
        chat_type: str,
        *,
        protected: bool = False,
    ) -> None:
        """Record one media file that was actually downloaded from Telegram."""
        state = self.get_chat(chat_id, title=title, chat_type=chat_type)
        state.downloaded_count += 1
        if protected:
            state.has_protected_content = True
            state.protected_downloaded_count += 1
        state.updated_at = utcnow_iso()
        self._touch()
        self.save()

    def mark_processed(
        self,
        chat_id: int | str,
        title: str,
        chat_type: str,
        message_id: int,
        *,
        score: float | None = None,
        kept: bool = False,
        deleted: bool = False,
        skipped: bool = False,
        protected: bool = False,
    ) -> None:
        """Advance state after one message was fully handled."""
        state = self.get_chat(chat_id, title=title, chat_type=chat_type)
        state.last_read_message_id = max(state.last_read_message_id, int(message_id))
        state.failed_message_ids = [
            item for item in state.failed_message_ids if item != int(message_id)
        ]
        state.processed_count += 1
        if kept:
            state.kept_count += 1
        if deleted:
            state.deleted_count += 1
        if skipped:
            state.skipped_count += 1
        if protected:
            state.has_protected_content = True
            state.protected_processed_count += 1
            if kept:
                state.protected_kept_count += 1
        if score is not None:
            numeric_score = float(score)
            state.scored_count += 1
            state.score_sum += numeric_score
            state.avg_score = state.score_sum / state.scored_count
            state.min_score = (
                numeric_score
                if state.min_score is None
                else min(state.min_score, numeric_score)
            )
            state.max_score = (
                numeric_score
                if state.max_score is None
                else max(state.max_score, numeric_score)
            )
        state.last_error = ""
        state.updated_at = utcnow_iso()
        self._touch()
        self.save()

    def mark_failed(
        self,
        chat_id: int | str,
        title: str,
        chat_type: str,
        message_id: int,
        error: str,
        protected: bool = False,
    ) -> None:
        """Record a failed message for retry."""
        state = self.get_chat(chat_id, title=title, chat_type=chat_type)
        if int(message_id) not in state.failed_message_ids:
            state.failed_message_ids.append(int(message_id))
            state.failed_message_ids.sort()
        state.failed_count += 1
        if protected:
            state.has_protected_content = True
            state.protected_failed_count += 1
        state.last_error = error
        state.updated_at = utcnow_iso()
        self._touch()
        self.save()

    def mark_batch(
        self,
        chat_id: int | str,
        title: str,
        chat_type: str,
        message_count: int,
        scored_count: int,
    ) -> None:
        """Record one completed chat batch."""
        state = self.get_chat(chat_id, title=title, chat_type=chat_type)
        state.batch_count += 1
        state.last_batch_message_count = int(message_count)
        state.last_batch_scored_count = int(scored_count)
        state.updated_at = utcnow_iso()
        self._touch()
        self.save()

    def save(self) -> None:
        """Persist current state atomically."""
        payload = {
            "version": self.version,
            "updated_at": self.updated_at,
            "totals": self.aggregate_totals(),
            "chats": {
                chat_id: state.to_dict()
                for chat_id, state in sorted(self.chats.items(), key=lambda item: item[0])
            },
        }
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self.path)

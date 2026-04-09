"""SQLite helpers for gated Telegram downloads."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from src.models.frozen_clip_engine import FrozenClipEngine
from tg_downloader.state import ChatState

LOCKED_ERROR = "database is locked"
WRITE_RETRY_ATTEMPTS = 8
WRITE_RETRY_BASE_DELAY = 0.25

MEDIA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS media_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    file_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INFERENCE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS media_inference (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_path TEXT UNIQUE NOT NULL,
    media_type TEXT NOT NULL CHECK(media_type IN ('image', 'video')),
    content_hash TEXT NOT NULL,
    score REAL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'completed', 'failed')),
    error TEXT,
    model_checkpoint TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_media_inference_hash ON media_inference(content_hash);
CREATE INDEX IF NOT EXISTS idx_media_inference_score ON media_inference(score DESC);
CREATE INDEX IF NOT EXISTS idx_media_inference_status ON media_inference(status);
CREATE INDEX IF NOT EXISTS idx_media_inference_checkpoint ON media_inference(model_checkpoint);
"""

CHAT_STATS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS telegram_chat_stats (
    chat_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    chat_type TEXT NOT NULL DEFAULT '',
    has_protected_content INTEGER NOT NULL DEFAULT 0,
    last_read_message_id INTEGER NOT NULL DEFAULT 0,
    downloaded_count INTEGER NOT NULL DEFAULT 0,
    processed_count INTEGER NOT NULL DEFAULT 0,
    kept_count INTEGER NOT NULL DEFAULT 0,
    deleted_count INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    protected_downloaded_count INTEGER NOT NULL DEFAULT 0,
    protected_processed_count INTEGER NOT NULL DEFAULT 0,
    protected_kept_count INTEGER NOT NULL DEFAULT 0,
    protected_failed_count INTEGER NOT NULL DEFAULT 0,
    batch_count INTEGER NOT NULL DEFAULT 0,
    scored_count INTEGER NOT NULL DEFAULT 0,
    score_sum REAL NOT NULL DEFAULT 0,
    avg_score REAL,
    min_score REAL,
    max_score REAL,
    last_error TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tg_chat_stats_avg_score ON telegram_chat_stats(avg_score DESC);
CREATE INDEX IF NOT EXISTS idx_tg_chat_stats_scored_count ON telegram_chat_stats(scored_count DESC);
"""

CHAT_STATS_REQUIRED_COLUMNS = {
    "downloaded_count": "INTEGER NOT NULL DEFAULT 0",
    "has_protected_content": "INTEGER NOT NULL DEFAULT 0",
    "protected_downloaded_count": "INTEGER NOT NULL DEFAULT 0",
    "protected_processed_count": "INTEGER NOT NULL DEFAULT 0",
    "protected_kept_count": "INTEGER NOT NULL DEFAULT 0",
    "protected_failed_count": "INTEGER NOT NULL DEFAULT 0",
}


def ensure_table_columns(
    conn: sqlite3.Connection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    """Add missing sqlite columns for lightweight schema evolution."""
    existing = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column_name, column_def in columns.items():
        if column_name in existing:
            continue
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"
        )


def ensure_db(conn: sqlite3.Connection) -> None:
    """Create the media cache tables if they do not already exist."""
    conn.executescript(MEDIA_TABLE_SQL)
    conn.executescript(INFERENCE_TABLE_SQL)
    conn.executescript(CHAT_STATS_TABLE_SQL)
    ensure_table_columns(conn, "telegram_chat_stats", CHAT_STATS_REQUIRED_COLUMNS)
    conn.commit()


def detect_media_type(path: Path) -> str:
    """Infer media type from file extension."""
    suffix = path.suffix.lower()
    if suffix in FrozenClipEngine.VIDEO_EXTENSIONS:
        return "video"
    if suffix in FrozenClipEngine.IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"Unsupported media type: {path}")


def compute_content_hash(file_path: Path) -> str:
    """Hash a file using size plus the first chunk for fast dedupe."""
    file_size = file_path.stat().st_size
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        chunk = handle.read(65536)
        if chunk:
            hasher.update(f"{file_size}:".encode("utf-8") + chunk)
    return hasher.hexdigest()


def is_locked_error(exc: sqlite3.OperationalError) -> bool:
    """Check whether an sqlite error is caused by a locked database."""
    return LOCKED_ERROR in str(exc).lower()


def commit_with_retry(conn: sqlite3.Connection) -> None:
    """Commit a transaction with retry logic for busy sqlite writers."""
    for attempt in range(WRITE_RETRY_ATTEMPTS):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if not is_locked_error(exc) or attempt == WRITE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(WRITE_RETRY_BASE_DELAY * (attempt + 1))


def get_completed_count(conn: sqlite3.Connection, checkpoint_path: str) -> int:
    """Count completed scored media for one checkpoint."""
    cursor = conn.execute(
        """
        SELECT COUNT(*)
        FROM media_inference
        WHERE status = 'completed'
          AND score IS NOT NULL
          AND model_checkpoint = ?
        """,
        (checkpoint_path,),
    )
    return int(cursor.fetchone()[0] or 0)


def replace_path_prefix(
    conn: sqlite3.Connection,
    old_prefix: Path,
    new_prefix: Path,
) -> int:
    """Rewrite media_inference and media_files paths from old to new prefix."""
    old_prefix = old_prefix.expanduser().resolve()
    new_prefix = new_prefix.expanduser().resolve()
    if old_prefix == new_prefix:
        return 0

    old_root = str(old_prefix) + "/"
    new_root = str(new_prefix) + "/"
    updated = 0

    old_paths = conn.execute(
        "SELECT rowid, media_path FROM media_inference WHERE media_path LIKE ?",
        (f"{old_root}%",),
    ).fetchall()
    for row in old_paths:
        rowid = row["rowid"]
        old_value = row["media_path"]
        if not old_value.startswith(old_root):
            continue
        conn.execute(
            "UPDATE media_inference SET media_path = ? WHERE rowid = ?",
            (f"{new_root}{old_value[len(old_root):]}", rowid),
        )
        updated += 1

    old_paths = conn.execute(
        "SELECT rowid, path FROM media_files WHERE path LIKE ?",
        (f"{old_root}%",),
    ).fetchall()
    for row in old_paths:
        rowid = row["rowid"]
        old_value = row["path"]
        if not old_value.startswith(old_root):
            continue
        conn.execute(
            "UPDATE media_files SET path = ? WHERE rowid = ?",
            (f"{new_root}{old_value[len(old_root):]}", rowid),
        )
        updated += 1

    return updated


def insert_media_file(conn: sqlite3.Connection, media_path: str, media_type: str) -> None:
    """Upsert a path into the media file table."""
    conn.execute(
        "INSERT OR IGNORE INTO media_files (path, file_type) VALUES (?, ?)",
        (media_path, media_type),
    )


def upsert_inference(
    conn: sqlite3.Connection,
    media_path: str,
    media_type: str,
    content_hash: str,
    checkpoint_path: str,
    score: float | None,
    status: str,
    error: str | None,
) -> None:
    """Insert or update one inference row keyed by media path."""
    conn.execute(
        """
        INSERT INTO media_inference (
            media_path, media_type, content_hash, score, status, error, model_checkpoint
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(media_path) DO UPDATE SET
            media_type = excluded.media_type,
            content_hash = excluded.content_hash,
            score = excluded.score,
            status = excluded.status,
            error = excluded.error,
            model_checkpoint = excluded.model_checkpoint,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            media_path,
            media_type,
            content_hash,
            score,
            status,
            error,
            checkpoint_path,
        ),
    )


def write_result(
    conn: sqlite3.Connection,
    media_path: str,
    media_type: str,
    content_hash: str,
    checkpoint_path: str,
    score: float | None,
    status: str,
    error: str | None,
) -> None:
    """Persist one completed or failed inference result."""
    for attempt in range(WRITE_RETRY_ATTEMPTS):
        try:
            insert_media_file(conn, media_path, media_type)
            upsert_inference(
                conn=conn,
                media_path=media_path,
                media_type=media_type,
                content_hash=content_hash,
                checkpoint_path=checkpoint_path,
                score=score,
                status=status,
                error=error,
            )
            commit_with_retry(conn)
            return
        except sqlite3.OperationalError as exc:
            conn.rollback()
            if not is_locked_error(exc) or attempt == WRITE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(WRITE_RETRY_BASE_DELAY * (attempt + 1))


def delete_result(conn: sqlite3.Connection, media_path: str) -> None:
    """Delete cached rows for one media path."""
    media_tasks_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'media_tasks'"
    ).fetchone()
    conn.execute("DELETE FROM media_inference WHERE media_path = ?", (media_path,))
    conn.execute("DELETE FROM media_files WHERE path = ?", (media_path,))
    if media_tasks_exists:
        conn.execute("DELETE FROM media_tasks WHERE media_path = ?", (media_path,))
    commit_with_retry(conn)


def rename_result(conn: sqlite3.Connection, old_media_path: str, new_media_path: str) -> None:
    """Rename one cached media path while preserving score history."""
    conn.execute(
        "UPDATE media_inference SET media_path = ?, updated_at = CURRENT_TIMESTAMP WHERE media_path = ?",
        (new_media_path, old_media_path),
    )
    conn.execute(
        "UPDATE media_files SET path = ? WHERE path = ?",
        (new_media_path, old_media_path),
    )
    commit_with_retry(conn)


def upsert_chat_stats(conn: sqlite3.Connection, chat_state: ChatState) -> None:
    """Insert or update one aggregate Telegram chat stats row."""
    conn.execute(
        """
        INSERT INTO telegram_chat_stats (
            chat_id,
            title,
            chat_type,
            has_protected_content,
            last_read_message_id,
            downloaded_count,
            processed_count,
            kept_count,
            deleted_count,
            skipped_count,
            failed_count,
            protected_downloaded_count,
            protected_processed_count,
            protected_kept_count,
            protected_failed_count,
            batch_count,
            scored_count,
            score_sum,
            avg_score,
            min_score,
            max_score,
            last_error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chat_id) DO UPDATE SET
            title = excluded.title,
            chat_type = excluded.chat_type,
            has_protected_content = excluded.has_protected_content,
            last_read_message_id = excluded.last_read_message_id,
            downloaded_count = excluded.downloaded_count,
            processed_count = excluded.processed_count,
            kept_count = excluded.kept_count,
            deleted_count = excluded.deleted_count,
            skipped_count = excluded.skipped_count,
            failed_count = excluded.failed_count,
            protected_downloaded_count = excluded.protected_downloaded_count,
            protected_processed_count = excluded.protected_processed_count,
            protected_kept_count = excluded.protected_kept_count,
            protected_failed_count = excluded.protected_failed_count,
            batch_count = excluded.batch_count,
            scored_count = excluded.scored_count,
            score_sum = excluded.score_sum,
            avg_score = excluded.avg_score,
            min_score = excluded.min_score,
            max_score = excluded.max_score,
            last_error = excluded.last_error,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            chat_state.chat_id,
            chat_state.title,
            chat_state.chat_type,
            int(chat_state.has_protected_content),
            chat_state.last_read_message_id,
            chat_state.downloaded_count,
            chat_state.processed_count,
            chat_state.kept_count,
            chat_state.deleted_count,
            chat_state.skipped_count,
            chat_state.failed_count,
            chat_state.protected_downloaded_count,
            chat_state.protected_processed_count,
            chat_state.protected_kept_count,
            chat_state.protected_failed_count,
            chat_state.batch_count,
            chat_state.scored_count,
            chat_state.score_sum,
            chat_state.avg_score if chat_state.scored_count > 0 else None,
            chat_state.min_score,
            chat_state.max_score,
            chat_state.last_error,
        ),
    )
    commit_with_retry(conn)


class InferenceStore:
    """Thin wrapper around the shared sqlite cache."""

    def __init__(self, db_path: Path):
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=120.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 120000")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        ensure_db(self.conn)

    def get_completed_by_hash(
        self,
        content_hash: str,
        checkpoint_path: str,
    ) -> sqlite3.Row | None:
        """Return a cached completed score for the same content hash."""
        cursor = self.conn.execute(
            """
            SELECT *
            FROM media_inference
            WHERE content_hash = ?
              AND model_checkpoint = ?
              AND status = 'completed'
              AND score IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (content_hash, checkpoint_path),
        )
        return cursor.fetchone()

    def write_result(
        self,
        media_path: Path,
        media_type: str,
        content_hash: str,
        checkpoint_path: Path,
        score: float | None,
        status: str,
        error: str | None = None,
    ) -> None:
        """Persist one file result."""
        write_result(
            conn=self.conn,
            media_path=str(media_path),
            media_type=media_type,
            content_hash=content_hash,
            checkpoint_path=str(checkpoint_path),
            score=score,
            status=status,
            error=error,
        )

    def close(self) -> None:
        """Close the sqlite connection."""
        self.conn.close()

    def get_completed_count(self, checkpoint_path: Path) -> int:
        """Return how many scored rows exist for one checkpoint."""
        return get_completed_count(self.conn, str(checkpoint_path))

    def replace_root_prefix(self, old_root: Path, new_root: Path) -> int:
        """Rewrite database media paths from one root to another."""
        updated = replace_path_prefix(self.conn, old_root, new_root)
        if updated:
            commit_with_retry(self.conn)
        return updated

    def delete_path(self, media_path: Path) -> None:
        """Delete all cached rows for one file path."""
        delete_result(self.conn, str(media_path))

    def write_chat_stats(self, chat_state: ChatState) -> None:
        """Persist one aggregate Telegram chat state row."""
        upsert_chat_stats(self.conn, chat_state)

    def rename_path(self, old_media_path: Path, new_media_path: Path) -> None:
        """Rename one cached media path in sqlite."""
        rename_result(self.conn, str(old_media_path), str(new_media_path))

    def list_results_under_root(self, root: Path) -> list[sqlite3.Row]:
        """Return cached inference rows whose path lives under one root."""
        resolved_root = root.expanduser().resolve()
        cursor = self.conn.execute(
            """
            SELECT media_path, media_type, content_hash, score, status, error, model_checkpoint, updated_at
            FROM media_inference
            WHERE media_path LIKE ?
            """,
            (f"{resolved_root}/%",),
        )
        return list(cursor.fetchall())

    def get_result_by_path(self, media_path: Path) -> sqlite3.Row | None:
        """Return one cached inference row for an exact media path."""
        cursor = self.conn.execute(
            """
            SELECT *
            FROM media_inference
            WHERE media_path = ?
            LIMIT 1
            """,
            (str(media_path),),
        )
        return cursor.fetchone()

    def has_existing_media_with_hash_under_root(self, content_hash: str, root: Path) -> bool:
        """Check whether a scored media hash already exists under another root."""
        resolved_root = root.expanduser().resolve()
        cursor = self.conn.execute(
            """
            SELECT media_path
            FROM media_inference
            WHERE content_hash = ?
              AND media_path LIKE ?
              AND status = 'completed'
            """,
            (content_hash, f"{resolved_root}/%"),
        )
        for row in cursor.fetchall():
            media_path = Path(row["media_path"])
            if media_path.exists() or media_path.is_symlink():
                return True
        return False

    def find_name_variants(self, parent_dir: Path, canonical_name: str) -> list[Path]:
        """Return cached rows that point at one target basename or its score-prefixed forms."""
        parent_dir = parent_dir.resolve()
        cursor = self.conn.execute(
            """
            SELECT media_path
            FROM media_inference
            WHERE media_path LIKE ?
            """,
            (f"{parent_dir}/%",),
        )

        matches: list[Path] = []
        for row in cursor.fetchall():
            media_path = Path(row["media_path"])
            if media_path.parent != parent_dir:
                continue
            if media_path.name == canonical_name or media_path.name.endswith(
                f"__{canonical_name}"
            ):
                matches.append(media_path)
        return matches

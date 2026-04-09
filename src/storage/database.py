"""SQLite database management with async support."""

from pathlib import Path
from typing import Any, Optional

import aiosqlite

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


class Database:
    """Async SQLite database manager for the published baseline."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Initialize database and create tables."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info("Database initialized at %s", self.db_path)

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _create_tables(self) -> None:
        """Create all tables required by the current baseline."""
        assert self._db is not None
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS media_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL UNIQUE,
                score REAL NOT NULL CHECK(score >= 0 AND score <= 9),
                user_id TEXT DEFAULT 'default',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (media_id) REFERENCES media_files(id)
            );

            CREATE TABLE IF NOT EXISTS media_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                media_path TEXT NOT NULL,
                task_type TEXT NOT NULL CHECK(task_type IN ('thumbnail', 'transcode')),
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
                progress INTEGER DEFAULT 0,
                error TEXT,
                result_path TEXT,
                content_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            );

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

            CREATE INDEX IF NOT EXISTS idx_media_tasks_status ON media_tasks(status);
            CREATE INDEX IF NOT EXISTS idx_media_tasks_media ON media_tasks(media_path);
            CREATE INDEX IF NOT EXISTS idx_media_tasks_hash ON media_tasks(content_hash);
            CREATE INDEX IF NOT EXISTS idx_media_inference_hash ON media_inference(content_hash);
            CREATE INDEX IF NOT EXISTS idx_media_inference_score ON media_inference(score DESC);
            CREATE INDEX IF NOT EXISTS idx_media_inference_status ON media_inference(status);
            CREATE INDEX IF NOT EXISTS idx_media_inference_checkpoint ON media_inference(model_checkpoint);
            """
        )
        await self._db.commit()
        await self._migrate_labels_constraint()
        await self._drop_legacy_prompt_config()

    async def _migrate_labels_constraint(self) -> None:
        """Migrate legacy label score constraints from 1-10 to 0-9."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='labels'"
        )
        row = await cursor.fetchone()
        if row and row["sql"] and "score >= 1 AND score <= 10" in row["sql"]:
            logger.info("Migrating labels table to 0-9 score range")
            await self._db.executescript(
                """
                CREATE TABLE labels_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id INTEGER NOT NULL UNIQUE,
                    score REAL NOT NULL CHECK(score >= 0 AND score <= 9),
                    user_id TEXT DEFAULT 'default',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (media_id) REFERENCES media_files(id)
                );

                INSERT INTO labels_new (id, media_id, score, user_id, created_at)
                SELECT id, media_id, MIN(9, MAX(0, score)), user_id, created_at FROM labels;

                DROP TABLE labels;
                ALTER TABLE labels_new RENAME TO labels;
                """
            )
            await self._db.commit()

    async def _drop_legacy_prompt_config(self) -> None:
        """Drop the old scoring prompt table when it exists."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scoring_config'"
        )
        row = await cursor.fetchone()
        if row:
            logger.info("Dropping legacy scoring_config table")
            await self._db.execute("DROP TABLE scoring_config")
            await self._db.commit()

    async def add_media(self, path: str, file_type: str) -> int:
        """Insert a media file if needed and return the row id when inserted."""
        assert self._db is not None
        async with self._db.execute(
            "INSERT OR IGNORE INTO media_files (path, file_type) VALUES (?, ?)",
            (path, file_type),
        ) as cursor:
            await self._db.commit()
            return cursor.lastrowid

    async def get_media(self, path: str) -> Optional[dict[str, Any]]:
        """Fetch media metadata by path."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM media_files WHERE path = ?",
            (path,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def upsert_media_inference(
        self,
        media_path: str,
        media_type: str,
        content_hash: str,
        score: Optional[float],
        status: str = "completed",
        error: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
    ) -> None:
        """Insert or update a media inference result."""
        assert self._db is not None
        await self._db.execute(
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
                model_checkpoint,
            ),
        )
        await self._db.commit()

    async def get_media_inference(
        self,
        media_path: str,
        model_checkpoint: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get a stored inference result by media path."""
        assert self._db is not None
        query = "SELECT * FROM media_inference WHERE media_path = ?"
        params: list[Any] = [media_path]
        if model_checkpoint is not None:
            query += " AND model_checkpoint = ?"
            params.append(model_checkpoint)

        async with self._db.execute(query, tuple(params)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_media_inference_by_hash(
        self,
        content_hash: str,
        model_checkpoint: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get the latest completed inference result by content hash."""
        assert self._db is not None
        query = """
            SELECT * FROM media_inference
            WHERE content_hash = ? AND status = 'completed' AND score IS NOT NULL
        """
        params: list[Any] = [content_hash]
        if model_checkpoint is not None:
            query += " AND model_checkpoint = ?"
            params.append(model_checkpoint)

        query += " ORDER BY updated_at DESC LIMIT 1"
        async with self._db.execute(query, tuple(params)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_top_media_inference(
        self,
        limit: int = 100,
        root_path: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List top scored completed inference results."""
        assert self._db is not None
        query = """
            SELECT * FROM media_inference
            WHERE status = 'completed' AND score IS NOT NULL
        """
        params: list[Any] = []
        if root_path is not None:
            query += " AND (media_path = ? OR media_path LIKE ?)"
            params.extend([root_path, f"{root_path.rstrip('/')}/%"])
        if model_checkpoint is not None:
            query += " AND model_checkpoint = ?"
            params.append(model_checkpoint)

        query += " ORDER BY score DESC, updated_at DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, tuple(params)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def add_label(
        self,
        media_id: int,
        score: float,
        user_id: str = "default",
    ) -> int:
        """Add or replace a label for a media file."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR REPLACE INTO labels (media_id, score, user_id)
            VALUES (?, ?, ?)
            """,
            (media_id, score, user_id),
        )
        await self._db.commit()
        return media_id

    async def get_labels(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch labels ordered by newest first."""
        assert self._db is not None
        async with self._db.execute(
            """
            SELECT l.*, m.path as media_path
            FROM labels l
            JOIN media_files m ON l.media_id = m.id
            ORDER BY l.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_label_count(self) -> int:
        """Return total label count."""
        assert self._db is not None
        async with self._db.execute("SELECT COUNT(*) FROM labels") as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def delete_label(self, label_id: int) -> bool:
        """Delete a label by ID."""
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM labels WHERE id = ?",
            (label_id,),
        ) as cursor:
            await self._db.commit()
            return cursor.rowcount > 0

    async def get_label(self, label_id: int) -> Optional[dict[str, Any]]:
        """Fetch one label by ID."""
        assert self._db is not None
        async with self._db.execute(
            """
            SELECT l.*, m.path as media_path
            FROM labels l
            JOIN media_files m ON l.media_id = m.id
            WHERE l.id = ?
            """,
            (label_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_all_labels_for_export(self) -> list[dict[str, Any]]:
        """Return all labels in export format."""
        assert self._db is not None
        async with self._db.execute(
            """
            SELECT m.path as media_path, l.score
            FROM labels l
            JOIN media_files m ON l.media_id = m.id
            ORDER BY l.created_at DESC
            """
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {"media_path": row["media_path"], "score": row["score"]}
                for row in rows
            ]

    async def get_labeling_stats(self) -> dict[str, Any]:
        """Return export/labeling statistics."""
        assert self._db is not None
        async with self._db.execute("SELECT COUNT(*) FROM labels") as cursor:
            total_labels_row = await cursor.fetchone()
        async with self._db.execute(
            "SELECT COUNT(DISTINCT media_id) FROM labels"
        ) as cursor:
            labeled_files_row = await cursor.fetchone()

        total_labels = int(total_labels_row[0]) if total_labels_row else 0
        labeled_files = int(labeled_files_row[0]) if labeled_files_row else 0
        return {
            "total": total_labels,
            "total_labels": total_labels,
            "labeled_files": labeled_files,
        }


db = Database(settings.database_path)

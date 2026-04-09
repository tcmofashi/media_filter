#!/usr/bin/env python
"""Run bulk inference for Telegram media and materialize the top-ranked files."""

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.frozen_clip_engine import FrozenClipEngine

DEFAULT_ROOT = PROJECT_ROOT / "data/tg_target"
DEFAULT_DB_PATH = PROJECT_ROOT / "data/media_filter.db"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/checkpoint_best.pt"
GENERATED_ENTRY_PATTERN = re.compile(r"^\d+(?:\.\d+)?_")
SKIPPED_DIR_NAMES = {"score_links"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk score Telegram media and link the top-ranked files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--sort-dir",
        type=Path,
        default=None,
        help="Directory where the top-ranked media links will be written.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite database path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Frozen CLIP checkpoint used for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Inference device, for example "auto", "cuda", or "cpu".',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top-ranked files to materialize into sort-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of files to process.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum score required for sort-dir materialization.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Progress print interval.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize files into sort-dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score files even if a matching cached result already exists.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the file list into N shards and process only one shard.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to process.",
    )
    parser.add_argument(
        "--skip-sort",
        action="store_true",
        help="Do not rebuild the sort directory after inference.",
    )
    parser.add_argument(
        "--refresh-sort-only",
        action="store_true",
        help="Skip inference and rebuild sort-dir from cached DB results only.",
    )
    return parser.parse_args()


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.executescript(MEDIA_TABLE_SQL)
    conn.executescript(INFERENCE_TABLE_SQL)
    conn.commit()


def detect_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in FrozenClipEngine.VIDEO_EXTENSIONS:
        return "video"
    if suffix in FrozenClipEngine.IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"Unsupported media type: {path}")


def iter_media_files(root: Path, sort_dir: Path) -> Iterator[Path]:
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if (current_path / name) != sort_dir and name not in SKIPPED_DIR_NAMES
        )

        for filename in sorted(filenames):
            candidate = current_path / filename
            suffix = candidate.suffix.lower()
            if (
                suffix in FrozenClipEngine.IMAGE_EXTENSIONS
                or suffix in FrozenClipEngine.VIDEO_EXTENSIONS
            ):
                yield candidate


def compute_content_hash(file_path: Path) -> str:
    file_size = file_path.stat().st_size
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        chunk = handle.read(65536)
        if chunk:
            hasher.update(f"{file_size}:".encode("utf-8") + chunk)
    return hasher.hexdigest()


def is_locked_error(exc: sqlite3.OperationalError) -> bool:
    return LOCKED_ERROR in str(exc).lower()


def commit_with_retry(conn: sqlite3.Connection) -> None:
    for attempt in range(WRITE_RETRY_ATTEMPTS):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if not is_locked_error(exc) or attempt == WRITE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(WRITE_RETRY_BASE_DELAY * (attempt + 1))


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


def insert_media_file(conn: sqlite3.Connection, media_path: str, media_type: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO media_files (path, file_type) VALUES (?, ?)",
        (media_path, media_type),
    )


def get_inference_by_path(conn: sqlite3.Connection, media_path: str) -> sqlite3.Row | None:
    cursor = conn.execute(
        "SELECT * FROM media_inference WHERE media_path = ?",
        (media_path,),
    )
    return cursor.fetchone()


def get_inference_by_hash(
    conn: sqlite3.Connection,
    content_hash: str,
    checkpoint_path: str,
) -> sqlite3.Row | None:
    cursor = conn.execute(
        """
        SELECT * FROM media_inference
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


def query_top_k(
    conn: sqlite3.Connection,
    root: Path,
    checkpoint_path: str,
    top_k: int,
    min_score: float | None = None,
) -> list[sqlite3.Row]:
    query = """
        SELECT media_path, media_type, content_hash, score
        FROM media_inference
        WHERE status = 'completed'
          AND score IS NOT NULL
          AND model_checkpoint = ?
          AND (media_path = ? OR media_path LIKE ?)
    """
    params: list[object] = [
        checkpoint_path,
        str(root),
        f"{str(root).rstrip('/')}/%",
    ]

    if min_score is not None:
        query += " AND score >= ?"
        params.append(min_score)

    query += """
        ORDER BY score DESC, updated_at DESC
        LIMIT ?
    """
    params.append(top_k)

    cursor = conn.execute(query, tuple(params))
    return list(cursor.fetchall())


def cleanup_sort_dir(sort_dir: Path) -> None:
    sort_dir.mkdir(parents=True, exist_ok=True)
    for child in sort_dir.iterdir():
        if child.name == "top100.jsonl" or GENERATED_ENTRY_PATTERN.match(child.name):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"\s+", "_", name.strip())
    cleaned = re.sub(r"[\\/:\0]", "_", cleaned)
    return cleaned or "media"


def strip_score_prefix(name: str) -> str:
    return GENERATED_ENTRY_PATTERN.sub("", name, count=1)


def materialize_media(source: Path, target: Path, link_mode: str) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()

    if link_mode == "symlink":
        target.symlink_to(source)
        return

    if link_mode == "hardlink":
        os.link(source, target)
        return

    shutil.copy2(source, target)


def build_sort_target_name(
    score: float,
    basename: str,
    suffix: str,
    used_names: set[str],
) -> str:
    score_prefix = f"{score:.4f}"
    candidate = f"{score_prefix}_{basename}{suffix}"
    index = 2
    while candidate in used_names:
        candidate = f"{score_prefix}_{basename}_{index}{suffix}"
        index += 1
    used_names.add(candidate)
    return candidate


def write_top_k(
    conn: sqlite3.Connection,
    root: Path,
    sort_dir: Path,
    checkpoint_path: str,
    top_k: int,
    link_mode: str,
    min_score: float | None = None,
) -> list[dict[str, object]]:
    rows = query_top_k(
        conn,
        root,
        checkpoint_path,
        top_k,
        min_score=min_score,
    )
    cleanup_sort_dir(sort_dir)

    manifest: list[dict[str, object]] = []
    used_names: set[str] = set()
    for rank, row in enumerate(rows, start=1):
        source = Path(row["media_path"])
        if not source.exists():
            continue

        suffix = source.suffix.lower()
        basename = sanitize_name(strip_score_prefix(source.stem))[:96]
        target_name = build_sort_target_name(
            score=float(row["score"]),
            basename=basename,
            suffix=suffix,
            used_names=used_names,
        )
        target = sort_dir / target_name
        materialize_media(source, target, link_mode)

        manifest.append(
            {
                "rank": rank,
                "score": float(row["score"]),
                "media_path": str(source),
                "media_type": row["media_type"],
                "content_hash": row["content_hash"],
                "link_path": str(target),
            }
        )

    manifest_path = sort_dir / "top100.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return manifest


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    sort_dir = args.sort_dir.resolve() if args.sort_dir else (root / "sort")
    db_path = args.db.resolve()
    checkpoint_path = args.checkpoint.resolve()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    all_files = list(iter_media_files(root, sort_dir))
    files = [
        path
        for index, path in enumerate(all_files)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        files = files[: args.limit]

    print(f"root={root}")
    print(f"db={db_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"sort_dir={sort_dir}")
    print(f"shard={args.shard_index + 1}/{args.num_shards}")
    print(f"all_files={len(all_files)}")
    print(f"files={len(files)}")
    print(f"link_mode={args.link_mode}")
    print(f"min_score={args.min_score}")

    conn = sqlite3.connect(db_path, timeout=120.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 120000")
    conn.execute("PRAGMA synchronous = NORMAL")
    ensure_db(conn)

    if args.refresh_sort_only:
        manifest = write_top_k(
            conn=conn,
            root=root,
            sort_dir=sort_dir,
            checkpoint_path=str(checkpoint_path),
            top_k=args.top_k,
            link_mode=args.link_mode,
            min_score=args.min_score,
        )
        conn.close()
        print(
            "[done] "
            f"refresh_sort_only=1 "
            f"top_k_written={len(manifest)}"
        )
        return 0

    engine = FrozenClipEngine(device=args.device)
    engine.load_model(checkpoint_path)
    print(f"model_info={json.dumps(engine.get_model_info(), ensure_ascii=False)}")

    stats = {
        "scored": 0,
        "reused_path": 0,
        "reused_hash": 0,
        "failed": 0,
    }
    checkpoint_str = str(checkpoint_path)
    started_at = time.time()

    for index, media_path in enumerate(files, start=1):
        media_type = detect_media_type(media_path)
        media_path_str = str(media_path)

        content_hash = compute_content_hash(media_path)
        cached_row = get_inference_by_path(conn, media_path_str)

        if (
            not args.force
            and cached_row is not None
            and cached_row["content_hash"] == content_hash
            and cached_row["model_checkpoint"] == checkpoint_str
            and cached_row["status"] == "completed"
            and cached_row["score"] is not None
        ):
            stats["reused_path"] += 1
        else:
            reused_row = None
            if not args.force:
                reused_row = get_inference_by_hash(conn, content_hash, checkpoint_str)

            if reused_row is not None:
                write_result(
                    conn=conn,
                    media_path=media_path_str,
                    media_type=media_type,
                    content_hash=content_hash,
                    checkpoint_path=checkpoint_str,
                    score=float(reused_row["score"]),
                    status="completed",
                    error=None,
                )
                stats["reused_hash"] += 1
            else:
                try:
                    result = engine.score(
                        media_path=media_path,
                        is_video=(media_type == "video"),
                    )
                    write_result(
                        conn=conn,
                        media_path=media_path_str,
                        media_type=media_type,
                        content_hash=content_hash,
                        checkpoint_path=checkpoint_str,
                        score=float(result["score"]),
                        status="completed",
                        error=None,
                    )
                    stats["scored"] += 1
                except Exception as exc:
                    write_result(
                        conn=conn,
                        media_path=media_path_str,
                        media_type=media_type,
                        content_hash=content_hash,
                        checkpoint_path=checkpoint_str,
                        score=None,
                        status="failed",
                        error=str(exc)[:2000],
                    )
                    stats["failed"] += 1

        if index % args.progress_every == 0 or index == len(files):
            elapsed = time.time() - started_at
            print(
                "[progress] "
                f"{index}/{len(files)} "
                f"scored={stats['scored']} "
                f"reused_path={stats['reused_path']} "
                f"reused_hash={stats['reused_hash']} "
                f"failed={stats['failed']} "
                f"elapsed={format_duration(elapsed)}"
            )

    manifest: list[dict[str, object]] = []
    if not args.skip_sort:
        manifest = write_top_k(
            conn=conn,
            root=root,
            sort_dir=sort_dir,
            checkpoint_path=checkpoint_str,
            top_k=args.top_k,
            link_mode=args.link_mode,
            min_score=args.min_score,
        )
    conn.commit()
    conn.close()

    elapsed = time.time() - started_at
    print(
        "[done] "
        f"processed={len(files)} "
        f"scored={stats['scored']} "
        f"reused_path={stats['reused_path']} "
        f"reused_hash={stats['reused_hash']} "
        f"failed={stats['failed']} "
        f"top_k_written={len(manifest)} "
        f"elapsed={format_duration(elapsed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

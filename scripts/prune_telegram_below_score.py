#!/usr/bin/env python
"""Delete scored Telegram media whose score falls below a threshold."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data/tg_target"
DEFAULT_DB_PATH = PROJECT_ROOT / "data/media_filter.db"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/checkpoint_best.pt"
SKIP_DIR_NAMES = {"score_links", "sort"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete Telegram media already scored by media_filter when the score is "
            "below a threshold."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Telegram root directory.",
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
        help="Model checkpoint whose cached scores should be used.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        required=True,
        help="Delete completed media whose predicted score is below this value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many low-score rows to prune.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletions without touching files or the database.",
    )
    return parser.parse_args()


def fetch_prune_rows(
    conn: sqlite3.Connection,
    root: Path,
    checkpoint: str,
    min_score: float,
    limit: int | None,
) -> list[sqlite3.Row]:
    query = """
        SELECT media_path, media_type, content_hash, score
        FROM media_inference
        WHERE status = 'completed'
          AND score IS NOT NULL
          AND score < ?
          AND model_checkpoint = ?
          AND (media_path = ? OR media_path LIKE ?)
        ORDER BY score ASC, updated_at ASC, media_path ASC
    """
    params: list[object] = [
        min_score,
        checkpoint,
        str(root),
        f"{str(root).rstrip('/')}/%",
    ]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return list(conn.execute(query, tuple(params)).fetchall())


def is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def delete_db_rows(conn: sqlite3.Connection, media_path: str) -> None:
    media_tasks_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'media_tasks'"
    ).fetchone()
    conn.execute("DELETE FROM media_inference WHERE media_path = ?", (media_path,))
    conn.execute("DELETE FROM media_files WHERE path = ?", (media_path,))
    if media_tasks_exists:
        conn.execute("DELETE FROM media_tasks WHERE media_path = ?", (media_path,))


def prune_empty_dirs(root: Path) -> int:
    removed = 0
    for current_root, dirnames, _ in os.walk(root, topdown=False):
        current_path = Path(current_root)
        if current_path == root:
            continue
        if current_path.name in SKIP_DIR_NAMES:
            continue
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        try:
            current_path.rmdir()
        except OSError:
            continue
        removed += 1
    return removed


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    db_path = args.db.resolve()
    checkpoint = args.checkpoint.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = fetch_prune_rows(
        conn=conn,
        root=root,
        checkpoint=str(checkpoint),
        min_score=args.min_score,
        limit=args.limit,
    )
    print(f"root={root}")
    print(f"db={db_path}")
    print(f"checkpoint={checkpoint}")
    print(f"min_score={args.min_score}")
    print(f"candidate_rows={len(rows)}")
    print(f"dry_run={int(args.dry_run)}")

    pruned = 0
    missing = 0
    skipped = 0
    for row in rows:
        source = Path(row["media_path"])
        if not is_under_root(source, root):
            skipped += 1
            print(f"[skip-outside-root] {source}")
            continue
        if source.name == ".DS_Store" or source.parent.name in SKIP_DIR_NAMES:
            skipped += 1
            print(f"[skip-generated] {source}")
            continue

        if args.dry_run:
            print(f"[dry-run] score={float(row['score']):.4f} {source}")
            pruned += 1
            continue

        if source.exists():
            source.unlink()
            pruned += 1
            print(f"[deleted] score={float(row['score']):.4f} {source}")
        else:
            missing += 1
            print(f"[missing] score={float(row['score']):.4f} {source}")

        delete_db_rows(conn, str(source))

    removed_dirs = 0
    if not args.dry_run:
        conn.commit()
        removed_dirs = prune_empty_dirs(root)
    conn.close()

    print(
        "[done] "
        f"pruned={pruned} "
        f"missing={missing} "
        f"skipped={skipped} "
        f"removed_dirs={removed_dirs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

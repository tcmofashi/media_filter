#!/usr/bin/env python
"""Redistribute scored Telegram media into score buckets and rename files."""

import argparse
import os
import shutil
import sqlite3
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOT = PROJECT_ROOT / "data/tg_target"
DEFAULT_DB_PATH = PROJECT_ROOT / "data/xpfilter.db"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/checkpoint_best.pt"
SCORE_PREFIX_PATTERN = re.compile(r"^\d+(?:\.\d+)?_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redistribute scored Telegram media into score buckets."
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
        "--limit",
        type=int,
        default=None,
        help="Optional number of completed rows to process.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum score required for bucket materialization.",
    )
    parser.add_argument(
        "--mode",
        choices=["move", "symlink", "hardlink", "copy"],
        default="move",
        help="How to materialize files into buckets.",
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Output directory for bucketed files. Defaults to root for move and root/score_links for other modes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without changing files or database.",
    )
    parser.add_argument(
        "--skip-sort-refresh",
        action="store_true",
        help="Do not refresh /sort after moves complete.",
    )
    return parser.parse_args()


def bucket_for_score(score: float) -> int:
    if score < 0:
        return 0
    if score >= 9:
        return 9
    return int(score)


def strip_score_prefix(name: str) -> str:
    return SCORE_PREFIX_PATTERN.sub("", name, count=1)


def build_target_path(
    bucket_root: Path,
    score: float,
    source: Path,
    reserved_targets: set[Path],
) -> Path:
    bucket_dir = bucket_root / str(bucket_for_score(score))

    clean_name = strip_score_prefix(source.stem)
    prefix = f"{score:.4f}"
    candidate = bucket_dir / f"{prefix}_{clean_name}{source.suffix}"

    if candidate == source:
        reserved_targets.add(candidate)
        return candidate

    index = 2
    while candidate in reserved_targets or (candidate.exists() and candidate != source):
        candidate = bucket_dir / f"{prefix}_{clean_name}_{index}{source.suffix}"
        index += 1

    reserved_targets.add(candidate)
    return candidate


def resolve_bucket_root(root: Path, bucket_root: Path | None, mode: str) -> Path:
    if bucket_root is not None:
        return bucket_root.resolve()
    if mode == "move":
        return root
    return (root / "score_links").resolve()


def reset_generated_buckets(bucket_root: Path, dry_run: bool) -> None:
    for bucket in range(10):
        bucket_dir = bucket_root / str(bucket)
        if not bucket_dir.exists():
            continue
        print(f"[cleanup] {bucket_dir}")
        if dry_run:
            continue
        shutil.rmtree(bucket_dir)


def materialize_output(source: Path, target: Path, mode: str) -> None:
    if mode == "move":
        source.rename(target)
        return
    if mode == "symlink":
        os.symlink(source, target)
        return
    if mode == "hardlink":
        os.link(source, target)
        return
    if mode == "copy":
        shutil.copy2(source, target)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def fetch_completed_rows(
    conn: sqlite3.Connection,
    root: Path,
    checkpoint: str,
    limit: int | None,
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
    params: list[object] = [checkpoint, str(root), f"{str(root).rstrip('/')}/%"]
    if min_score is not None:
        query += " AND score >= ?"
        params.append(min_score)
    query += " ORDER BY score DESC, updated_at DESC, media_path ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return list(conn.execute(query, tuple(params)).fetchall())


def update_paths(
    conn: sqlite3.Connection,
    old_path: str,
    new_path: str,
) -> None:
    media_tasks_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'media_tasks'"
    ).fetchone()
    conn.execute(
        "UPDATE media_inference SET media_path = ?, updated_at = CURRENT_TIMESTAMP WHERE media_path = ?",
        (new_path, old_path),
    )
    conn.execute(
        "UPDATE media_files SET path = ? WHERE path = ?",
        (new_path, old_path),
    )
    if media_tasks_exists:
        conn.execute(
            "UPDATE media_tasks SET media_path = ? WHERE media_path = ?",
            (new_path, old_path),
        )


def refresh_sort(project_root: Path) -> int:
    import subprocess
    import sys

    command = [sys.executable, "scripts/bulk_infer_telegram.py", "--refresh-sort-only"]
    result = subprocess.run(command, cwd=project_root, check=False)
    return result.returncode


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    db_path = args.db.resolve()
    checkpoint = args.checkpoint.resolve()
    bucket_root = resolve_bucket_root(root, args.bucket_root, args.mode)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.mode != "move" and bucket_root == root:
        raise ValueError("Non-move mode requires bucket_root to be outside root.")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = fetch_completed_rows(
        conn=conn,
        root=root,
        checkpoint=str(checkpoint),
        limit=args.limit,
        min_score=args.min_score,
    )
    print(f"root={root}")
    print(f"bucket_root={bucket_root}")
    print(f"db={db_path}")
    print(f"checkpoint={checkpoint}")
    print(f"completed_rows={len(rows)}")
    print(f"mode={args.mode}")
    print(f"dry_run={int(args.dry_run)}")
    print(f"min_score={args.min_score}")

    if args.mode != "move":
        reset_generated_buckets(bucket_root, args.dry_run)

    reserved_targets: set[Path] = set()
    updates: list[tuple[str, str]] = []
    materialized = 0
    skipped = 0
    missing = 0

    for row in rows:
        source = Path(row["media_path"])
        if not source.exists():
            missing += 1
            print(f"[missing] {source}")
            continue

        target = build_target_path(
            bucket_root=bucket_root,
            score=float(row["score"]),
            source=source,
            reserved_targets=reserved_targets,
        )

        if args.mode == "move" and target == source:
            skipped += 1
            continue

        updates.append((str(source), str(target)))
        materialized += 1
        if args.dry_run:
            print(f"[dry-run] {source} -> {target}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        materialize_output(source, target, args.mode)
        if args.mode == "move":
            update_paths(conn, str(source), str(target))

    if not args.dry_run:
        conn.commit()
    conn.close()

    print(f"[done] materialized={materialized} skipped={skipped} missing={missing}")

    if args.mode == "move" and not args.dry_run and not args.skip_sort_refresh:
        refresh_code = refresh_sort(PROJECT_ROOT)
        print(f"[sort_refresh] exit_code={refresh_code}")
        if refresh_code != 0:
            return refresh_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
数据导入脚本 - 将 JSON 数据导入 SQLite 数据库

Usage:
    python scripts/import_to_db.py --train data/train.json --val data/val.json --db data/mediaflusher.db
"""

import json
import argparse
import asyncio
import sqlite3
from pathlib import Path
from typing import List, Dict

# 视频扩展名
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 格式的文件"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def get_file_type(media_path: str) -> str:
    """判断文件类型"""
    ext = Path(media_path).suffix.lower()
    return "video" if ext in VIDEO_EXTENSIONS else "image"


def import_data(train_path: str, val_path: str, db_path: str) -> Dict:
    """导入数据到 SQLite 数据库"""

    # 连接数据库
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 创建表（如果不存在）
    cursor.executescript("""
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
            split TEXT DEFAULT 'train',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media_files(id)
        );
        
        CREATE TABLE IF NOT EXISTS dataset_split (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('train', 'val')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media_files(id),
            UNIQUE(media_id)
        );
    """)
    conn.commit()

    # 加载数据
    train_samples = load_jsonl(train_path)
    val_samples = load_jsonl(val_path)

    print(f"训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")

    stats = {
        "train": {"total": 0, "images": 0, "videos": 0},
        "val": {"total": 0, "images": 0, "videos": 0},
    }

    # 导入训练集
    for sample in train_samples:
        media_path = sample["media_path"]
        score = sample["score"]
        file_type = get_file_type(media_path)

        # 插入媒体文件
        cursor.execute(
            "INSERT OR IGNORE INTO media_files (path, file_type) VALUES (?, ?)",
            (media_path, file_type),
        )

        # 获取 media_id
        cursor.execute("SELECT id FROM media_files WHERE path = ?", (media_path,))
        row = cursor.fetchone()
        media_id = row["id"]

        # 插入标签
        cursor.execute(
            """INSERT OR REPLACE INTO labels (media_id, score, split) VALUES (?, ?, 'train')""",
            (media_id, score),
        )

        # 记录数据集划分
        cursor.execute(
            """INSERT OR REPLACE INTO dataset_split (media_id, split) VALUES (?, 'train')""",
            (media_id,),
        )

        stats["train"]["total"] += 1
        if file_type == "video":
            stats["train"]["videos"] += 1
        else:
            stats["train"]["images"] += 1

    # 导入验证集
    for sample in val_samples:
        media_path = sample["media_path"]
        score = sample["score"]
        file_type = get_file_type(media_path)

        # 插入媒体文件
        cursor.execute(
            "INSERT OR IGNORE INTO media_files (path, file_type) VALUES (?, ?)",
            (media_path, file_type),
        )

        # 获取 media_id
        cursor.execute("SELECT id FROM media_files WHERE path = ?", (media_path,))
        row = cursor.fetchone()
        media_id = row["id"]

        # 插入标签
        cursor.execute(
            """INSERT OR REPLACE INTO labels (media_id, score, split) VALUES (?, ?, 'val')""",
            (media_id, score),
        )

        # 记录数据集划分
        cursor.execute(
            """INSERT OR REPLACE INTO dataset_split (media_id, split) VALUES (?, 'val')""",
            (media_id,),
        )

        stats["val"]["total"] += 1
        if file_type == "video":
            stats["val"]["videos"] += 1
        else:
            stats["val"]["images"] += 1

    conn.commit()

    # 验证导入结果
    cursor.execute("SELECT COUNT(*) FROM media_files")
    media_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM labels")
    label_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM labels WHERE split = 'train'")
    train_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM labels WHERE split = 'val'")
    val_count = cursor.fetchone()[0]

    conn.close()

    print(f"\n导入完成!")
    print(f"  - 媒体文件总数: {media_count}")
    print(f"  - 标签总数: {label_count}")
    print(f"  - 训练集标签: {train_count}")
    print(f"  - 验证集标签: {val_count}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="导入 JSON 数据到 SQLite 数据库")
    parser.add_argument(
        "--train",
        "-t",
        type=str,
        default="data/train.json",
        help="训练集 JSONL 文件路径",
    )
    parser.add_argument(
        "--val", "-v", type=str, default="data/val.json", help="验证集 JSONL 文件路径"
    )
    parser.add_argument(
        "--db", "-d", type=str, default="data/mediaflusher.db", help="数据库文件路径"
    )
    args = parser.parse_args()

    print(f"导入数据到数据库: {args.db}")
    print(f"训练集: {args.train}")
    print(f"验证集: {args.val}")

    stats = import_data(args.train, args.val, args.db)

    print(
        f"\n训练集统计: {stats['train']['total']} 样本, {stats['train']['images']} 图片, {stats['train']['videos']} 视频"
    )
    print(
        f"验证集统计: {stats['val']['total']} 样本, {stats['val']['images']} 图片, {stats['val']['videos']} 视频"
    )


if __name__ == "__main__":
    main()

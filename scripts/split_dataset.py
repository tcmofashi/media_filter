#!/usr/bin/env python
"""
数据集拆分脚本 - 将 labels.json 拆分为训练集和验证集

Usage:
    python scripts/split_dataset.py --input labels.json --output-dir data --val-ratio 0.2 --seed 42
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 格式的文件"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: List[Dict], file_path: str) -> None:
    """保存为 JSONL 格式"""
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def split_dataset(samples: List[Dict], val_ratio: float = 0.2, seed: int = 42) -> tuple:
    """拆分数据集为训练集和验证集"""
    random.seed(seed)

    # 打乱数据
    indices = list(range(len(samples)))
    random.shuffle(indices)

    # 计算验证集大小
    val_size = int(len(indices) * val_ratio)

    # 拆分
    val_indices = set(indices[:val_size])
    train_samples = [samples[i] for i in range(len(samples)) if i not in val_indices]
    val_samples = [samples[i] for i in range(len(samples)) if i in val_indices]

    return train_samples, val_samples


def analyze_samples(samples: List[Dict]) -> Dict:
    """分析样本统计信息"""
    scores = [s["score"] for s in samples]

    # 统计分数分布
    score_dist = {}
    for score in scores:
        score_int = int(score)
        score_dist[score_int] = score_dist.get(score_int, 0) + 1

    # 统计媒体类型
    video_count = 0
    image_count = 0
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}

    for sample in samples:
        path = sample["media_path"].lower()
        ext = Path(path).suffix
        if ext in video_extensions:
            video_count += 1
        else:
            image_count += 1

    return {
        "total": len(samples),
        "min_score": min(scores),
        "max_score": max(scores),
        "avg_score": sum(scores) / len(scores),
        "score_distribution": dict(sorted(score_dist.items())),
        "video_count": video_count,
        "image_count": image_count,
    }


def main():
    parser = argparse.ArgumentParser(description="拆分数据集为训练集和验证集")
    parser.add_argument(
        "--input", "-i", type=str, default="labels.json", help="输入 JSONL 文件路径"
    )
    parser.add_argument("--output-dir", "-o", type=str, default="data", help="输出目录")
    parser.add_argument(
        "--val-ratio", "-v", type=float, default=0.2, help="验证集比例 (默认 0.2)"
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="随机种子 (默认 42)")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print(f"加载数据: {args.input}")
    samples = load_jsonl(args.input)
    print(f"总样本数: {len(samples)}")

    # 分析原始数据
    print("\n原始数据分析:")
    stats = analyze_samples(samples)
    print(f"  - 分数范围: {stats['min_score']} - {stats['max_score']}")
    print(f"  - 平均分数: {stats['avg_score']:.2f}")
    print(f"  - 视频数量: {stats['video_count']}")
    print(f"  - 图片数量: {stats['image_count']}")
    print(f"  - 分数分布: {stats['score_distribution']}")

    # 拆分数据集
    print(f"\n拆分数据集 (验证集比例: {args.val_ratio})...")
    train_samples, val_samples = split_dataset(samples, args.val_ratio, args.seed)

    # 保存
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)

    print(f"\n训练集: {len(train_samples)} 样本 -> {train_path}")
    print(f"验证集: {len(val_samples)} 样本 -> {val_path}")

    # 分析拆分后的数据
    print("\n训练集分析:")
    train_stats = analyze_samples(train_samples)
    print(f"  - 分数范围: {train_stats['min_score']} - {train_stats['max_score']}")
    print(f"  - 平均分数: {train_stats['avg_score']:.2f}")
    print(f"  - 视频数量: {train_stats['video_count']}")
    print(f"  - 图片数量: {train_stats['image_count']}")

    print("\n验证集分析:")
    val_stats = analyze_samples(val_samples)
    print(f"  - 分数范围: {val_stats['min_score']} - {val_stats['max_score']}")
    print(f"  - 平均分数: {val_stats['avg_score']:.2f}")
    print(f"  - 视频数量: {val_stats['video_count']}")
    print(f"  - 图片数量: {val_stats['image_count']}")

    print("\n拆分完成!")


if __name__ == "__main__":
    main()

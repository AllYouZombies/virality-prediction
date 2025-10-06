#!/usr/bin/env python3
"""
Скрипт для подготовки TikTok-10M датасета к fine-tuning VideoMAE.

Создаёт binary classification датасет: viral vs non-viral видео
на основе engagement метрик.
"""

import os
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import json


def calculate_virality_score(row):
    """
    Вычисляет virality score на основе engagement метрик.

    Formula:
        engagement_rate = (likes + shares*3 + comments*2) / views
        viral if: engagement_rate > 5% OR views > 1M

    Returns:
        dict: {"is_viral": bool, "engagement_rate": float, "virality_score": int}
    """
    views = row.get('play_count', 1)  # avoid division by zero
    likes = row.get('digg_count', 0)
    shares = row.get('share_count', 0)
    comments = row.get('comment_count', 0)

    # Weighted engagement calculation
    # Shares count 3x (most valuable signal)
    # Comments count 2x (high engagement indicator)
    # Likes count 1x (baseline engagement)
    weighted_engagement = likes + (shares * 3) + (comments * 2)
    engagement_rate = weighted_engagement / max(views, 1)

    # Virality criteria:
    # 1. High engagement rate (>5% is exceptional for TikTok)
    # 2. OR high absolute views (>1M = viral by definition)
    is_viral = (engagement_rate > 0.05) or (views > 1_000_000)

    # Score 0-100 based on combination of metrics
    # 50% from engagement rate, 50% from view count
    engagement_score = min(100, engagement_rate * 1000)  # 5% = 50 points
    views_score = min(100, (views / 2_000_000) * 100)    # 2M views = 50 points
    virality_score = int((engagement_score + views_score) / 2)

    return {
        "is_viral": is_viral,
        "engagement_rate": engagement_rate,
        "virality_score": virality_score,
        "metrics": {
            "views": views,
            "likes": likes,
            "shares": shares,
            "comments": comments
        }
    }


def filter_valid_videos(dataset, min_views=1000, max_duration=180):
    """
    Фильтрует видео по критериям качества.

    Args:
        dataset: HuggingFace dataset
        min_views: Минимум просмотров (фильтруем спам/новые видео)
        max_duration: Максимальная длительность в секундах (фокус на shorts)

    Returns:
        Filtered dataset
    """
    def is_valid(example):
        views = example.get('play_count', 0)
        duration = example.get('duration', 0)

        # Фильтры:
        # 1. Минимум просмотров (убирает спам и новые видео без статистики)
        # 2. Максимум длительность (фокус на short-form контент)
        # 3. Есть описание (quality signal)
        has_description = bool(example.get('desc', '').strip())

        return (
            views >= min_views and
            0 < duration <= max_duration and
            has_description
        )

    return dataset.filter(is_valid)


def prepare_dataset_for_training(
    output_dir: str = "./data/tiktok_viral",
    num_samples: int = 100_000,
    viral_ratio: float = 0.3,
    test_split: float = 0.1,
    val_split: float = 0.1
):
    """
    Подготавливает balanced датасет для training VideoMAE.

    Args:
        output_dir: Директория для сохранения
        num_samples: Количество видео для загрузки
        viral_ratio: Процент viral видео в датасете (0.3 = 30%)
        test_split: Размер test set (0.1 = 10%)
        val_split: Размер validation set (0.1 = 10%)
    """

    print(f"🚀 Loading TikTok-10M dataset (first {num_samples:,} samples)...")

    # Загрузка датасета
    dataset = load_dataset(
        "The-data-company/TikTok-10M",
        split=f"train[:{num_samples}]",
        streaming=False  # Load into memory for faster processing
    )

    print(f"✅ Loaded {len(dataset):,} videos")

    # Фильтрация по качеству
    print("🔍 Filtering valid videos (min 1K views, max 180s duration)...")
    dataset = filter_valid_videos(dataset, min_views=1000, max_duration=180)
    print(f"✅ Filtered to {len(dataset):,} valid videos")

    # Вычисление virality scores
    print("📊 Calculating virality scores...")
    viral_videos = []
    non_viral_videos = []

    for idx, example in enumerate(dataset):
        if idx % 10000 == 0:
            print(f"  Processed {idx:,} / {len(dataset):,} videos...")

        virality_data = calculate_virality_score(example)

        video_entry = {
            "video_id": example.get('id', f'video_{idx}'),
            "duration": example.get('duration', 0),
            "description": example.get('desc', ''),
            "hashtags": example.get('challenges', []),
            "create_time": example.get('create_time', ''),
            **virality_data
        }

        if virality_data['is_viral']:
            viral_videos.append(video_entry)
        else:
            non_viral_videos.append(video_entry)

    print(f"✅ Found {len(viral_videos):,} viral videos")
    print(f"✅ Found {len(non_viral_videos):,} non-viral videos")

    # Балансировка датасета
    target_viral = int(num_samples * viral_ratio)
    target_non_viral = num_samples - target_viral

    print(f"\n⚖️  Balancing dataset to {viral_ratio*100:.0f}% viral...")
    print(f"  Target: {target_viral:,} viral + {target_non_viral:,} non-viral")

    # Сэмплируем нужное количество
    import random
    random.shuffle(viral_videos)
    random.shuffle(non_viral_videos)

    viral_sample = viral_videos[:target_viral]
    non_viral_sample = non_viral_videos[:target_non_viral]

    # Объединяем и перемешиваем
    all_videos = viral_sample + non_viral_sample
    random.shuffle(all_videos)

    # Split на train/val/test
    total = len(all_videos)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - test_size - val_size

    train_data = all_videos[:train_size]
    val_data = all_videos[train_size:train_size + val_size]
    test_data = all_videos[train_size + val_size:]

    print(f"\n📦 Dataset splits:")
    print(f"  Train: {len(train_data):,} videos")
    print(f"  Val:   {len(val_data):,} videos")
    print(f"  Test:  {len(test_data):,} videos")

    # Сохранение
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving to {output_dir}...")

    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        df = pd.DataFrame(split_data)

        # CSV для удобства просмотра
        csv_path = output_path / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)

        # JSON для загрузки в PyTorch/HuggingFace
        json_path = output_path / f"{split_name}.json"
        with open(json_path, 'w') as f:
            json.dump(split_data, f, indent=2)

        print(f"  ✅ {split_name}: {csv_path}")

    # Статистика
    stats = {
        "total_videos": total,
        "viral_count": sum(1 for v in all_videos if v['is_viral']),
        "non_viral_count": sum(1 for v in all_videos if not v['is_viral']),
        "viral_ratio": sum(1 for v in all_videos if v['is_viral']) / total,
        "avg_virality_score": sum(v['virality_score'] for v in all_videos) / total,
        "avg_engagement_rate": sum(v['engagement_rate'] for v in all_videos) / total,
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        }
    }

    stats_path = output_path / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n📊 Dataset statistics:")
    print(f"  Viral ratio: {stats['viral_ratio']*100:.1f}%")
    print(f"  Avg virality score: {stats['avg_virality_score']:.1f}")
    print(f"  Avg engagement rate: {stats['avg_engagement_rate']*100:.2f}%")
    print(f"  Stats saved to: {stats_path}")

    print("\n✨ Dataset preparation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Download videos using video IDs from the CSVs")
    print(f"  2. Run fine-tuning script with prepared labels")
    print(f"  3. Evaluate on test set")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare TikTok-10M dataset for viral video classification")
    parser.add_argument("--output-dir", type=str, default="./data/tiktok_viral", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100_000, help="Number of samples to process")
    parser.add_argument("--viral-ratio", type=float, default=0.3, help="Ratio of viral videos (0-1)")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set ratio")

    args = parser.parse_args()

    prepare_dataset_for_training(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        viral_ratio=args.viral_ratio,
        test_split=args.test_split,
        val_split=args.val_split
    )

"""分桶统计：按 (厚度, 品类, 质量模式) 统计样本量与真值样本量。

用途：后期飞轮的"可训门"判据——各在产桶 is_ground_truth 样本是否攒够。
"""

from __future__ import annotations

from collections import Counter

from schemas.archive import ArchiveSample

Bucket = tuple[float, str, str]


def count_ground_truth_by_bucket(samples: list[ArchiveSample]) -> dict[Bucket, int]:
    """按桶统计 is_ground_truth=True 的样本量。"""
    counter: Counter[Bucket] = Counter()
    for s in samples:
        if s.is_ground_truth:
            counter[s.bucket] += 1
    return dict(counter)


def bucket_table(samples: list[ArchiveSample]) -> list[tuple[Bucket, int, int]]:
    """返回 [(bucket, total, ground_truth), ...]（按 bucket 排序），便于打印看板。"""
    totals: Counter[Bucket] = Counter()
    gts: Counter[Bucket] = Counter()
    for s in samples:
        totals[s.bucket] += 1
        if s.is_ground_truth:
            gts[s.bucket] += 1
    return [(b, totals[b], gts[b]) for b in sorted(totals)]

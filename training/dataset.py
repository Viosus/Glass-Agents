"""数据集 + 切分。

切分铁律：**按时间/批次/桶顺序切，禁止随机打乱**——同一炉/同一天样本不得横跨 train/test，
面向未来测试，诚实反映概念漂移（项目笔记：concept drift）。
占位：真实特征工程（ArchiveSample → 特征/标签）待 N3 定稿。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """从 (features, 多头 targets) 张量构造数据集。"""

    def __init__(self, features: torch.Tensor, targets: dict[str, torch.Tensor]) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, i: int):
        return self.features[i], {k: v[i] for k, v in self.targets.items()}


def time_ordered_split(
    n: int, val_frac: float = 0.2, test_frac: float = 0.2
) -> tuple[list[int], list[int], list[int]]:
    """按既有（时间）顺序切分：train 在前、val 居中、test 在尾。**不打乱**，防未来泄漏。

    要求样本已按时间/批次排序。返回 (train_idx, val_idx, test_idx)，三者不相交且首尾有序。
    """
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("切分比例过大：train 为空")
    idx = list(range(n))  # 不打乱（时间序）
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

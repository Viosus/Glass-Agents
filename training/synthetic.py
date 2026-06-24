"""合成/占位训练数据：随机特征 + 各头随机标签。

仅用于跑通管线，**无任何真实含义**。真数据由 schemas.ArchiveSample 经特征工程产出（待 N3）。
"""

from __future__ import annotations

import torch


def make_synthetic(
    n: int = 64, in_dim: int = 12, param_dim: int = 6, attr_dim: int = 4, seed: int = 0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(n, in_dim, generator=g)
    targets = {
        "param": torch.randn(n, param_dim, generator=g),
        "quality": torch.randn(n, 1, generator=g),
        "energy": torch.randn(n, 1, generator=g),
        "attribution": torch.randn(n, attr_dim, generator=g),
    }
    return features, targets

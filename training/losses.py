"""多头损失 + 不确定性加权（Kendall et al.）平衡各头、防负迁移。

每头一个可学习 log 方差 s：total = Σ ( exp(-s)·L_head + s )。
副头（质量/能耗/归因）初始给较大不确定性 → 较小权重起步，避免主任务（参数头）被淹没。
"""

from __future__ import annotations

import torch
from torch import nn

_INIT_LOG_VAR = {"param": 0.0, "quality": 1.0, "energy": 1.0, "attribution": 1.0}


class MultiHeadLoss(nn.Module):
    def __init__(self, heads: tuple[str, ...] = ("param", "quality", "energy", "attribution")) -> None:
        super().__init__()
        self.heads = list(heads)
        self.log_vars = nn.ParameterDict(
            {h: nn.Parameter(torch.tensor(_INIT_LOG_VAR.get(h, 1.0))) for h in self.heads}
        )

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.zeros(())
        for h in self.heads:
            s = self.log_vars[h]
            total = total + torch.exp(-s) * losses[h] + s
        return total

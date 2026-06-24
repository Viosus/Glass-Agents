"""多头核心模型骨架（共享编码器 + 参数/质量/能耗/归因头）。

承接架构决策（项目笔记 §3）：输入〔感知特征 + 指标 + 配方 + 设备 + 工况〕的拼接表格特征
（感知特征先留占位接口）；参数头走**残差/Δ**框架（输出相对基准配方的调整量）。
I/O 维度为临时契约，待 N3 定稿。
"""

from __future__ import annotations

import torch
from torch import nn


class MultiHeadCore(nn.Module):
    """共享编码器 + 四个头。forward 返回 dict。"""

    def __init__(self, in_dim: int, hidden: int = 64, param_dim: int = 6, attr_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.param_head = nn.Linear(hidden, param_dim)    # 残差 Δ（相对基准配方）
        self.quality_head = nn.Linear(hidden, 1)          # 实测质量（海量标签）
        self.energy_head = nn.Linear(hidden, 1)           # 实测能耗（海量标签）
        self.attr_head = nn.Linear(hidden, attr_dim)      # 归因弱标签（蒸馏/反推）

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {
            "param_delta": self.param_head(z),
            "quality": self.quality_head(z),
            "energy": self.energy_head(z),
            "attribution": self.attr_head(z),
        }

    @staticmethod
    def apply_residual(baseline: torch.Tensor, param_delta: torch.Tensor) -> torch.Tensor:
        """残差策略：最终参数 = 基准配方 + Δ。"""
        return baseline + param_delta

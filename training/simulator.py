"""已知数据生成过程（DGP）的合成模拟器 —— 用于在无真数据期**验证**训练管线。

与 synthetic.py（纯随机、不可学）不同：这里特征与各头标签之间存在**已知的、可学的**关系，
故可检验"多头架构确实能学、不确定性加权有效、时间切分能防泄漏、评测门能抓回归"。

⚠️ 本文件所有系数均为**虚构**，仅用于仿真研究（统计上的"已知真值模拟"），
**绝非工厂真值**，绝不可写入 config/ 当真值使用（铁律#5/#8）。

DGP（features ~ N(0,1)，各头为 features 的线性/轻非线性映射 + 噪声）：
- param      = features @ W_param + 噪声            （参数头：残差 Δ 目标）
- quality    = features·w_q + 0.3·feat0² + 噪声      （质量头：含轻度非线性）
- energy     = features·w_e + 噪声                   （能耗头）
- attribution= features @ W_attr + 噪声             （归因头）
样本按时间顺序产出；drift>0 时在一条信息特征上加时间斜坡 → 训练(过去)与测试(未来)分布不同，
用于演示概念漂移、佐证"按时间切分而非随机打乱"。
"""

from __future__ import annotations

import torch

# 各头目标相对噪声的信号缩放（虚构）
_PARAM_SCALE = 1.0
_QUALITY_QUAD = 0.3  # 质量头轻非线性项系数（虚构）


def make_simulated(
    n: int = 256,
    in_dim: int = 12,
    param_dim: int = 6,
    attr_dim: int = 4,
    *,
    seed: int = 0,
    noise: float = 0.1,
    drift: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """生成 (features, 多头 targets)，特征→标签为已知可学映射 + 噪声。

    参数：
      noise —— 加性高斯噪声标准差（相对信号）；
      drift —— 概念漂移强度：在 feature[:,0] 上叠加 0..drift 的时间斜坡（样本按时间序）。
    返回结构同 synthetic.make_synthetic，可直接替换数据源。
    """
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(n, in_dim, generator=g)

    # 概念漂移：按时间序在第 0 维加线性斜坡（训练在前、测试在后→分布偏移）
    if drift:
        ramp = torch.linspace(0.0, drift, steps=n).unsqueeze(1)  # (n,1)
        features = features + torch.cat([ramp, torch.zeros(n, in_dim - 1)], dim=1)

    # 固定（虚构）投影矩阵：给定 seed 即确定，保证映射稳定可学
    w_param = torch.randn(in_dim, param_dim, generator=g)
    w_quality = torch.randn(in_dim, 1, generator=g)
    w_energy = torch.randn(in_dim, 1, generator=g)
    w_attr = torch.randn(in_dim, attr_dim, generator=g)

    def _noise(*shape: int) -> torch.Tensor:
        """生成给定形状的加性高斯噪声（标准差为 noise）。"""
        return noise * torch.randn(*shape, generator=g)

    param = features @ w_param * _PARAM_SCALE + _noise(n, param_dim)
    quality = features @ w_quality + _QUALITY_QUAD * features[:, :1] ** 2 + _noise(n, 1)
    energy = features @ w_energy + _noise(n, 1)
    attribution = features @ w_attr + _noise(n, attr_dim)

    targets = {"param": param, "quality": quality, "energy": energy, "attribution": attribution}
    return features, targets

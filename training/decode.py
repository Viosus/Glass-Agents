"""Δ→参数解码：把模型输出的残差 Δ 叠加到基准配方，产出结构合法的 ProcessParams。

残差策略（复用 MultiHeadCore.apply_residual）：最终参数 = 基准配方 + Δ。
**临时契约**：当前把 6 维 Δ 映射到下列 6 个标量参数；zone_temps 暂沿用基准（真实 Δ→参数映射待 N3 定稿）。
clamp 仅用于满足 pydantic 的结构正性（>0 / ≥0），**不做任何安全判定**——
安全/硬约束仍由 tools.constraints.validate 在解码之后强制（规则 > AI，安全闸门在模型之外）。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from schemas.process_params import ProcessParams
from training.model import MultiHeadCore

# Δ 的 6 个分量依次对应的目标字段（临时契约）
DELTA_FIELDS: tuple[str, ...] = (
    "temp_upper",
    "temp_lower",
    "convection_speed",
    "convection_ratio_upper_lower",
    "oscillation_amplitude",
    "heating_duration_s",
)

_EPS = 1e-6
# 各字段的结构性下限（对应 ProcessParams 的 ge/gt 约束；非安全限值）
_FLOORS: dict[str, float] = {
    "convection_speed": 0.0,
    "convection_ratio_upper_lower": _EPS,
    "oscillation_amplitude": 0.0,
    "heating_duration_s": _EPS,
}


def decode_params(param_delta: Sequence[float] | torch.Tensor, baseline: ProcessParams) -> ProcessParams:
    """基准配方 + Δ → 新的 ProcessParams（仅保证结构合法，安全交给 constraints）。

    param_delta 取前 6 个分量，依次叠加到 DELTA_FIELDS；其余字段沿用 baseline。
    """
    delta = torch.as_tensor(list(param_delta), dtype=torch.float32).flatten()
    if delta.numel() < len(DELTA_FIELDS):
        raise ValueError(f"param_delta 维度不足：需 ≥{len(DELTA_FIELDS)}，得 {delta.numel()}")

    base_vec = torch.tensor([getattr(baseline, f) for f in DELTA_FIELDS], dtype=torch.float32)
    adjusted = MultiHeadCore.apply_residual(base_vec, delta[: len(DELTA_FIELDS)])

    updates: dict[str, float] = {}
    for f, v in zip(DELTA_FIELDS, adjusted.tolist()):
        floor = _FLOORS.get(f)
        updates[f] = max(floor, v) if floor is not None else v

    # 沿用基准的其余字段（zone_temps/枚举/厚度等），仅覆盖 Δ 涉及的 6 个
    data = baseline.model_dump()
    data.update(updates)
    return ProcessParams(**data)

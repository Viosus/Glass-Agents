"""特征工程契约：ArchiveSample → 固定顺序的数值特征向量（确定性，可测）。

真数据到来前先把"哪些字段、什么顺序、缺值怎么填"定死；真数据一到即复用此契约。
原则：缺失指标不杜撰——填 0 并附 presence 标志位（让模型自己学"缺值"的含义）。
单位语义随字段（CONVENTIONS.md）；本函数不调用任何模型、不读 config。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from schemas.archive import ArchiveSample

# 固定特征顺序（新增特征往后追加，勿插队，以免破坏已训练权重的列对应）
FEATURE_NAMES: tuple[str, ...] = (
    "thickness_mm",
    "temp_upper",
    "temp_lower",
    "convection_speed",
    "convection_ratio_upper_lower",
    "oscillation_speed",
    "oscillation_amplitude",
    "heating_duration_s",
    "zone_temp_mean",
    "zone_temp_max",
    "zone_temp_min",
    "zone_temp_std",
    "glass_ultra_clear",
    "glass_clear",
    "mode_high_quality",
    "mode_high_efficiency",
    "x0_95_nm",
    "x0_95_present",
    "iso_t_pct",
    "iso_t_present",
    "ccp_value",
    "ccp_present",
    # ---- v2 追加（2026-07-02）：炉体老化协变量（原料 = FurnaceConfig 日期字段）----
    "furnace_age_years",       # 炉龄（样本时刻 − 投产日期，年）
    "furnace_age_present",
    "days_since_overhaul",     # 距上次大修天数
    "overhaul_present",
)


def feature_dim() -> int:
    """特征向量长度（= FEATURE_NAMES 个数）。"""
    return len(FEATURE_NAMES)


def _opt(value: float | None) -> tuple[float, float]:
    """可选数值 → (值, presence)：None 填 0.0 且 presence=0，否则原值且 presence=1。"""
    return (0.0, 0.0) if value is None else (float(value), 1.0)


def _days_between(sample: ArchiveSample, from_date: object) -> float | None:
    """样本时刻 − 某日期 的天数；日期缺失或**晚于样本时刻**（物理不可能=脏数据）→ None 按缺失。"""
    from datetime import date

    if not isinstance(from_date, date):
        return None
    days = (sample.created_at.date() - from_date).days
    return float(days) if days >= 0 else None


def featurize(sample: ArchiveSample) -> torch.Tensor:
    """把一条 ArchiveSample 转成 float32 特征向量（顺序见 FEATURE_NAMES）。"""
    p = sample.params
    m = sample.metrics
    zt = torch.tensor(p.zone_temps, dtype=torch.float32)

    x0_95, x0_95_present = _opt(m.x0_95_nm)
    iso_t, iso_t_present = _opt(m.iso_t_pct)
    ccp, ccp_present = _opt(m.ccp_value)

    fc = sample.furnace_config
    age_days = _days_between(sample, fc.commissioning_date) if fc is not None else None
    overhaul_days = _days_between(sample, fc.last_overhaul_date) if fc is not None else None
    age_years, age_present = _opt(None if age_days is None else age_days / 365.25)
    overhaul, overhaul_present = _opt(overhaul_days)

    vec = [
        float(p.thickness_mm),
        float(p.temp_upper),
        float(p.temp_lower),
        float(p.convection_speed),
        float(p.convection_ratio_upper_lower),
        float(p.oscillation_speed),
        float(p.oscillation_amplitude),
        float(p.heating_duration_s),
        float(zt.mean()),
        float(zt.max()),
        float(zt.min()),
        float(zt.std(unbiased=False)),  # 总体标准差：长度 1 时为 0（不产生 NaN）
        1.0 if p.glass_type == "ultra_clear" else 0.0,
        1.0 if p.glass_type == "clear" else 0.0,
        1.0 if p.quality_mode == "high_quality" else 0.0,
        1.0 if p.quality_mode == "high_efficiency" else 0.0,
        x0_95, x0_95_present,
        iso_t, iso_t_present,
        ccp, ccp_present,
        age_years, age_present,
        overhaul, overhaul_present,
    ]
    return torch.tensor(vec, dtype=torch.float32)


def featurize_many(samples: list[ArchiveSample]) -> torch.Tensor:
    """批量特征化，返回形状 (len(samples), feature_dim()) 的张量。"""
    if not samples:
        return torch.empty(0, feature_dim(), dtype=torch.float32)
    return torch.stack([featurize(s) for s in samples])

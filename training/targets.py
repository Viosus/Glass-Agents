"""参数头标签构造：ArchiveSample → (基准侧特征, Δ 标签)。真数据训练的取数层。

核心原则：
- Δ = 最终参数 − 基准参数（按 DELTA_FIELDS 逐字段，单一来源 training.decode）。
- **防标签泄漏**：features.featurize 默认吃 sample.params（最终参数=答案本身），
  参数头输入必须用 baseline_params 替换后再特征化——模型看到的是"调参前"的状态。
- baseline_params 缺失 → 该样本剔除（返回 None），不猜基准。
- 主观判级 → 数值目标的映射读 config/training.yaml（训练编码非国标真值，可调）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from schemas.archive import ArchiveSample, Grade
from schemas.bucketing import Bucket
from schemas.process_params import ProcessParams
from training.decode import DELTA_FIELDS
from training.features import FEATURE_NAMES, featurize

# 参数头标签字段（单一来源 = decode 的 Δ 契约；届时加 zone_temps 只改 decode 一处）
PARAM_TARGET_FIELDS: tuple[str, ...] = DELTA_FIELDS


def feature_schema_sha256() -> str:
    """22 维特征契约指纹：跨机器/跨版本比对（模型包导入校验用）。"""
    import hashlib

    return hashlib.sha256(",".join(FEATURE_NAMES).encode("utf-8")).hexdigest()


def delta_fields_sha256() -> str:
    """Δ 字段契约指纹：与 feature_schema_sha256 同用途。"""
    import hashlib

    return hashlib.sha256(",".join(DELTA_FIELDS).encode("utf-8")).hexdigest()

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "training.yaml"


def load_training_config(path: Path | None = None) -> dict:
    """读 config/training.yaml（每次调用按需读取，改 yaml 即时生效）。"""
    p = Path(path) if path is not None else _CONFIG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def todo_or_float(value: object) -> float | None:
    """config 值 → float；TODO(plant)/None/非数 → None（如实降级，不填占位数）。"""
    if value is None:
        return None
    if isinstance(value, str):
        return None if value.strip().startswith("TODO(plant)") else float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def param_delta_target(sample: ArchiveSample) -> torch.Tensor | None:
    """Δ 标签 = final − baseline（按 PARAM_TARGET_FIELDS）；无基准 → None（剔除，不猜）。"""
    if sample.baseline_params is None:
        return None
    final = sample.params
    base = sample.baseline_params
    delta = [float(getattr(final, f)) - float(getattr(base, f)) for f in PARAM_TARGET_FIELDS]
    return torch.tensor(delta, dtype=torch.float32)


def featurize_for_param_head(sample: ArchiveSample) -> torch.Tensor:
    """基准侧特征化（防标签泄漏）：用 baseline_params 顶替 params 后走 22 维契约。

    要求 baseline_params 非空（调用前应已按 param_delta_target 过滤）。
    """
    if sample.baseline_params is None:
        raise ValueError(f"样本 {sample.sample_id} 无 baseline_params，不能做参数头输入")
    shadow = sample.model_copy(update={"params": sample.baseline_params})
    return featurize(shadow)


def grade_score(grade: Grade | None, config: dict | None = None) -> float | None:
    """主观判级 → 数值目标（映射读 config/training.yaml 的 grade_scores）；未判级 → None。"""
    if grade is None:
        return None
    cfg = config if config is not None else load_training_config()
    scores = cfg.get("grade_scores") or {}
    v = scores.get(grade)
    return float(v) if v is not None else None


@dataclass
class ParamTrainingSet:
    """参数头训练集：基准侧特征 + Δ 标签 + 逐样本基准（解码/过闸门用）。"""

    features: torch.Tensor                 # (N, feature_dim) 基准侧特征
    deltas: torch.Tensor                   # (N, len(PARAM_TARGET_FIELDS)) Δ 标签
    baselines: list[ProcessParams]         # 逐样本基准配方，与行一一对应
    buckets: list[Bucket]                  # 各样本分桶键（按桶报指标用）
    kept_samples: list[ArchiveSample]      # 实际入训的样本（时间序）
    dropped: int = 0                       # 被剔除数（非真值/无基准）


def build_param_training_set(samples: list[ArchiveSample]) -> ParamTrainingSet:
    """过滤 → 排序 → 张量化。只收 is_ground_truth=True 且有 baseline 的样本，按 created_at 时间序。"""
    usable: list[tuple[ArchiveSample, torch.Tensor]] = []
    dropped = 0
    for s in samples:
        delta = param_delta_target(s) if s.is_ground_truth else None
        if delta is None:
            dropped += 1
            continue
        usable.append((s, delta))
    usable.sort(key=lambda pair: pair[0].created_at)  # 时间序（切分铁律：不打乱）

    if not usable:
        from training.features import feature_dim

        return ParamTrainingSet(
            features=torch.empty(0, feature_dim(), dtype=torch.float32),
            deltas=torch.empty(0, len(PARAM_TARGET_FIELDS), dtype=torch.float32),
            baselines=[],
            buckets=[],
            kept_samples=[],
            dropped=dropped,
        )

    kept = [s for s, _ in usable]
    features = torch.stack([featurize_for_param_head(s) for s in kept])
    deltas = torch.stack([d for _, d in usable])
    baselines = [s.baseline_params for s in kept if s.baseline_params is not None]  # 过滤后必非空
    return ParamTrainingSet(
        features=features,
        deltas=deltas,
        baselines=baselines,
        buckets=[s.bucket for s in kept],
        kept_samples=kept,
        dropped=dropped,
    )

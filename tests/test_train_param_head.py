"""单头训练冒烟：合成确定性规律 → 只训参数头可学（loss 降）→ 版本门可放行/可拦截 → checkpoint 完整。

不加载 GGUF、不碰真实 data/；thresholds 注入完整字典（同 test_schemas 的 full_thr 写法）。
"""

from datetime import datetime, timedelta

import pytest
import torch

from schemas.archive import ArchiveSample, MetricRecord
from schemas.process_params import ProcessParams
from tools.eval_gate import evaluate_param_gate
from training.dataset import time_ordered_split
from training.targets import build_param_training_set
from training.train_param_head import per_bucket_metrics, save_checkpoint, train_param_head

FULL_THR = {
    "gradient": {"adjacent_zone_max_delta_c": 5, "single_step_max_delta_c": 3},
    "thickness_duration": {"6": [100, 300]},
    "convection": {"clear": [1.0, 2.0]},
    "safety": {"blowup_rule": "rule_v0", "max_gradient": 50},
}

BASE = ProcessParams(
    zone_temps=[102.0, 100.0],
    zone_roles=["center", "edge"],
    temp_upper=700.0,
    temp_lower=650.0,
    convection_speed=1.0,
    convection_ratio_upper_lower=1.0,
    oscillation_speed=1.0,
    oscillation_amplitude=1.0,
    heating_duration_s=200.0,
    glass_type="clear",
    thickness_mm=6.0,
    quality_mode="high_quality",
)


def make_learnable_samples(n: int = 48) -> list[ArchiveSample]:
    """确定性 DGP（系数虚构，仅供测试可学性）：Δ(temp_upper)=0.05·(x0_95−80)，其余字段固定小增量。"""
    samples = []
    t0 = datetime(2026, 7, 2, 8, 0, 0)
    for i in range(n):
        x0_95 = 60.0 + 40.0 * i / max(n - 1, 1)
        final = BASE.model_copy(
            update={
                "temp_upper": BASE.temp_upper + 0.05 * (x0_95 - 80.0),
                "convection_speed": BASE.convection_speed + 0.3,
                "heating_duration_s": BASE.heating_duration_s + 5.0,
            }
        )
        samples.append(
            ArchiveSample(
                sample_id=f"syn{i:03d}",
                created_at=t0 + timedelta(minutes=i),
                source="synthetic/test",
                furnace_id="F1",
                thickness_mm=6.0,
                glass_type="clear",
                quality_mode="high_quality",
                is_ground_truth=True,
                params=final,
                baseline_params=BASE,
                metrics=MetricRecord(x0_95_nm=x0_95),
            )
        )
    return samples


@pytest.fixture(scope="module")
def trained():
    """训一次全模块复用（48 条、120 步，秒级）。"""
    ts = build_param_training_set(make_learnable_samples())
    tr, va, te = time_ordered_split(ts.features.shape[0])
    model, history = train_param_head(ts, tr, steps=120, device="cpu", seed=0)
    return ts, (tr, va, te), model, history


def test_loss_decreases(trained):
    _, _, _, history = trained
    assert history[-1] < history[0]                  # 确实在学


def test_gate_passes_with_full_thresholds(trained):
    ts, (tr, va, te), model, _ = trained
    gate = evaluate_param_gate(
        model,
        ts.features[te],
        ts.deltas[te],
        [ts.baselines[i] for i in te],
        thresholds=FULL_THR,
    )
    assert gate.constraints_ok, f"出参违规: {gate.details}"
    assert gate.passed                               # max_mae 未配 → 只查违规 → 放行
    assert any("TODO(plant)" in d for d in gate.details)  # 如实注明降级


def test_gate_blocks_on_tight_mae(trained):
    ts, (tr, va, te), model, _ = trained
    gate = evaluate_param_gate(
        model,
        ts.features[te],
        ts.deltas[te],
        [ts.baselines[i] for i in te],
        thresholds=FULL_THR,
        max_mae=1e-9,                                # 苛刻上限 → 必拦
    )
    assert not gate.passed and any("MAE 超限" in d for d in gate.details)


def test_gate_rejects_mismatched_lengths(trained):
    ts, _, model, _ = trained
    with pytest.raises(ValueError, match="数量不一致"):
        evaluate_param_gate(model, ts.features[:4], ts.deltas[:4], ts.baselines[:3])


def test_per_bucket_metrics_reports(trained):
    ts, (tr, va, te), model, _ = trained
    m = per_bucket_metrics(model, ts, va)
    bucket = (6.0, "clear", "high_quality")
    assert bucket in m and m[bucket]["n"] == len(va)


def test_checkpoint_contains_contract_fingerprints(trained, tmp_path):
    import json

    _, _, model, _ = trained
    weights = save_checkpoint(model, tmp_path / "ckpt", {"train_sample_count": 48})
    assert weights.exists()
    meta = json.loads((tmp_path / "ckpt" / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["feature_schema_sha256"]) == 64
    assert len(meta["delta_fields_sha256"]) == 64
    assert meta["train_sample_count"] == 48
    state = torch.load(weights, map_location="cpu", weights_only=True)
    assert any(k.startswith("param_head") for k in state)

"""标签构造单测：Δ 正确、无基准剔除、**防标签泄漏**（特征来自基准侧）、时间序、契约指纹。"""

from datetime import datetime

import pytest
import torch

from schemas.archive import ArchiveSample, MetricRecord
from schemas.process_params import ProcessParams
from training.features import FEATURE_NAMES
from training.targets import (
    PARAM_TARGET_FIELDS,
    build_param_training_set,
    delta_fields_sha256,
    feature_schema_sha256,
    featurize_for_param_head,
    grade_score,
    param_delta_target,
)


def make_params(**kw) -> ProcessParams:
    base = dict(
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
    base.update(kw)
    return ProcessParams(**base)


def make_sample(sample_id, *, minute=0, baseline=None, final=None, is_ground_truth=True) -> ArchiveSample:
    return ArchiveSample(
        sample_id=sample_id,
        created_at=datetime(2026, 7, 2, 9, minute, 0),
        source="line1/早班",
        thickness_mm=6.0,
        glass_type="clear",
        quality_mode="high_quality",
        is_ground_truth=is_ground_truth,
        params=final or make_params(),
        baseline_params=baseline,
        metrics=MetricRecord(x0_95_nm=80.0),
    )


def test_param_delta_target_is_final_minus_baseline():
    base = make_params()
    final = make_params(temp_upper=703.0, heating_duration_s=215.0)
    delta = param_delta_target(make_sample("s1", baseline=base, final=final))
    assert delta is not None
    expected = {"temp_upper": 3.0, "heating_duration_s": 15.0}
    for i, f in enumerate(PARAM_TARGET_FIELDS):
        assert float(delta[i]) == pytest.approx(expected.get(f, 0.0))


def test_missing_baseline_dropped():
    assert param_delta_target(make_sample("s1", baseline=None)) is None
    ts = build_param_training_set([make_sample("s1", baseline=None)])
    assert ts.features.shape[0] == 0 and ts.dropped == 1


def test_non_ground_truth_dropped():
    ts = build_param_training_set([make_sample("s1", baseline=make_params(), is_ground_truth=False)])
    assert ts.features.shape[0] == 0 and ts.dropped == 1


def test_featurize_uses_baseline_side_no_leakage():
    """铁证：参数头输入特征必须来自基准侧——否则等于把答案喂进输入。"""
    base = make_params(temp_upper=700.0)
    final = make_params(temp_upper=705.0)
    vec = featurize_for_param_head(make_sample("s1", baseline=base, final=final))
    i = FEATURE_NAMES.index("temp_upper")
    assert float(vec[i]) == pytest.approx(700.0)     # 基准值，而非最终值 705


def test_featurize_without_baseline_raises():
    with pytest.raises(ValueError, match="baseline_params"):
        featurize_for_param_head(make_sample("s1", baseline=None))


def test_build_set_time_ordered():
    base = make_params()
    samples = [
        make_sample("late", minute=30, baseline=base),
        make_sample("early", minute=5, baseline=base),
    ]
    ts = build_param_training_set(samples)
    assert [s.sample_id for s in ts.kept_samples] == ["early", "late"]  # 时间序，不打乱
    assert len(ts.baselines) == 2 and len(ts.buckets) == 2


def test_grade_score_reads_config_mapping():
    cfg = {"grade_scores": {"A": 1.0, "B": 0.5, "C": 0.0}}
    assert grade_score("A", cfg) == 1.0
    assert grade_score("C", cfg) == 0.0
    assert grade_score(None, cfg) is None
    assert grade_score("A", {"grade_scores": {}}) is None   # 映射缺失 → None，不猜


def test_contract_fingerprints_stable():
    f1, f2 = feature_schema_sha256(), delta_fields_sha256()
    assert len(f1) == 64 and len(f2) == 64 and f1 != f2
    assert f1 == feature_schema_sha256()             # 确定性

    delta = param_delta_target(make_sample("s1", baseline=make_params()))
    assert delta is not None and delta.dtype == torch.float32

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
    """构造一组合法工艺参数；kw 覆盖任意字段。"""
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


def make_sample(
    sample_id, *, minute=0, baseline=None, final=None, is_ground_truth=True, glass_type="clear"
) -> ArchiveSample:
    """构造一条归档样本（可指定基准/最终参数与品类）。"""
    return ArchiveSample(
        sample_id=sample_id,
        created_at=datetime(2026, 7, 2, 9, minute, 0),
        source="line1/早班",
        thickness_mm=6.0,
        glass_type=glass_type,
        quality_mode="high_quality",
        is_ground_truth=is_ground_truth,
        params=final or make_params(),
        baseline_params=baseline,
        metrics=MetricRecord(x0_95_nm=80.0),
    )


def test_param_delta_target_is_final_minus_baseline():
    """参数增量标签 = 最终 − 基准，逐字段相减。"""
    base = make_params()
    final = make_params(temp_upper=703.0, heating_duration_s=215.0)
    delta = param_delta_target(make_sample("s1", baseline=base, final=final))
    assert delta is not None
    expected = {"temp_upper": 3.0, "heating_duration_s": 15.0}
    for i, f in enumerate(PARAM_TARGET_FIELDS):
        assert float(delta[i]) == pytest.approx(expected.get(f, 0.0))


def test_missing_baseline_dropped():
    """无基准参数的样本不产生增量标签（不猜基准）。"""
    assert param_delta_target(make_sample("s1", baseline=None)) is None
    ts = build_param_training_set([make_sample("s1", baseline=None)])
    assert ts.features.shape[0] == 0 and ts.dropped == 1


def test_non_ground_truth_dropped():
    """未经专家复核的样本不进训练集。"""
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
    """缺基准时特征化应显式报错，而非静默填 0。"""
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


# ------------------ v2 老化特征（22→26 维）/ v3 品类补位（26→31 维） ------------------ #
def test_feature_dim_is_31():
    from training.features import feature_dim

    assert feature_dim() == 31
    # v2 老化四项仍在原位（v3 追加在其后，未插队 —— features.py 的「勿插队」纪律）
    assert FEATURE_NAMES[-9:-5] == (
        "furnace_age_years", "furnace_age_present", "days_since_overhaul", "overhaul_present",
    )
    assert FEATURE_NAMES[-5:] == (
        "glass_low_e", "glass_coated", "glass_enameled", "glass_patterned", "glass_other",
    )


def test_every_glass_type_gets_exactly_one_hot():
    """7 品类各自恰好点亮一个 one-hot 位 —— 修复前 Low-E 等五类全塌缩成全 0。"""
    from schemas.process_params import GLASS_TYPES
    from training.features import FEATURE_NAMES as NAMES
    from training.features import featurize

    idx = [i for i, n in enumerate(NAMES) if n.startswith("glass_")]
    assert len(idx) == len(GLASS_TYPES), "品类 one-hot 位数必须等于品类数"
    for gt in GLASS_TYPES:
        # sample 与 params 两处品类保持一致（featurize 读的是 params.glass_type）
        vec = featurize(make_sample(f"g-{gt}", glass_type=gt, final=make_params(glass_type=gt)))
        hot = [NAMES[i] for i in idx if vec[i] == 1.0]
        assert hot == [f"glass_{gt}"], f"{gt} 应恰好点亮 glass_{gt}，实际 {hot}"


def test_aging_features_from_furnace_config():
    from datetime import date

    from schemas.furnace import FurnaceConfig
    from training.features import featurize

    fc = FurnaceConfig(
        furnace_id="F1",
        commissioning_date=date(2016, 7, 2),      # 距样本时刻(2026-07-02 09:00)整 10 年
        last_overhaul_date=date(2026, 6, 2),      # 距样本 30 天
    )
    sample = make_sample("s1", baseline=make_params()).model_copy(update={"furnace_config": fc})
    vec = featurize(sample)
    i = FEATURE_NAMES.index("furnace_age_years")
    assert float(vec[i]) == pytest.approx(10.0, abs=0.02)     # 3652/365.25 ≈ 10.0
    assert float(vec[i + 1]) == 1.0
    assert float(vec[FEATURE_NAMES.index("days_since_overhaul")]) == pytest.approx(30.0)
    assert float(vec[FEATURE_NAMES.index("overhaul_present")]) == 1.0


def test_aging_features_missing_and_dirty():
    from datetime import date

    from schemas.furnace import FurnaceConfig
    from training.features import featurize

    plain = make_sample("s1", baseline=make_params())         # 无 furnace_config
    vec = featurize(plain)
    for name in ("furnace_age_years", "furnace_age_present", "days_since_overhaul", "overhaul_present"):
        assert float(vec[FEATURE_NAMES.index(name)]) == 0.0   # 缺 → 0+presence=0

    dirty_fc = FurnaceConfig(furnace_id="F1", commissioning_date=date(2030, 1, 1))  # 晚于样本=脏
    dirty = plain.model_copy(update={"furnace_config": dirty_fc})
    vec2 = featurize(dirty)
    assert float(vec2[FEATURE_NAMES.index("furnace_age_present")]) == 0.0  # 脏日期按缺失，不污染


def test_contract_fingerprints_stable():
    f1, f2 = feature_schema_sha256(), delta_fields_sha256()
    assert len(f1) == 64 and len(f2) == 64 and f1 != f2
    assert f1 == feature_schema_sha256()             # 确定性

    delta = param_delta_target(make_sample("s1", baseline=make_params()))
    assert delta is not None and delta.dtype == torch.float32

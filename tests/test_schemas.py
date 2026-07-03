"""schemas 数据层单测：脏数据被拒、合法样本可写读、分桶计数正确。"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from schemas.archive import (
    ArchiveSample,
    ImageRef,
    MetricRecord,
    load_all,
    read_sample,
    write_sample,
)
from schemas.bucketing import bucket_table, count_ground_truth_by_bucket
from schemas.process_params import ProcessParams

SHA = "a" * 64  # 合法 64 位十六进制


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


def make_sample(
    sample_id, thickness_mm=6.0, glass_type="clear", quality_mode="high_quality", is_ground_truth=False
) -> ArchiveSample:
    return ArchiveSample(
        sample_id=sample_id,
        created_at=datetime(2026, 6, 24, 10, 0, 0),
        source="line-A/shift-1/teacher-v0",
        thickness_mm=thickness_mm,
        glass_type=glass_type,
        quality_mode=quality_mode,
        is_ground_truth=is_ground_truth,
        stress_image=ImageRef(path="img/0001.png", sha256=SHA, width_px=200, height_px=200, mm_per_px=1.0),
        params=make_params(thickness_mm=thickness_mm, glass_type=glass_type, quality_mode=quality_mode),
        metrics=MetricRecord(x0_95_nm=65.0, x0_95_grade="A"),
    )


# --------------------------- ProcessParams 校验 --------------------------- #
def test_valid_params_to_param_set_passes_gate():
    from tools.constraints import validate

    full_thr = {
        "gradient": {"adjacent_zone_max_delta_c": 5, "single_step_max_delta_c": 3},
        "thickness_duration": {"6": [100, 300]},
        "convection": {"clear": [1.0, 2.0]},
        "safety": {"blowup_rule": "rule_v0", "max_gradient": 50},
    }
    res = validate(make_params().to_param_set(), thresholds=full_thr)
    assert res.within_limits is True


def test_dirty_length_mismatch_rejected():
    with pytest.raises(ValidationError):
        make_params(zone_roles=["center"])  # 与 zone_temps(2) 不等长


def test_dirty_bad_glass_type_rejected():
    with pytest.raises(ValidationError):
        make_params(glass_type="green")


def test_dirty_negative_thickness_rejected():
    with pytest.raises(ValidationError):
        make_params(thickness_mm=-1.0)


def test_dirty_extra_field_rejected():
    with pytest.raises(ValidationError):
        ProcessParams(
            zone_temps=[100.0],
            zone_roles=["center"],
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
            bogus=123,
        )


def test_dirty_bad_sha256_rejected():
    with pytest.raises(ValidationError):
        ImageRef(path="x", sha256="tooshort", width_px=10, height_px=10, mm_per_px=1.0)


# --------------------------- 写读 + 分桶 --------------------------- #
def test_write_read_roundtrip_and_bucketing(tmp_path):
    samples = [
        make_sample("s1", is_ground_truth=True),
        make_sample("s2", is_ground_truth=True),
        make_sample("s3", is_ground_truth=False),
        make_sample("s4", thickness_mm=8.0, glass_type="ultra_clear", is_ground_truth=True),
        make_sample(
            "s5", thickness_mm=8.0, glass_type="ultra_clear", quality_mode="high_efficiency", is_ground_truth=True
        ),
    ]
    for s in samples:
        write_sample(s, tmp_path)

    loaded = load_all(tmp_path)
    assert len(loaded) == 5
    by_id = {s.sample_id: s for s in loaded}
    assert by_id["s1"].metrics.x0_95_grade == "A"      # 读回内容一致
    assert by_id["s4"].glass_type == "ultra_clear"

    gt = count_ground_truth_by_bucket(loaded)
    assert gt[(6.0, "clear", "high_quality")] == 2      # s1,s2（s3 非真值不计）
    assert gt[(8.0, "ultra_clear", "high_quality")] == 1
    assert gt[(8.0, "ultra_clear", "high_efficiency")] == 1

    row = next(r for r in bucket_table(loaded) if r[0] == (6.0, "clear", "high_quality"))
    assert row[1] == 3 and row[2] == 2                  # total=3, ground_truth=2


# --------------------------- v2 新字段向后兼容 --------------------------- #
def test_v2_new_fields_have_defaults():
    """老构造方式（不传新字段）零改动可用：炉体身份取诚实缺省。"""
    s = make_sample("s1")
    assert s.furnace_id == "unknown"
    assert s.furnace_config is None
    assert s.operator_id is None and s.repeat_group_id is None and s.condition_note is None
    assert s.metrics.fringe_score_0_100 is None


def test_furnace_config_date_fields_default_none():
    """FurnaceConfig 新增日期字段向后兼容：不传默认 None（老化特征按缺失处理）。"""
    from schemas.furnace import FurnaceConfig

    fc = FurnaceConfig(furnace_id="F1")
    assert fc.commissioning_date is None and fc.last_overhaul_date is None


def test_v1_json_without_new_fields_reads_back(tmp_path):
    """v1 时代落库的 JSON（无 furnace_id 等键）读回按默认值补齐，不报错。"""
    import json

    old = json.loads(make_sample("s-old").model_dump_json())
    for key in ("furnace_id", "furnace_config", "operator_id", "repeat_group_id", "condition_note"):
        old.pop(key, None)
    old["metrics"].pop("fringe_score_0_100", None)
    p = tmp_path / "s-old.json"
    p.write_text(json.dumps(old), encoding="utf-8")

    s = read_sample(p)
    assert s.furnace_id == "unknown" and s.furnace_config is None


def test_dirty_archive_json_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"sample_id":"x","created_at":"2026-06-24T10:00:00","source":"s",'
        '"thickness_mm":6.0,"glass_type":"clear","quality_mode":"high_quality",'
        '"params":{},"bogus":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        read_sample(bad)

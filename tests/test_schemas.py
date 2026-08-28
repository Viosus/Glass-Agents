"""schemas 数据层单测：脏数据被拒、合法样本可写读、分桶计数正确。"""

from datetime import datetime
from typing import get_args

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
from schemas.process_params import GLASS_TYPE_ZH, GLASS_TYPES, GlassType, ProcessParams

SHA = "a" * 64  # 合法 64 位十六进制


def make_params(**kw) -> ProcessParams:
    """构造一组合法工艺参数；kw 覆盖任意字段用于造脏数据。"""
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
    """构造一条合法归档样本（分桶键可覆盖）。"""
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
    """脏数据：品类不在枚举内 → 写入即拒。"""
    with pytest.raises(ValidationError):
        make_params(glass_type="green")


# ------------------- 品类枚举三处一致（2026-08-23 扩为 7 值） ------------------- #
def test_glass_type_sources_agree():
    """GLASS_TYPES / GLASS_TYPE_ZH / GlassType 必须逐值一致。

    Literal 无法由元组动态构造（类型检查器要求字面量），只能三处并列 + 本测试锁定；
    漏改任一处即测试红，避免枚举再次悄悄漂移（正是本次要修的历史问题）。
    """
    assert set(get_args(GlassType)) == set(GLASS_TYPES), "GlassType 与 GLASS_TYPES 不一致"
    assert set(GLASS_TYPE_ZH) == set(GLASS_TYPES), "GLASS_TYPE_ZH 与 GLASS_TYPES 不一致"
    assert len(GLASS_TYPES) == len(set(GLASS_TYPES)), "GLASS_TYPES 有重复值"


def test_all_glass_types_accepted_by_params_and_archive():
    """7 品类都必须能进 ProcessParams 与 ArchiveSample。

    这是本次修复的核心回归：Low-E/彩釉/压花/镀膜 曾因枚举只认两值被 ingest 整行拒收。
    """
    for gt in GLASS_TYPES:
        assert make_params(glass_type=gt).glass_type == gt
        assert make_sample(f"s-{gt}", glass_type=gt).glass_type == gt


def test_glass_type_matches_annotation_template_enum():
    """39 列标注表的下拉允许值必须与权威枚举同源（表头不变，只扩允许值）。"""
    from tools.make_annotation_template import ENUMS

    assert list(ENUMS["glass_type"]) == list(GLASS_TYPES)


def test_dirty_negative_thickness_rejected():
    """脏数据：厚度为负 → 写入即拒（gt=0）。"""
    with pytest.raises(ValidationError):
        make_params(thickness_mm=-1.0)


def test_dirty_extra_field_rejected():
    """脏数据：多出未定义字段 → 写入即拒（extra=forbid）。"""
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
    """脏数据：图像哈希格式非法 → 写入即拒。"""
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
    """脏数据：归档 JSON 结构非法 → 读回时拒。"""
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"sample_id":"x","created_at":"2026-06-24T10:00:00","source":"s",'
        '"thickness_mm":6.0,"glass_type":"clear","quality_mode":"high_quality",'
        '"params":{},"bogus":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        read_sample(bad)

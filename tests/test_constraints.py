"""tools/constraints.py 的硬约束单测（覆盖临界值）。"""

import copy

from tools.constraints import ParamSet, validate

# 一套"全部填齐"的阈值，用来隔离测某一条 C1 规则（避免被 TODO 缺值项干扰）。
FULL_THR = {
    "gradient": {"adjacent_zone_max_delta_c": 5, "single_step_max_delta_c": 3},
    "thickness_duration": {"6": [100, 300]},
    "convection": {"clear": [1.0, 2.0]},
    "safety": {"blowup_rule": "rule_v0", "max_gradient": 50},
}


def make_param(zone_temps, zone_roles=None, temp_upper=700.0, temp_lower=650.0):
    n = len(zone_temps)
    return ParamSet(
        zone_temps=list(zone_temps),
        zone_roles=zone_roles if zone_roles is not None else ["center"] * n,
        temp_upper=temp_upper,
        temp_lower=temp_lower,
        convection_speed=1.0,
        convection_ratio_upper_lower=1.0,
        oscillation_speed=1.0,
        oscillation_amplitude=1.0,
        heating_duration_s=200.0,
        glass_type="clear",
        thickness_mm=6.0,
        quality_mode="high_quality",
    )


def _has_c1(res):
    return any(v.startswith("C1") for v in res.violations)


# --------------------------- C1 相邻分区温差 --------------------------- #
def test_adjacent_delta_at_limit_passes():
    res = validate(make_param([100.0, 105.0]), thresholds=FULL_THR)
    assert res.within_limits is True
    assert not _has_c1(res)


def test_adjacent_delta_over_limit_blocked():
    res = validate(make_param([100.0, 105.01]), thresholds=FULL_THR)
    assert res.within_limits is False
    assert _has_c1(res)


# --------------------------- C1 单次调温幅度 --------------------------- #
def test_single_step_at_limit_passes():
    prev = make_param([100.0, 100.0])
    cur = make_param([103.0, 100.0])
    res = validate(cur, prev=prev, thresholds=FULL_THR)
    assert res.within_limits is True
    assert not _has_c1(res)


def test_single_step_over_limit_blocked():
    prev = make_param([100.0, 100.0])
    cur = make_param([103.01, 100.0])
    res = validate(cur, prev=prev, thresholds=FULL_THR)
    assert res.within_limits is False
    assert _has_c1(res)


def test_prev_none_skips_single_step():
    # 与 prev 相比会超 3℃，但 prev=None 应跳过该规则
    res = validate(make_param([100.0, 101.0]), prev=None, thresholds=FULL_THR)
    assert res.within_limits is True


# --------------------------- C1 中心>边缘 --------------------------- #
def test_center_not_greater_than_edge_blocked():
    res = validate(
        make_param([100.0, 101.0], zone_roles=["center", "edge"]),
        thresholds=FULL_THR,
    )
    assert res.within_limits is False
    assert any("center" in v for v in res.violations)


# --------------------------- 安全红线缺值 --------------------------- #
def test_missing_safety_max_gradient_is_unpassable():
    thr = copy.deepcopy(FULL_THR)
    thr["safety"]["max_gradient"] = "TODO(plant)"
    res = validate(make_param([100.0, 102.0]), thresholds=thr)
    assert res.within_limits is False
    assert any("max_gradient" in v for v in res.violations)


def test_default_config_blocks_due_to_todo():
    # 默认 config/thresholds.yaml 多数为 TODO(plant) → 一律不放行
    res = validate(make_param([100.0, 102.0]))
    assert res.within_limits is False


# --------------------------- C1 中心>边缘 边界 --------------------------- #
def test_center_equals_edge_blocked():
    # 边界：center == edge 不满足"严格大于" → 拦
    res = validate(
        make_param([100.0, 100.0], zone_roles=["center", "edge"]),
        thresholds=FULL_THR,
    )
    assert res.within_limits is False
    assert _has_c1(res)


def test_center_just_above_edge_passes():
    res = validate(
        make_param([100.01, 100.0], zone_roles=["center", "edge"]),
        thresholds=FULL_THR,
    )
    assert res.within_limits is True
    assert not _has_c1(res)


# --------------------------- C2 / C3 缺值拦截 --------------------------- #
def test_c2_thickness_duration_todo_blocks():
    thr = copy.deepcopy(FULL_THR)
    thr["thickness_duration"] = "TODO(plant)"
    res = validate(make_param([100.0, 102.0]), thresholds=thr)
    assert res.within_limits is False
    assert any(v.startswith("C2") for v in res.violations)


def test_c3_convection_todo_blocks():
    thr = copy.deepcopy(FULL_THR)
    thr["convection"] = "TODO(plant)"
    res = validate(make_param([100.0, 102.0]), thresholds=thr)
    assert res.within_limits is False
    assert any(v.startswith("C3") for v in res.violations)


# --------------------------- 全阈值齐备 → 放行 --------------------------- #
def test_all_thresholds_present_passes_clean():
    # 阈值全部填齐 + 合法参数 + 无越界调温 → within_limits=True 且 violations 为空
    prev = make_param([100.0, 102.0])
    cur = make_param([100.0, 102.0])
    res = validate(cur, prev=prev, thresholds=FULL_THR)
    assert res.within_limits is True
    assert res.violations == []
    assert res.gradient_ok is True
    assert res.blow_up_risk is False

"""tools/constraints.py 的硬约束单测（覆盖临界值）。"""

import copy

import pytest

from tools.constraints import ParamSet, _is_todo, load_thresholds, validate

# 一套"全部填齐"的阈值，用来隔离测某一条 C1 规则（避免被 TODO 缺值项干扰）。
FULL_THR = {
    "gradient": {"adjacent_zone_max_delta_c": 5, "single_step_max_delta_c": 3},
    "thickness_duration": {"6": [100, 300]},
    "convection": {"clear": [1.0, 2.0]},
    "safety": {"blowup_rule": "rule_v0", "max_gradient": 50},
}


def make_param(zone_temps, zone_roles=None, temp_upper=700.0, temp_lower=650.0):
    """构造一组参数（分区温度必给，其余走安全默认值）。"""
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
    """判断结果里是否含 C1 类违规。"""
    return any(v.startswith("C1") for v in res.violations)


# --------------------------- C1 相邻分区温差 --------------------------- #
def test_adjacent_delta_at_limit_passes():
    res = validate(make_param([100.0, 105.0]), thresholds=FULL_THR)
    assert res.within_limits is True
    assert not _has_c1(res)


def test_adjacent_delta_over_limit_blocked():
    """相邻分区温差越限（>5℃）→ 拦截。"""
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
    """单次调温越限（>±3℃）→ 拦截。"""
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
    """中心温度刚好高于边缘 → 放行（临界值）。"""
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
    """C3 对流阈值缺失 → 无法判定→不通过。"""
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


# ==================================================================================== #
# 哨兵：C2 / C3 / 安全红线 目前**只判「键在不在」，没有任何比较逻辑**
#
# validate() 里这四段全是 `if _is_todo(x): violations.append(...)` —— 一旦某项被填上真值，
# 那个 if 就不进、什么都不做，within_limits 直接变 True，**一个数都没比过**。
# blowup_rule 更险：填上真值后 `blow_up_risk = True` 这行不执行 → 变 False「无风险」，
# 但根本没跑任何炸板判定。这比现在的保守拒绝危险得多。
#
# 真值缺口（docs/03-hard-constraints.md 至今不存在，信息需求清单 A3~A6 未回收）无法靠写代码补上，
# 所以本段不修逻辑，只**上锁**：谁哪天填了真值，pytest 当场红，pre_commit_gate 直接拦下提交。
# ==================================================================================== #
_UNIMPLEMENTED_THRESHOLDS = {
    "thickness_duration": "C2 厚度→加热时长区间",
    "convection": "C3 品类→风速区间 / 上下对流配比范围",
    "safety.max_gradient": "安全红线·温度梯度上限",
    "safety.blowup_rule": "安全红线·炸板判定规则",
}


def _dig(thr: dict, dotted: str):
    """按点号路径取值；中途不是 dict 或键缺失 → None（_is_todo(None) 为真）。"""
    cur = thr
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def test_sentinel_unimplemented_thresholds_must_stay_todo():
    """这四项**必须**保持 TODO(plant)，直到比较逻辑与真值同一笔提交落地。

    填真值而不写比较逻辑 = 安全闸门静默变绿。本测试是那个隐患的唯一自动化防线。
    """
    thr = load_thresholds()
    filled = [
        f"  · {key}（{zh}）= {_dig(thr, key)!r}"
        for key, zh in _UNIMPLEMENTED_THRESHOLDS.items()
        if not _is_todo(_dig(thr, key))
    ]
    if filled:
        pytest.fail(
            "config/thresholds.yaml 中以下项已填真值，但 tools/constraints.py 对它们"
            "**只判存在性、没有比较逻辑**——照此提交会让安全闸门静默放行：\n"
            + "\n".join(filled)
            + "\n\n正确做法（三件事同一笔提交，缺一不可）：\n"
            "  1. 在 tools/constraints.validate() 里为该项写真正的比较逻辑；\n"
            "  2. 为该逻辑补临界值单测（照 C1 那几条的写法）；\n"
            "  3. 把该项从本文件 _UNIMPLEMENTED_THRESHOLDS 里删掉，并同步 AnnotationApp "
            "server/vendor/（那边 test_vendor_sync 逐字节比对，不同步会红）。\n"
            "在此之前请把该项改回 TODO(plant)——缺值判「无法判定→不通过」是本项目铁律。"
        )


def test_todo_segments_stay_conservative():
    """锁定 TODO 期间的保守行为：四项缺值都必须拒绝，炸板风险必须保守置真。

    与上面的哨兵配对——哨兵防「填真值静默变绿」，本条防「有人把这几个 if 删了」。
    """
    res = validate(make_param([100.0, 102.0]), thresholds={"gradient": FULL_THR["gradient"]})
    assert res.within_limits is False
    assert res.blow_up_risk is True, "无炸板判定规则时必须保守视为有风险"
    for frag in ("C2", "C3", "max_gradient", "blowup_rule"):
        assert any(frag in v for v in res.violations), f"缺 {frag} 的 violation"

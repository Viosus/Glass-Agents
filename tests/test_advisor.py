"""研判输出层（advisor）单测：六项输出的两态行为（缺真值拦截 / 注入参考出值）。

口径：默认 config 的真值全为 TODO(plant) → 对应节 cannot_determine 且 missing 列明缺口；
注入合成参考（ref）→ 出值且 is_calibrated=False（不当真值下发）——与 CCP 模式一致。
"""

from advisor import advise, report_to_text
from advisor.attribution import attribute
from advisor.energy import estimate_energy
from advisor.loading import suggest_loading
from advisor.maintenance import assess_maintenance
from schemas.inputs import EquipmentUsage
from schemas.process_params import ProcessParams


def make_params(**overrides) -> ProcessParams:
    """构造一组结构合法的工艺参数（数值仅供结构测试，非工艺真值）。"""
    data = dict(
        zone_temps=[700.0, 690.0, 680.0],
        zone_roles=["center", "center", "edge"],
        temp_upper=705.0,
        temp_lower=695.0,
        convection_speed=50.0,
        convection_ratio_upper_lower=1.2,
        oscillation_speed=10.0,
        oscillation_amplitude=5.0,
        heating_duration_s=180.0,
        glass_type="clear",
        thickness_mm=6.0,
        quality_mode="high_quality",
    )
    data.update(overrides)
    return ProcessParams(**data)


ENERGY_REF = {"ref_temp_c": 600.0, "kw_per_zone_c": 1.5, "fan_kw_per_speed": 0.8, "base_kw": 20.0}

MAINT_REF = {
    "weights": {"hours": 0.5, "cycles": 0.3, "load": 0.2},
    "service_wear_threshold": 0.6,
    "components": [
        {"name": "热电偶", "rated_hours": 10000.0, "rated_cycles": 5000.0},
        {"name": "对流风机", "rated_hours": 20000.0, "rated_cycles": 8000.0},
    ],
}

LOADING_CFG = {"strategy": "grid", "bed_length_mm": 3000.0, "bed_width_mm": 2000.0, "min_gap_mm": 50.0}


# ---------------- 能耗线 ----------------
def test_energy_default_config_cannot_determine():
    # 默认 config 系数全 TODO(plant) → 无法判定且列明缺口，绝不出占位数
    adv = estimate_energy(make_params())
    assert not adv.status.ok
    assert adv.status.missing and adv.estimate_kwh is None
    assert "最低能耗方案" in adv.plan_note


def test_energy_injected_ref_uncalibrated_value():
    # 注入合成系数 → 出估计值，显式标注未标定
    adv = estimate_energy(make_params(), ref=ENERGY_REF)
    assert adv.status.ok and adv.estimate_kwh is not None and adv.estimate_kwh > 0
    assert adv.is_calibrated is False


def test_energy_monotonic_in_duration():
    # 加热时长更长 → 能耗估计更大（代理公式基本理性）
    short = estimate_energy(make_params(heating_duration_s=100.0), ref=ENERGY_REF)
    long = estimate_energy(make_params(heating_duration_s=300.0), ref=ENERGY_REF)
    assert long.estimate_kwh > short.estimate_kwh


def test_energy_without_params():
    # 上游未产参 → 如实标缺，不猜
    adv = estimate_energy(None, ref=ENERGY_REF)
    assert not adv.status.ok and "工艺参数" in adv.status.missing[0]


# ---------------- 维保线 ----------------
def test_maintenance_default_config_cannot_determine():
    # 默认 config 权重/额定值全 TODO(plant) → 无法判定
    usage = EquipmentUsage(run_hours=5000.0, changeover_count=1000, load_frac=0.7)
    adv = assess_maintenance(usage)
    assert not adv.status.ok and adv.status.missing


def test_maintenance_injected_ref_wear_and_service():
    # 注入合成规则 → 老化度∈[0,1]；达阈值部件列入检修项目；未标定标注
    usage = EquipmentUsage(run_hours=9000.0, changeover_count=4000, load_frac=0.9)
    adv = assess_maintenance(usage, ref=MAINT_REF)
    assert adv.status.ok and adv.is_calibrated is False
    assert all(0.0 <= c.wear_frac <= 1.0 for c in adv.components)
    # 热电偶：0.5*0.9 + 0.3*0.8 + 0.2*0.9 = 0.87 ≥ 0.6 → 检修
    assert "热电偶" in adv.service_items


def test_maintenance_wear_monotonic_in_hours():
    # 运行小时更长 → 老化度不降（加权公式单调性）
    low = assess_maintenance(EquipmentUsage(run_hours=1000.0, changeover_count=0, load_frac=0.0), ref=MAINT_REF)
    high = assess_maintenance(EquipmentUsage(run_hours=8000.0, changeover_count=0, load_frac=0.0), ref=MAINT_REF)
    assert high.components[0].wear_frac > low.components[0].wear_frac


def test_maintenance_without_usage():
    # 无设备运行数据 → 如实标缺（信息需求清单 B4）
    adv = assess_maintenance(None, ref=MAINT_REF)
    assert not adv.status.ok and "B4" in adv.status.missing[0]


# ---------------- 摆炉线 ----------------
def test_loading_default_config_cannot_determine():
    # 默认 config 炉床尺寸/间距全 TODO(plant) → 无法判定
    adv = suggest_loading(900.0, 600.0)
    assert not adv.status.ok and adv.status.missing


def test_loading_grid_arithmetic():
    # 床 3000×2000、玻璃 900×600、间距 50 → 行=floor(3050/950)=3、列=floor(2050/650)=3 → 9 片
    adv = suggest_loading(900.0, 600.0, config=LOADING_CFG)
    assert adv.status.ok
    assert adv.sheets_per_bed == 9 and adv.layout == "3行×3列" and adv.gap_mm == 50.0


def test_loading_oversized_glass_rejected():
    # 玻璃超出炉床 → 无法排布，如实拒绝
    adv = suggest_loading(4000.0, 600.0, config=LOADING_CFG)
    assert not adv.status.ok


# ---------------- 归因线 ----------------
def test_attribution_default_passes_violations_through():
    # 映射表 TODO(plant) → 无法判定；但闸门 violations 仍无条件透传为线索
    adv = attribute(violations=["C2: 厚度→时长映射缺失"], signals={"centrality": 0.9})
    assert not adv.status.ok
    assert any("C2" in s.evidence for s in adv.suspects)


def test_attribution_injected_rules_hit():
    # 注入映射表：gte/lte 两向规则各验证一条；zone 恒 None（A9 未定）
    cfg = {"rules": [
        {"signal": "centrality", "op": "gte", "threshold": 0.7, "issue": "局部过热/欠温（斑集中于中心）"},
        {"signal": "fringe_score_0_100", "op": "lte", "threshold": 60.0, "issue": "整体应力斑偏重"},
    ]}
    adv = attribute(signals={"centrality": 0.85, "fringe_score_0_100": 55.0}, config=cfg)
    assert adv.status.ok
    issues = {s.issue for s in adv.suspects}
    assert "局部过热/欠温（斑集中于中心）" in issues and "整体应力斑偏重" in issues
    assert all(s.zone is None for s in adv.suspects)


# ---------------- 装配与报告 ----------------
def test_advise_all_sections_present_and_honest():
    # 什么都不给 → 报告六节俱全，全部如实 cannot_determine（不猜）
    report = advise()
    assert not report.params.status.ok
    assert not report.energy.status.ok
    assert not report.maintenance.status.ok
    assert not report.loading.status.ok
    assert not report.attribution.status.ok
    text = report_to_text(report)
    assert "研判报告" in text and "无法判定" in text


def test_advise_with_params_runs_gate_and_feeds_attribution():
    # 传参未传 check → 装配时自动过安全闸门；violations 传导进归因线索
    report = advise(params=make_params(), refs={"energy": ENERGY_REF})
    assert report.params.check is not None          # 规则 > AI：出参必过闸门
    assert report.params.check.violations           # 当前阈值多为 TODO → 必有违规
    assert any("硬约束违规" in s.issue for s in report.attribution.suspects)
    assert report.energy.status.ok and report.energy.is_calibrated is False
    text = report_to_text(report)
    assert "未标定" in text


def test_advise_delta_signals_reach_attribution():
    # 建议 vs 基准的 Δ 进入归因信号：delta_heating_duration_s 触发注入规则
    cfg = {"attribution": {"rules": [
        {"signal": "delta_heating_duration_s", "op": "gte", "threshold": 30.0, "issue": "加热时长上调过大"},
    ]}}
    report = advise(
        params=make_params(heating_duration_s=240.0),
        baseline=make_params(heating_duration_s=180.0),
        configs=cfg,
    )
    assert any(s.issue == "加热时长上调过大" for s in report.attribution.suspects)


# ---------------- 新字段（讨论稿 §4.2-2） ----------------
def test_new_optional_fields_roundtrip():
    # convection_temp / fan_startup_logic：可选、默认 None、能 roundtrip 到 ParamSet
    p_default = make_params()
    assert p_default.convection_temp is None and p_default.fan_startup_logic is None
    p = make_params(convection_temp=45.0, fan_startup_logic="低速预启动")
    ps = p.to_param_set()
    assert ps.convection_temp == 45.0 and ps.fan_startup_logic == "低速预启动"

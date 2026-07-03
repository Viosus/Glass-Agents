"""研判报告装配：advise() 一次给全讨论稿 §4.2 六项输出 + 中文摘要 report_to_text()。

定位（结构重构点）：此前六项输出散落各模块——本层把它们装配成一个 AdvisoryReport。
不内嵌 Teacher LLM：上游（llm_roles/teacher_loop 或对话壳）产参后传入 params/rationale，
本层只做确定性装配与闸门校验，保持可测与解耦；LLM 永远不产数字（规则 > AI）。
"""

from __future__ import annotations

from advisor.attribution import attribute
from advisor.energy import estimate_energy
from advisor.loading import suggest_loading
from advisor.maintenance import assess_maintenance
from advisor.report import AdvisoryReport, ParamsAdvice, SectionStatus
from schemas.inputs import EnvironmentInput, EquipmentUsage, OpticalFeatureSlots
from schemas.process_params import ProcessParams
from tools.constraints import CheckResult, validate

# 参数残差进归因信号的数值字段（Δ = 建议 − 基准）
_DELTA_FIELDS = (
    "temp_upper", "temp_lower", "convection_speed", "convection_ratio_upper_lower",
    "oscillation_speed", "oscillation_amplitude", "heating_duration_s",
)


def _params_section(
    params: ProcessParams | None, check: CheckResult | None, rationale: str
) -> ParamsAdvice:
    """参数节装配：有参必过闸门（未传 check 则现场校验）；无参如实标缺。"""
    if params is None:
        return ParamsAdvice(
            status=SectionStatus(ok=False, missing=["工艺参数建议（上游 Teacher/多头核心未产参）"]),
            rationale=rationale,
        )
    if check is None:
        check = validate(params.to_param_set())  # 规则 > AI：出参必过安全闸门
    return ParamsAdvice(status=SectionStatus(ok=True), params=params, check=check, rationale=rationale)


def _delta_signals(params: ProcessParams | None, baseline: ProcessParams | None) -> dict:
    """建议参数 − 基准参数 → delta_* 信号（供归因映射表用）；缺任一方则为空。"""
    if params is None or baseline is None:
        return {}
    return {
        f"delta_{k}": getattr(params, k) - getattr(baseline, k)
        for k in _DELTA_FIELDS
    }


def advise(
    *,
    params: ProcessParams | None = None,
    check: CheckResult | None = None,
    rationale: str = "",
    baseline: ProcessParams | None = None,
    usage: EquipmentUsage | None = None,
    quality_signals: dict | None = None,
    glass_length_mm: float | None = None,
    glass_width_mm: float | None = None,
    environment: EnvironmentInput | None = None,
    optical: OpticalFeatureSlots | None = None,
    configs: dict | None = None,
    refs: dict | None = None,
) -> AdvisoryReport:
    """装配一份完整研判报告（六项 section 恒在；缺真值的节如实 cannot_determine）。

    - quality_signals：质量诊断量字典（如 fringe_score_0_100/centrality），与 Δ 合并进归因；
    - configs/refs：按节注入配置与未标定参考（键 energy/maintenance/loading/attribution），
      测试与演示用；生产路径读 config/*.yaml。
    """
    cfgs = configs or {}
    injections = refs or {}

    params_advice = _params_section(params, check, rationale)

    signals = dict(quality_signals or {})
    signals.update(_delta_signals(params, baseline))
    violations = list(params_advice.check.violations) if params_advice.check is not None else []

    return AdvisoryReport(
        params=params_advice,
        energy=estimate_energy(params, config=cfgs.get("energy"), ref=injections.get("energy")),
        maintenance=assess_maintenance(usage, config=cfgs.get("maintenance"), ref=injections.get("maintenance")),
        loading=suggest_loading(glass_length_mm, glass_width_mm, config=cfgs.get("loading")),
        attribution=attribute(violations=violations, signals=signals, config=cfgs.get("attribution")),
        environment=environment,
        optical=optical,
    )


def _status_line(name: str, status: SectionStatus) -> list[str]:
    """一节的状态行：ok 或 无法判定+缺项清单。"""
    if status.ok:
        return [f"【{name}】可判定"]
    lines = [f"【{name}】无法判定，缺："]
    lines += [f"  - {m}" for m in status.missing]
    return lines


def report_to_text(report: AdvisoryReport) -> str:
    """报告 → 确定性中文摘要（不调 LLM；数值原样呈现，未标定值明确标注）。"""
    lines: list[str] = ["=== 研判报告（讨论稿 §4.2 六项）==="]

    lines += _status_line("工艺参数（§4.2-1/2/3）", report.params.status)
    if report.params.params is not None and report.params.check is not None:
        verdict = "通过安全闸门" if report.params.check.within_limits else "未过安全闸门（拒绝下发）"
        lines.append(f"  闸门：{verdict}；违规 {len(report.params.check.violations)} 条")

    lines += _status_line("能耗（§4.2-5）", report.energy.status)
    if report.energy.estimate_kwh is not None:
        tag = "" if report.energy.is_calibrated else "（未标定参考值，不当真值）"
        lines.append(f"  当前参数能耗估计：{report.energy.estimate_kwh:.2f} kWh{tag}")
    if report.energy.plan_note:
        lines.append(f"  {report.energy.plan_note}")

    lines += _status_line("设备维保（§4.2-6）", report.maintenance.status)
    for comp in report.maintenance.components:
        if comp.service_due:
            tail = "→ 建议检修"
        elif comp.est_hours_to_service is not None:  # 精准维保时间（线性外推，随报告未标定标注）
            tail = f"（预计再运行 {comp.est_hours_to_service:.0f}h 达检修阈值）"
        else:
            tail = ""
        lines.append(f"  {comp.component}: 老化度 {comp.wear_frac:.0%} {tail}")

    lines += _status_line("摆炉排布（§4.2-4）", report.loading.status)
    if report.loading.sheets_per_bed is not None:
        loading = report.loading
        lines.append(f"  建议每床 {loading.sheets_per_bed} 片（{loading.layout}，间距 {loading.gap_mm}mm）")

    lines += _status_line("缺陷归因（§3.1.2）", report.attribution.status)
    for s in report.attribution.suspects:
        zone = f"（疑似分区 {s.zone}）" if s.zone is not None else ""
        lines.append(f"  疑似：{s.issue}{zone} ← {s.evidence}")

    lines.append("【光学特征插槽（§3.1.1）】" + (
        "未接入（光焦度/弓波图像源待现场）" if report.optical is None else "已随报告存档"
    ))
    return "\n".join(lines)

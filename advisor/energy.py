"""能耗线（讨论稿 §3.1.3 / §4.2-5）：当前参数能耗估计 + 最低能耗方案缺口说明。

代理公式（线性，系数在 config/energy.yaml，全 TODO(plant) 待现场电表回归标定）：
  energy_kwh ≈ [ base_kw + kw_per_zone_c·Σ_zone max(zone_temp − ref_temp_c, 0)
               + fan_kw_per_speed·convection_speed ] × heating_duration_s / 3600
ref 可注入合成系数（测试/演示）→ is_calibrated=False，不当真值下发（同 CCP 模式）。
"可优化降参空间"与"同质量最低能耗方案"（§3.1.3 后两项输出）同源：都依赖 A3-A6
硬约束真值（当前闸门全拦、无可行域）+ 质量模型——结构留接口、缺口如实写进
plan_note，不臆造（规则 > AI）；真值到位后先补降参空间（可行域内单参数余量），
再补完整方案（可行域搜索）。
"""

from __future__ import annotations

from pathlib import Path

from advisor.report import EnergyAdvice, SectionStatus
from schemas.process_params import ProcessParams

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "energy.yaml"
_COEF_KEYS = ("ref_temp_c", "kw_per_zone_c", "fan_kw_per_speed", "base_kw")

_PLAN_GAP_NOTE = (
    "可优化降参空间与最低能耗方案暂无法给出（两者同源）：需 A3-A6 硬约束真值"
    "（当前安全闸门对任何参数全拦，无可行域可搜索）与质量预测模型；"
    "系数与真值到位后在本模块先补降参空间、再补方案搜索器。"
)


def _is_todo(v) -> bool:
    """判断配置值是否缺失（None 或 TODO 占位）→ 按无法判定处理。"""
    return v is None or (isinstance(v, str) and v.strip().upper().startswith("TODO"))


def _load_config(path: Path | None = None) -> dict:
    """读 config/energy.yaml（每次调用按需读盘，改 yaml 即时生效）。"""
    import yaml

    p = Path(path) if path is not None else _CONFIG
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def estimate_energy(
    params: ProcessParams | None,
    config: dict | None = None,
    ref: dict | None = None,
) -> EnergyAdvice:
    """工艺参数 → 能耗建议。系数缺失 → cannot_determine 并列明缺口；ref 注入 → 未标定值。"""
    if params is None:
        return EnergyAdvice(
            status=SectionStatus(ok=False, missing=["工艺参数（上游未产参）"]),
            plan_note=_PLAN_GAP_NOTE,
        )

    cfg = config if config is not None else _load_config()
    coefs = ref if ref is not None else cfg
    is_calibrated = ref is None  # config 真值=已标定；注入参考=未标定

    missing = [f"config/energy.yaml: {k}（信息需求清单 E1）" for k in _COEF_KEYS if _is_todo(coefs.get(k))]
    if missing:
        return EnergyAdvice(status=SectionStatus(ok=False, missing=missing), plan_note=_PLAN_GAP_NOTE)

    ref_temp = float(coefs["ref_temp_c"])
    heat_kw = float(coefs["kw_per_zone_c"]) * sum(max(t - ref_temp, 0.0) for t in params.zone_temps)
    fan_kw = float(coefs["fan_kw_per_speed"]) * params.convection_speed
    total_kw = float(coefs["base_kw"]) + heat_kw + fan_kw
    kwh = total_kw * params.heating_duration_s / 3600.0

    return EnergyAdvice(
        status=SectionStatus(ok=True),
        estimate_kwh=kwh,
        is_calibrated=is_calibrated,
        plan_note=_PLAN_GAP_NOTE,
    )

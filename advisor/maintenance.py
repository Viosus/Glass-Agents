"""维保线（讨论稿 §3.1.4 / §4.2-6）：部件老化度 → 维保时点与检修项目。

老化度公式（讨论稿 §3.2"设备损耗计算算法"，权重/额定值在 config/maintenance.yaml，
全 TODO(plant) 待设备厂家手册+维修班组经验）：
  wear_frac(部件) = clip( w_hours·run_hours/rated_hours
                        + w_cycles·changeover_count/rated_cycles
                        + w_load·load_frac, 0, 1 )
wear_frac ≥ service_wear_threshold → 部件列入检修项目。
"精准维保时间"（§3.1.4）= 线性外推剩余小时：假设换产频次/负荷维持现状，老化只随
运行小时累积 → est = (threshold − wear)·rated_hours / w_hours；已达阈值取 0。
ref 可注入合成规则（测试/演示）→ is_calibrated=False，不当真值下发。
"""

from __future__ import annotations

from pathlib import Path

from advisor.report import ComponentWear, MaintenanceAdvice, SectionStatus
from schemas.inputs import EquipmentUsage

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "maintenance.yaml"


def _is_todo(v) -> bool:
    """判断配置值是否缺失（None 或 TODO 占位）→ 按无法判定处理。"""
    return v is None or (isinstance(v, str) and v.strip().upper().startswith("TODO"))


def _load_config(path: Path | None = None) -> dict:
    """读 config/maintenance.yaml（每次调用按需读盘，改 yaml 即时生效）。"""
    import yaml

    p = Path(path) if path is not None else _CONFIG
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _collect_missing(rules: dict, usage: EquipmentUsage) -> list[str]:
    """核对规则表与运行数据的缺口，返回 missing 清单（空=可判定）。"""
    missing: list[str] = []
    weights = rules.get("weights") or {}
    for k in ("hours", "cycles", "load"):
        if _is_todo(weights.get(k)):
            missing.append(f"config/maintenance.yaml: weights.{k}（信息需求清单 E2）")
    if _is_todo(rules.get("service_wear_threshold")):
        missing.append("config/maintenance.yaml: service_wear_threshold（信息需求清单 E2）")
    components = rules.get("components") or []
    if not components:
        missing.append("config/maintenance.yaml: components 部件清单（信息需求清单 E2）")
    for comp in components:
        for k in ("rated_hours", "rated_cycles"):
            if _is_todo(comp.get(k)):
                missing.append(f"config/maintenance.yaml: {comp.get('name', '?')}.{k}（信息需求清单 E2）")
    if usage.run_hours is None or usage.changeover_count is None or usage.load_frac is None:
        missing.append("设备运行数据 run_hours/changeover_count/load_frac（信息需求清单 B4）")
    return missing


def assess_maintenance(
    usage: EquipmentUsage | None,
    config: dict | None = None,
    ref: dict | None = None,
) -> MaintenanceAdvice:
    """设备运行数据 → 维保建议。规则/数据缺失 → cannot_determine；ref 注入 → 未标定值。"""
    if usage is None:
        return MaintenanceAdvice(
            status=SectionStatus(ok=False, missing=["设备运行数据 EquipmentUsage（信息需求清单 B4）"])
        )

    cfg = config if config is not None else _load_config()
    rules = ref if ref is not None else cfg
    is_calibrated = ref is None

    missing = _collect_missing(rules, usage)
    if missing:
        return MaintenanceAdvice(status=SectionStatus(ok=False, missing=missing))

    weights = rules["weights"]
    threshold = float(rules["service_wear_threshold"])
    components: list[ComponentWear] = []
    service_items: list[str] = []
    for comp in rules["components"]:
        wear = (
            float(weights["hours"]) * usage.run_hours / float(comp["rated_hours"])
            + float(weights["cycles"]) * usage.changeover_count / float(comp["rated_cycles"])
            + float(weights["load"]) * usage.load_frac
        )
        wear = min(max(wear, 0.0), 1.0)   # clip 到 [0,1]
        due = wear >= threshold
        # 精准维保时间：线性外推（换产/负荷不变，老化只随小时涨）；小时权重为 0 无法外推
        w_hours = float(weights["hours"])
        if due:
            est_hours: float | None = 0.0
        elif w_hours > 0.0:
            est_hours = (threshold - wear) * float(comp["rated_hours"]) / w_hours
        else:
            est_hours = None
        components.append(ComponentWear(
            component=str(comp["name"]), wear_frac=wear, service_due=due,
            est_hours_to_service=est_hours,
        ))
        if due:
            service_items.append(str(comp["name"]))

    return MaintenanceAdvice(
        status=SectionStatus(ok=True),
        components=components,
        service_items=service_items,
        is_calibrated=is_calibrated,
    )

"""钢化炉工艺参数硬约束校验（安全闸门）。

规则 > AI：任何产出工艺参数的代码路径，落地前必须过 ``validate()``。
阈值一律从 ``config/thresholds.yaml`` 读取，禁止把限值硬编码进逻辑。

铁律：阈值为 ``TODO(plant)`` 未填的项 → 判"无法判定 → 不通过"，并在
``violations`` 写明需补哪个 TODO(plant)，**绝不放行**。

真值来源：docs/03-hard-constraints.md（暂缺，多数阈值当前为 TODO(plant)）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"


def _is_todo(v) -> bool:
    """判断一个配置值是否缺失（None 或 TODO(plant) 占位）。"""
    return v is None or (isinstance(v, str) and v.strip().upper().startswith("TODO"))


def load_thresholds(path: Path | None = None) -> dict:
    """读取阈值配置。测试可注入自定义 path 或直接给 validate() 传 dict。"""
    p = Path(path) if path is not None else _CONFIG
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ParamSet:
    """一组钢化炉工艺参数。单位：温度 ℃、时长 s、厚度 mm。"""

    zone_temps: list[float]                 # ℃，按分区顺序
    zone_roles: list[str]                   # 与 zone_temps 等长，"center"/"edge"/...
    temp_upper: float
    temp_lower: float
    convection_speed: float
    convection_ratio_upper_lower: float
    oscillation_speed: float
    oscillation_amplitude: float
    heating_duration_s: float
    glass_type: str                         # ultra_clear | clear
    thickness_mm: float
    quality_mode: str                       # high_quality | high_efficiency
    # ---- 架构讨论稿 §4.2-2 补充字段（可选；约束规则待 docs/03，None 不拦）----
    convection_temp: float | None = None    # 对流风温（℃，温度类按项目约定不带后缀）
    fan_startup_logic: str | None = None    # 风机启动逻辑（结构待现场定义，暂自由文本）


@dataclass
class CheckResult:
    """硬约束校验结果：within_limits=全部通过；violations 列出人类可读原因（含规则编号）。"""

    within_limits: bool
    blow_up_risk: bool
    gradient_ok: bool
    violations: list[str] = field(default_factory=list)


def validate(p: ParamSet, prev: ParamSet | None = None, thresholds: dict | None = None) -> CheckResult:
    """对参数集 ``p`` 运行全部硬约束。``prev`` 为上一组参数（用于单次调温幅度）。

    ``thresholds`` 可注入（测试用）；默认读 config/thresholds.yaml。
    任一项不通过即 ``within_limits=False``，``violations`` 列出人类可读原因。
    """
    thr = thresholds if thresholds is not None else load_thresholds()
    violations: list[str] = []
    gradient_ok = True          # 仅反映 C1 梯度温控四项
    blow_up_risk = False

    grad = thr.get("gradient", {}) or {}

    # ---- C1-a: 中心区温度须高于边缘区 ----
    centers = [t for t, r in zip(p.zone_temps, p.zone_roles) if r == "center"]
    edges = [t for t, r in zip(p.zone_temps, p.zone_roles) if r == "edge"]
    if centers and edges and not (min(centers) > max(edges)):
        gradient_ok = False
        violations.append(
            f"C1: 存在 center 温度({min(centers)}) ≤ edge 温度({max(edges)})，须 min(center) > max(edge)"
        )

    # ---- C1-b: 上炉温须高于下炉温 ----
    if not (p.temp_upper > p.temp_lower):
        gradient_ok = False
        violations.append(f"C1: temp_upper({p.temp_upper}) 须 > temp_lower({p.temp_lower})")

    # ---- C1-c: 相邻分区温差 ----（相邻=索引相邻，二维邻接为 TODO(plant) 简化）
    adj = grad.get("adjacent_zone_max_delta_c")
    if _is_todo(adj):
        gradient_ok = False
        violations.append(
            "C1: 相邻分区温差上限缺失，需补 TODO(plant) "
            "gradient.adjacent_zone_max_delta_c（无法判定→不通过）"
        )
    else:
        assert adj is not None  # _is_todo 已排除 None
        for i in range(len(p.zone_temps) - 1):
            d = abs(p.zone_temps[i + 1] - p.zone_temps[i])
            if d > adj:
                gradient_ok = False
                violations.append(f"C1: 相邻分区温差 {d:.2f}℃ > 上限 {adj}℃（zone {i}-{i + 1}）")

    # ---- C1-d: 单次调温幅度（需 prev；prev=None 时跳过）----
    step = grad.get("single_step_max_delta_c")
    if prev is not None:
        if _is_todo(step):
            gradient_ok = False
            violations.append(
                "C1: 单次调温上限缺失，需补 TODO(plant) "
                "gradient.single_step_max_delta_c（无法判定→不通过）"
            )
        else:
            assert step is not None  # _is_todo 已排除 None
            n = min(len(p.zone_temps), len(prev.zone_temps))
            for i in range(n):
                d = abs(p.zone_temps[i] - prev.zone_temps[i])
                if d > step:
                    gradient_ok = False
                    violations.append(f"C1: 单次调温 {d:.2f}℃ > 上限 {step}℃（zone {i}）")

    # ---- C2: 厚度→时长 ----（阈值缺失 → 无法判定→不通过）
    if _is_todo(thr.get("thickness_duration")):
        violations.append("C2: 厚度→时长映射缺失，需补 TODO(plant) thickness_duration（无法判定→不通过）")

    # ---- C3: 对流风速/配比 ----
    if _is_todo(thr.get("convection")):
        violations.append("C3: 对流风速/配比区间缺失，需补 TODO(plant) convection（无法判定→不通过）")

    # ---- 安全红线 ----
    safety = thr.get("safety", {}) or {}
    if _is_todo(safety.get("max_gradient")):
        violations.append("安全: 温度梯度上限缺失，需补 TODO(plant) safety.max_gradient（无法判定→不通过）")
    if _is_todo(safety.get("blowup_rule")):
        blow_up_risk = True     # 无判定规则 → 无法排除 → 保守视为有风险
        violations.append("安全: 炸板判定规则缺失，需补 TODO(plant) safety.blowup_rule（无法排除→视为有风险）")

    within_limits = (not violations) and gradient_ok and (not blow_up_risk)
    return CheckResult(
        within_limits=within_limits,
        blow_up_risk=blow_up_risk,
        gradient_ok=gradient_ok,
        violations=violations,
    )

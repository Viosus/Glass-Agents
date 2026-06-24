"""评测靶子一键运行：对"未被阻塞"的判级/约束用例打印通过率。

覆盖：X0.95@6mm 判级、约束边界（合规/违规）。
不覆盖（待 docs/01 → TODO(plant)）：IsoT / CCP 判级表、X0.95 其余厚度。

用法：& "D:\\Glass Agents\\.venv\\Scripts\\python.exe" tools\\run_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 支持直接 `python tools/run_eval.py`

from tools.constraints import ParamSet, validate  # noqa: E402
from tools.metrics import grade  # noqa: E402

try:  # Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

FULL_THR = {
    "gradient": {"adjacent_zone_max_delta_c": 5, "single_step_max_delta_c": 3},
    "thickness_duration": {"6": [100, 300]},
    "convection": {"clear": [1.0, 2.0]},
    "safety": {"blowup_rule": "rule_v0", "max_gradient": 50},
}


def _mk(zone_temps, zone_roles=None) -> ParamSet:
    n = len(zone_temps)
    return ParamSet(
        zone_temps=list(zone_temps),
        zone_roles=zone_roles or ["center"] * n,
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


def _wl(zone_temps, zone_roles=None) -> bool:
    return validate(_mk(zone_temps, zone_roles), thresholds=FULL_THR).within_limits


def run() -> bool:
    cases = [
        ("X0.95=65 @6mm", grade(65.0, 6, "x0_95"), "A"),
        ("X0.95=70 @6mm 边界", grade(70.0, 6, "x0_95"), "A"),
        ("X0.95=95 @6mm 边界", grade(95.0, 6, "x0_95"), "B"),
        ("X0.95=95.01 @6mm", grade(95.01, 6, "x0_95"), "C"),
        ("相邻温差 5.0℃ 放行", _wl([100.0, 105.0]), True),
        ("相邻温差 5.01℃ 拦截", _wl([100.0, 105.01]), False),
        ("center≤edge 拦截", _wl([100.0, 100.0], ["center", "edge"]), False),
        ("默认 config(多 TODO) 拦截", validate(_mk([100.0, 102.0])).within_limits, False),
    ]

    passed = 0
    print(f"{'用例':<26}{'实际':<8}{'期望':<8}结果")
    print("-" * 52)
    for desc, actual, expected in cases:
        ok = actual == expected
        passed += int(ok)
        print(f"{desc:<26}{str(actual):<8}{str(expected):<8}{'PASS' if ok else 'FAIL'}")
    print("-" * 52)
    print(f"通过率: {passed}/{len(cases)} = {passed / len(cases) * 100:.0f}%")
    print("注：IsoT/CCP 判级表、X0.95 其余厚度待 docs/01 → TODO(plant)，未纳入。")
    return passed == len(cases)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)

"""命名 / 单位后缀自检 —— linter(pep8-naming) 管不了的项目语义特例兜底。

依据 CONVENTIONS.md：携带物理量的字段须带约定单位后缀；温差类配置键须以 `_c` 结尾；
分级表键须单位入名（`x0_95_nm`）。这些是项目特例，ruff 无法表达，故在此用单测兜底。
"""

import dataclasses
from pathlib import Path

import yaml

from tools.constraints import ParamSet

_CONFIG = Path(__file__).resolve().parent.parent / "config"


def test_paramset_physical_fields_carry_unit_suffix():
    # 明确携带物理量的字段须带单位后缀（温度场按约定恒为 ℃，不在此列）。
    required = {"heating_duration_s": "_s", "thickness_mm": "_mm"}
    field_names = {f.name for f in dataclasses.fields(ParamSet)}
    for name, suffix in required.items():
        assert name in field_names, f"ParamSet 缺字段 {name}"
        assert name.endswith(suffix), f"{name} 应以 {suffix} 结尾（单位入名）"


def test_temperature_delta_config_keys_end_with_c():
    thr = yaml.safe_load((_CONFIG / "thresholds.yaml").read_text(encoding="utf-8"))
    grad = thr["gradient"]
    for key in grad:
        if "delta" in key:  # 温差 / 调温幅度上限须以 _c 结尾
            assert key.endswith("_c"), f"温差键 {key} 应以 _c 结尾"
    assert "adjacent_zone_max_delta_c" in grad
    assert "single_step_max_delta_c" in grad


def test_grading_x0_95_uses_nm_key_and_thickness_mm():
    grading = yaml.safe_load((_CONFIG / "grading.yaml").read_text(encoding="utf-8"))
    assert "x0_95_nm" in grading, "X0.95 分级表键须为 x0_95_nm（单位入名）"
    rows = grading["x0_95_nm"]
    assert any(isinstance(r, dict) and "thickness_mm" in r for r in rows)

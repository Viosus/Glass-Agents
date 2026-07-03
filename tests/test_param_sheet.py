"""参数单单测：xlsx 读回断言（闸门/字段/分区/Δ）、violations 区块、对话壳导出意图。全程 tmp_path。"""

import pytest

from llm_roles.dialogue import DialogueState, load_dialogue_rules, respond, route_intent
from schemas.process_params import ProcessParams
from tools.constraints import CheckResult
from tools.param_sheet import build_rows, write_param_sheet

FULL_THR = {
    "gradient": {"adjacent_zone_max_delta_c": 5, "single_step_max_delta_c": 3},
    "thickness_duration": {"6": [100, 300]},
    "convection": {"clear": [1.0, 2.0]},
    "safety": {"blowup_rule": "rule_v0", "max_gradient": 50},
}


def make_params(**kw) -> ProcessParams:
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


def ok_check() -> CheckResult:
    return CheckResult(within_limits=True, blow_up_risk=False, gradient_ok=True, violations=[])


def bad_check() -> CheckResult:
    return CheckResult(
        within_limits=False, blow_up_risk=True, gradient_ok=False,
        violations=["C1: temp_upper(640.0) 须 > temp_lower(650.0)"],
    )


def _cells(rows):
    """行数据拍平成字符串集合，便于包含性断言。"""
    return {c for row in rows for c in row if c}


def test_rows_contain_fields_zones_delta():
    params = make_params(temp_upper=702.0, zone_temps=[103.0, 100.0])
    rows = build_rows(params, ok_check(), baseline=make_params(), meta={"furnace_id": "F1", "sample_id": "s9"})
    cells = _cells(rows)
    assert "上炉温" in cells and "702" in cells and "℃" in cells
    assert "F1" in cells and "s9" in cells
    assert "分区1温度(center)" in cells and "分区2温度(edge)" in cells
    assert "+2" in cells and "+1" in cells                    # Δ 列（temp_upper、分区1）
    assert "✅ 通过" in cells
    assert not any("禁止照此操作" in c for c in cells)


def test_rows_violations_block_when_gate_fails():
    rows = build_rows(make_params(temp_upper=640.0), bad_check())
    cells = _cells(rows)
    assert any("禁止照此操作" in c for c in cells)
    assert any("C1:" in c for c in cells)                     # violations 原样进单


def test_write_xlsx_roundtrip(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    out = write_param_sheet(
        make_params(), ok_check(), baseline=make_params(),
        meta={"furnace_id": "F1"}, out_path=tmp_path / "参数单.xlsx",
    )
    assert out.suffix == ".xlsx" and out.exists()
    ws = openpyxl.load_workbook(out).active
    texts = {str(c.value) for row in ws.iter_rows() for c in row if c.value is not None}
    assert "钢化炉工艺参数单" in texts and "上炉温" in texts and "✅ 通过" in texts


# --------------------------- 对话壳「导出参数单」 --------------------------- #
@pytest.fixture(scope="module")
def rules():
    return load_dialogue_rules()


def test_route_export_intent(rules):
    assert route_intent("导出参数单", rules) == "export_sheet"
    assert route_intent("出个表看看", rules) == "export_sheet"
    assert route_intent("检查当前参数", rules) == "param_check"  # 不被误抢


def test_dialogue_export_without_params(rules):
    state, reply = respond(DialogueState(), "导出参数单", rules=rules)
    assert "没有参数组" in reply


def test_dialogue_export_writes_file(rules, tmp_path):
    state = DialogueState(furnace_id="F1", params=make_params())
    state, reply = respond(
        state, "导出参数单", rules=rules, thresholds=FULL_THR, sheet_dir=tmp_path,
    )
    assert "参数单已导出" in reply and "已通过安全闸门" in reply
    files = list(tmp_path.glob("参数单_*"))
    assert len(files) == 1                                    # 落在指定目录，不碰 data/outbox

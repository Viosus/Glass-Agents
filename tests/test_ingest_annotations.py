"""导入器单测：模板往返、表头漂移拒、坏行拒带行号、去重/冲突、炉体身份注入。全程不碰真实 data/。"""

import csv

import pytest

from schemas.archive import load_all
from schemas.furnace import load_furnace_registry
from tools.ingest_annotations import ingest
from tools.make_annotation_template import COLUMNS, write_csv

HEADER = [c[0] for c in COLUMNS]

FURNACES_YAML = """\
furnaces:
  - furnace_id: F1
    zone_count: 4
    zone_layout: "2x2"
    fan_count: TODO(plant)
    nameplate: {"model": "炉型X"}
  - furnace_id: TODO(plant)
    zone_count: 8
"""


def make_row(**over) -> dict[str, str]:
    """一行结构合法的标注数据（字符串形态，模拟老师傅填表）。"""
    row = {
        "sample_id": "r001",
        "created_at": "2026-07-02T09:30",
        "operator_id": "S01",
        "source": "line1/早班",
        "thickness_mm": "6",
        "glass_type": "clear",
        "quality_mode": "high_quality",
        "x0_95_nm": "82",
        "iso_t_pct": "",
        "ccp_value": "",
        "ccp_is_calibrated": "FALSE",
        "stress_image_path": "",
        "stress_image_sha256": "",
        "condition_note": "炉温稳定",
        "baseline_zone_temps_c": "102;100",
        "baseline_zone_roles": "center;edge",
        "baseline_temp_upper_c": "700",
        "baseline_temp_lower_c": "650",
        "baseline_convection_speed": "1.0",
        "baseline_convection_ratio_upper_lower": "1.0",
        "baseline_oscillation_speed": "1.0",
        "baseline_oscillation_amplitude": "1.0",
        "baseline_heating_duration_s": "200",
        "final_zone_temps_c": "103;100",
        "final_zone_roles": "center;edge",
        "final_temp_upper_c": "702",
        "final_temp_lower_c": "650",
        "final_convection_speed": "1.2",
        "final_convection_ratio_upper_lower": "1.0",
        "final_oscillation_speed": "1.0",
        "final_oscillation_amplitude": "1.0",
        "final_heating_duration_s": "210",
        "expert_quality_grade": "A",
        "rationale": "边缘偏低，升上炉温+延时",
        "cause_tag": "",
        "is_ground_truth": "TRUE",
        "repeat_group_id": "",
        "measured_quality_grade": "",
        "measured_energy_kwh": "",
    }
    row.update(over)
    return row


def write_filled_csv(path, rows):
    """模板（含示例行）+ 真数据行 → 一张“填好的表”。"""
    write_csv(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow([row[col] for col in HEADER])


@pytest.fixture
def furnaces_yaml(tmp_path):
    """临时炉体登记表 fixture（含一条 TODO(plant) 身份的坏条目）。"""
    p = tmp_path / "furnaces.yaml"
    p.write_text(FURNACES_YAML, encoding="utf-8")
    return p


# --------------------------- 炉体登记表 --------------------------- #
def test_registry_loads_and_skips_todo(furnaces_yaml, tmp_path):
    reg = load_furnace_registry(furnaces_yaml)
    assert set(reg) == {"F1"}                       # TODO(plant) 身份的条目不登记
    f1 = reg["F1"]
    assert f1.zone_count == 4 and f1.zone_layout == "2x2"
    assert f1.fan_count is None                     # TODO(plant) 字段按缺失处理，不猜
    assert f1.nameplate == {"model": "炉型X"}
    assert load_furnace_registry(tmp_path / "不存在.yaml") == {}


# --------------------------- 模板说明行契约 --------------------------- #
def test_template_label_row_marks_who_fills(tmp_path):
    """第 2 行中文说明行：师傅列标★、系统列标勿动、outcome 标出炉后补；且被导入器当示例行跳过。"""
    path = tmp_path / "t.csv"
    write_csv(path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    labels = dict(zip(rows[0], rows[1]))
    assert labels["sample_id"].startswith("示例-")          # 借示例行机制跳过，导入零改动
    assert labels["final_temp_upper_c"].startswith("★填这里")
    assert labels["baseline_temp_upper_c"].startswith("勿动")
    assert labels["measured_energy_kwh"].startswith("出炉后补")
    assert len(rows[1]) == len(HEADER)                      # 说明行与列一一对应（_ZH 全覆盖）


def test_template_xlsx_outline_declared(tmp_path):
    """xlsx 结构回归：列分组必须带 sheet 级 outlineLevelCol 声明，颜色须 FF 不透明。

    缺 outlineLevelCol 时 Excel/WPS 按"文件损坏"拒载（2026-07-03 现场踩坑）；
    00-alpha 颜色 WPS 渲染成透明。openpyxl 都不会自动写对，靠本测试锚住。
    """
    openpyxl = pytest.importorskip("openpyxl")
    from tools.make_annotation_template import write_xlsx

    path = tmp_path / "t.xlsx"
    assert write_xlsx(path)
    ws = openpyxl.load_workbook(path)["标注表"]
    assert ws.sheet_format.outlineLevelCol == 1
    master_col = HEADER.index("final_zone_temps_c") + 1
    assert ws.cell(1, master_col).fill.start_color.rgb == "FFFFF2CC"


# --------------------------- 模板往返 + 身份注入 --------------------------- #
def test_roundtrip_injects_furnace_identity(tmp_path, furnaces_yaml):
    csv_path = tmp_path / "filled.csv"
    write_filled_csv(csv_path, [make_row(), make_row(sample_id="r002", final_temp_upper_c="701")])
    out = tmp_path / "archive"

    report = ingest(csv_path, "F1", out, furnaces_yaml)
    assert report.accepted == 2
    assert report.skipped_examples == 3             # 模板自带：中文说明行 + 两行示例（均"示例-"前缀）
    assert not report.rejected

    loaded = {s.sample_id: s for s in load_all(out)}
    s = loaded["r001"]
    assert s.furnace_id == "F1"
    assert s.furnace_config is not None and s.furnace_config.zone_count == 4
    assert s.baseline_params is not None
    assert s.params.temp_upper - s.baseline_params.temp_upper == pytest.approx(2.0)  # Δ 可算
    assert s.constraint is not None                 # 闸门结果已归档（只记录不拒收）
    assert s.operator_id == "S01" and s.is_ground_truth is True


def test_all_glass_types_ingest_end_to_end(tmp_path, furnaces_yaml):
    """7 品类的填好表都必须整行进得来（2026-08-23 修复的核心回归）。

    修复前 glass_type 只认 ultra_clear/clear，Low-E/彩釉/压花/镀膜/其他 的行会被
    ProcessParams 的 Literal 拒掉 → 整行 rejected → 这几类炉次事实上无法回流。
    """
    from schemas.process_params import GLASS_TYPES

    rows = [make_row(sample_id=f"g{i:03d}", glass_type=gt) for i, gt in enumerate(GLASS_TYPES)]
    csv_path = tmp_path / "filled.csv"
    write_filled_csv(csv_path, rows)

    report = ingest(csv_path, "F1", tmp_path / "archive", furnaces_yaml)
    assert report.accepted == len(GLASS_TYPES), f"应全数接受，实际拒收：{report.rejected}"
    assert not report.rejected
    got = {s.glass_type for s in report.samples}
    assert got == set(GLASS_TYPES)


def test_unregistered_furnace_warns_but_ingests(tmp_path, furnaces_yaml):
    csv_path = tmp_path / "filled.csv"
    write_filled_csv(csv_path, [make_row()])
    report = ingest(csv_path, "F9", tmp_path / "archive", furnaces_yaml)
    assert report.accepted == 1
    assert any("未在" in w for w in report.warnings)
    assert report.samples[0].furnace_config is None  # 未登记 → 不构造假配置


# --------------------------- 表头漂移 / 坏行 / 缺值 --------------------------- #
def test_header_drift_rejected(tmp_path, furnaces_yaml):
    csv_path = tmp_path / "drifted.csv"
    bad_header = HEADER[:-1] + ["私自加的列"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow(bad_header)
    with pytest.raises(ValueError, match="表头与模板不一致"):
        ingest(csv_path, "F1", tmp_path / "archive", furnaces_yaml)


def test_bad_row_rejected_with_row_number(tmp_path, furnaces_yaml):
    csv_path = tmp_path / "filled.csv"
    write_filled_csv(
        csv_path,
        [make_row(), make_row(sample_id="r002", final_temp_upper_c="abc")],  # 第二行坏数
    )
    report = ingest(csv_path, "F1", tmp_path / "archive", furnaces_yaml)
    assert report.accepted == 1
    assert len(report.rejected) == 1
    row_no, reason = report.rejected[0]
    assert row_no == 6                              # 表头1 + 说明行2 + 示例3-4 + 好行5 → 坏行=6
    assert "final_temp_upper_c" in reason


def test_todo_cell_treated_as_missing(tmp_path, furnaces_yaml):
    csv_path = tmp_path / "filled.csv"
    write_filled_csv(csv_path, [make_row(iso_t_pct="TODO(plant)")])
    report = ingest(csv_path, "F1", tmp_path / "archive", furnaces_yaml)
    assert report.accepted == 1
    assert report.samples[0].metrics.iso_t_pct is None  # 缺值不猜


# --------------------------- 去重 / 冲突 / dry-run --------------------------- #
def test_duplicate_and_conflict_handling(tmp_path, furnaces_yaml):
    csv_path = tmp_path / "filled.csv"
    write_filled_csv(csv_path, [make_row()])
    out = tmp_path / "archive"

    assert ingest(csv_path, "F1", out, furnaces_yaml).accepted == 1
    rep2 = ingest(csv_path, "F1", out, furnaces_yaml)               # 原样重导 → 幂等跳过
    assert rep2.accepted == 0 and rep2.duplicates == 1

    csv2 = tmp_path / "conflict.csv"
    write_filled_csv(csv2, [make_row(rationale="改了理由")])          # 同 id 不同内容
    rep3 = ingest(csv2, "F1", out, furnaces_yaml)
    assert rep3.accepted == 0 and any("内容不同" in r for _, r in rep3.rejected)

    rep4 = ingest(csv2, "F1", out, furnaces_yaml, force=True)        # --force 才覆盖
    assert rep4.accepted == 1


def test_dry_run_writes_nothing(tmp_path, furnaces_yaml):
    """dry-run 只报告不写盘。"""
    csv_path = tmp_path / "filled.csv"
    write_filled_csv(csv_path, [make_row()])
    out = tmp_path / "archive"
    report = ingest(csv_path, "F1", out, furnaces_yaml, dry_run=True)
    assert report.accepted == 1
    assert not out.exists() or not list(out.glob("*.json"))


def test_in_table_duplicate_id_rejected(tmp_path, furnaces_yaml):
    """同一张表内重复 sample_id：后一行拒绝。"""
    csv_path = tmp_path / "filled.csv"
    write_filled_csv(csv_path, [make_row(), make_row(rationale="另一行同 id")])
    report = ingest(csv_path, "F1", tmp_path / "archive", furnaces_yaml)
    assert report.accepted == 1
    assert any("表内重复" in r for _, r in report.rejected)

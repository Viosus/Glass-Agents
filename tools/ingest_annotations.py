"""标注表导入器：老师傅填好的 CSV → 校验 → ArchiveSample 落库 data/archive/ + 按桶看板。

设计要点（P2 三件套之一）：
- 炉体身份用 --furnace-id 注入（标注表 39 列已冻结不能改；一张表=一台炉）；
  炉子若已在 config/furnaces.yaml 登记，则把配置快照冻结进样本。
- 表头漂移守卫：与 tools.make_annotation_template.COLUMNS（单一来源）比对，不一致直接拒。
- 结构校验走 ProcessParams（写入即校验）；随后跑 tools.constraints.validate 把结果**只记录不拒收**
  ——老师傅实操参数是事实；违规多因 config 缺值 TODO(plant) 未填，如实归档。
- 缺值不猜：空串/TODO(plant) 单元格 → None；坏行整行拒绝并报告行号，绝不猜数。
- 去重：同 sample_id 同内容跳过；同 sample_id 不同内容拒绝（--force 才覆盖）。

用法：& "D:\\Glass Agents\\.venv\\Scripts\\python.exe" tools\\ingest_annotations.py ^
        data\\annotation\\填好的表.csv --furnace-id F1 [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 支持直接 `python tools/ingest_annotations.py`

from schemas.archive import ArchiveSample, ConstraintSummary, MetricRecord, write_sample  # noqa: E402
from schemas.bucketing import bucket_table  # noqa: E402
from schemas.furnace import FurnaceConfig, load_furnace_registry  # noqa: E402
from schemas.process_params import ProcessParams  # noqa: E402
from tools.constraints import validate  # noqa: E402
from tools.make_annotation_template import COLUMNS  # noqa: E402

_EXAMPLE_PREFIX = "示例-"
_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "archive"

# 表内 9 个参数列名 → ProcessParams 字段（baseline_/final_ 前缀共用此映射）
_PARAM_COLS: dict[str, str] = {
    "zone_temps_c": "zone_temps",
    "zone_roles": "zone_roles",
    "temp_upper_c": "temp_upper",
    "temp_lower_c": "temp_lower",
    "convection_speed": "convection_speed",
    "convection_ratio_upper_lower": "convection_ratio_upper_lower",
    "oscillation_speed": "oscillation_speed",
    "oscillation_amplitude": "oscillation_amplitude",
    "heating_duration_s": "heating_duration_s",
}


@dataclass
class IngestReport:
    """导入报告：入库/跳过/拒绝明细（行号 + 原因），供看板与人工复核。"""

    accepted: int = 0
    skipped_examples: int = 0
    duplicates: int = 0
    rejected: list[tuple[int, str]] = field(default_factory=list)   # (CSV 行号, 原因)
    warnings: list[str] = field(default_factory=list)
    samples: list[ArchiveSample] = field(default_factory=list)      # 本次入库（或 dry-run 通过）的样本


def _clean(cell: str | None) -> str | None:
    """单元格清洗：空串 / TODO(plant) 占位 → None（缺值不猜）。"""
    if cell is None:
        return None
    s = cell.strip()
    if not s or s.startswith("TODO(plant)"):
        return None
    return s


def _req(row: dict[str, str], col: str) -> str:
    """必填单元格：缺失即抛 ValueError（整行拒绝）。"""
    v = _clean(row.get(col))
    if v is None:
        raise ValueError(f"必填列 {col} 为空")
    return v


def _to_float(text: str, col: str) -> float:
    """字符串转 float，失败抛带列名的 ValueError。"""
    try:
        return float(text)
    except ValueError as e:
        raise ValueError(f"列 {col} 不是数值: {text!r}") from e


def _to_bool(text: str | None, *, default: bool, col: str) -> bool:
    """TRUE/FALSE → bool；空取默认；其他值拒绝。"""
    if text is None:
        return default
    t = text.strip().upper()
    if t == "TRUE":
        return True
    if t == "FALSE":
        return False
    raise ValueError(f"列 {col} 只接受 TRUE/FALSE: {text!r}")


def _split_semicolon(text: str) -> list[str]:
    """分号切列表（模板约定），去空段。"""
    return [seg.strip() for seg in text.split(";") if seg.strip()]


def _parse_params(row: dict[str, str], prefix: str, spec: dict[str, str | float]) -> ProcessParams:
    """按前缀（baseline_/final_）从行中解析一组 ProcessParams（结构校验交给 pydantic）。"""
    data: dict[str, object] = {}
    for suffix, target in _PARAM_COLS.items():
        col = prefix + suffix
        raw = _req(row, col)
        if target == "zone_temps":
            data[target] = [_to_float(seg, col) for seg in _split_semicolon(raw)]
        elif target == "zone_roles":
            data[target] = _split_semicolon(raw)
        else:
            data[target] = _to_float(raw, col)
    data.update(spec)  # 规格三键（thickness_mm/glass_type/quality_mode）来自 B 组列
    return ProcessParams(**data)  # type: ignore[arg-type]


def parse_row(
    row: dict[str, str],
    furnace_id: str,
    furnace_config: FurnaceConfig | None,
) -> tuple[ArchiveSample, list[str]]:
    """单行 → ArchiveSample（写入即校验）。返回 (样本, 行内警告)；坏行抛 ValueError。"""
    row_warnings: list[str] = []
    spec: dict[str, str | float] = {
        "thickness_mm": _to_float(_req(row, "thickness_mm"), "thickness_mm"),
        "glass_type": _req(row, "glass_type"),
        "quality_mode": _req(row, "quality_mode"),
    }
    baseline = _parse_params(row, "baseline_", spec)
    final = _parse_params(row, "final_", spec)

    # 硬约束闸门结果只记录不拒收（老师傅实操是事实；违规多因 config TODO 未填）
    check = validate(final.to_param_set())
    constraint = ConstraintSummary(
        within_limits=check.within_limits,
        blow_up_risk=check.blow_up_risk,
        gradient_ok=check.gradient_ok,
        violations=check.violations,
    )

    x0_95 = _clean(row.get("x0_95_nm"))
    iso_t = _clean(row.get("iso_t_pct"))
    ccp = _clean(row.get("ccp_value"))
    metrics = MetricRecord(
        x0_95_nm=_to_float(x0_95, "x0_95_nm") if x0_95 is not None else None,
        iso_t_pct=_to_float(iso_t, "iso_t_pct") if iso_t is not None else None,
        ccp_value=_to_float(ccp, "ccp_value") if ccp is not None else None,
        ccp_is_calibrated=_to_bool(_clean(row.get("ccp_is_calibrated")), default=False, col="ccp_is_calibrated"),
    )

    # 表内只有图像路径+哈希，无尺寸/尺度列 → 无法构造完整 ImageRef，如实置 None 并警告
    img_path = _clean(row.get("stress_image_path"))
    if img_path is not None:
        row_warnings.append(f"stress_image_path 有值但表内无尺寸/尺度列，图像引用置空待补: {img_path}")

    measured_energy = _clean(row.get("measured_energy_kwh"))
    sample = ArchiveSample(
        sample_id=_req(row, "sample_id"),
        created_at=datetime.fromisoformat(_req(row, "created_at")),
        source=_req(row, "source"),
        furnace_id=furnace_id,
        furnace_config=furnace_config,
        thickness_mm=float(spec["thickness_mm"]),
        glass_type=spec["glass_type"],  # type: ignore[arg-type]
        quality_mode=spec["quality_mode"],  # type: ignore[arg-type]
        is_ground_truth=_to_bool(_clean(row.get("is_ground_truth")), default=True, col="is_ground_truth"),
        params=final,
        baseline_params=baseline,
        metrics=metrics,
        constraint=constraint,
        expert_quality_grade=_clean(row.get("expert_quality_grade")),  # type: ignore[arg-type]
        cause_tag=_clean(row.get("cause_tag")),
        operator_id=_clean(row.get("operator_id")),
        repeat_group_id=_clean(row.get("repeat_group_id")),
        condition_note=_clean(row.get("condition_note")),
        measured_quality_grade=_clean(row.get("measured_quality_grade")),  # type: ignore[arg-type]
        measured_energy_kwh=_to_float(measured_energy, "measured_energy_kwh") if measured_energy else None,
        rationale=_clean(row.get("rationale")),
    )
    return sample, row_warnings


def _check_header(header: list[str]) -> None:
    """表头漂移守卫：与模板 COLUMNS 单一来源比对，不一致直接拒。"""
    expected = [c[0] for c in COLUMNS]
    if header != expected:
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        raise ValueError(f"表头与模板不一致（缺: {missing} 多: {extra}）——请勿改列，用最新模板")


def ingest(
    csv_path: Path,
    furnace_id: str,
    out_root: Path = _DEFAULT_OUT,
    furnaces_path: Path | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> IngestReport:
    """整表导入：逐行解析校验 → 去重 → 落库。返回 IngestReport（dry_run 只报告不写盘）。"""
    report = IngestReport()
    registry = load_furnace_registry(furnaces_path)
    furnace_config = registry.get(furnace_id)
    if furnace_config is None:
        report.warnings.append(
            f"炉 {furnace_id} 未在 config/furnaces.yaml 登记（或配置全 TODO），furnace_config 置空"
        )

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as e:
            raise ValueError("CSV 为空，无表头") from e
        _check_header(header)

        seen_ids: set[str] = set()
        for row_no, cells in enumerate(reader, start=2):  # 行号按 Excel 习惯从 2 起（1=表头）
            row = dict(zip(header, cells))
            sid = (row.get("sample_id") or "").strip()
            if sid.startswith(_EXAMPLE_PREFIX):
                report.skipped_examples += 1
                continue
            try:
                if sid and sid in seen_ids:
                    raise ValueError(f"表内重复 sample_id: {sid}")
                sample, row_warnings = parse_row(row, furnace_id, furnace_config)
            except (ValueError, TypeError) as e:
                report.rejected.append((row_no, str(e)))
                continue
            seen_ids.add(sample.sample_id)
            report.warnings.extend(f"行{row_no}: {w}" for w in row_warnings)

            # 与库中已有同名样本比对（同内容跳过；异内容拒绝，--force 才覆盖）
            new_json = sample.model_dump_json(indent=2)
            existing = Path(out_root) / f"{sample.sample_id}.json"
            if existing.exists():
                old_sha = hashlib.sha256(existing.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                new_sha = hashlib.sha256(new_json.encode("utf-8")).hexdigest()
                if old_sha == new_sha:
                    report.duplicates += 1
                    continue
                if not force:
                    reason = f"库中已有同名样本且内容不同: {sample.sample_id}（--force 才覆盖）"
                    report.rejected.append((row_no, reason))
                    continue

            if not dry_run:
                write_sample(sample, Path(out_root))
            report.accepted += 1
            report.samples.append(sample)
    return report


def render_report(report: IngestReport, *, dry_run: bool) -> str:
    """报告 → 人读文本：计数 + 拒绝明细 + 按桶/按炉看板。"""
    lines = [
        f"{'[dry-run] ' if dry_run else ''}入库 {report.accepted}  示例行跳过 {report.skipped_examples}  "
        f"重复跳过 {report.duplicates}  拒绝 {len(report.rejected)}"
    ]
    for row_no, reason in report.rejected:
        lines.append(f"  ✗ 行{row_no}: {reason}")
    for w in report.warnings:
        lines.append(f"  ⚠ {w}")
    if report.samples:
        lines.append("按桶覆盖（本次入库部分）:")
        for bucket, total, gt in bucket_table(report.samples):
            lines.append(f"  {bucket}: 共 {total}，真值 {gt}")
        by_furnace: dict[str, int] = {}
        for s in report.samples:
            by_furnace[s.furnace_id] = by_furnace.get(s.furnace_id, 0) + 1
        lines.append("按炉计数: " + ", ".join(f"{k}={v}" for k, v in sorted(by_furnace.items())))
    return "\n".join(lines)


def main() -> int:
    """CLI 入口：解析参数 → ingest → 打印报告。拒绝行不影响其余行入库。"""
    ap = argparse.ArgumentParser(description="标注表 CSV → ArchiveSample 落库（缺值不猜，坏行拒绝）")
    ap.add_argument("csv_path", type=Path, help="老师傅填好的标注表 CSV（utf-8-sig）")
    ap.add_argument("--furnace-id", required=True, help="本表对应的炉子标识（一张表=一台炉）")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="归档目录（默认 data/archive）")
    ap.add_argument("--furnaces", type=Path, default=None, help="炉体登记表（默认 config/furnaces.yaml）")
    ap.add_argument("--dry-run", action="store_true", help="只解析校验并报告，不写盘")
    ap.add_argument("--force", action="store_true", help="同名样本内容不同时允许覆盖")
    args = ap.parse_args()

    report = ingest(
        args.csv_path,
        args.furnace_id,
        args.out,
        args.furnaces,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(render_report(report, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    sys.exit(main())

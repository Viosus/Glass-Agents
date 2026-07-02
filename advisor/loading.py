"""摆炉线（讨论稿 §4.2-4）：装载排布建议——每床片数与网格排布。

规则表在 config/loading.yaml（炉床有效尺寸 + 片间最小间距，全 TODO(plant) 待现场量测
与工艺规定）。真值到位后 grid 策略即生效：
  每行片数 = floor((bed_width + gap) / (glass_width + gap))，行数沿传送方向同理。
更优排布（旋转/混排/批次顺序）待现场规则，不臆造。
"""

from __future__ import annotations

import math
from pathlib import Path

from advisor.report import LoadingAdvice, SectionStatus

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "loading.yaml"
_RULE_KEYS = ("bed_length_mm", "bed_width_mm", "min_gap_mm")


def _is_todo(v) -> bool:
    """判断配置值是否缺失（None 或 TODO 占位）→ 按无法判定处理。"""
    return v is None or (isinstance(v, str) and v.strip().upper().startswith("TODO"))


def _load_config(path: Path | None = None) -> dict:
    """读 config/loading.yaml（每次调用按需读盘，改 yaml 即时生效）。"""
    import yaml

    p = Path(path) if path is not None else _CONFIG
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def suggest_loading(
    glass_length_mm: float | None,
    glass_width_mm: float | None,
    config: dict | None = None,
) -> LoadingAdvice:
    """玻璃尺寸 → 摆炉建议。规则/尺寸缺失 → cannot_determine 并列明缺口。"""
    missing: list[str] = []
    if glass_length_mm is None or glass_width_mm is None:
        missing.append("玻璃板面尺寸 glass_length_mm/glass_width_mm（订单/工艺配方）")

    cfg = config if config is not None else _load_config()
    missing += [f"config/loading.yaml: {k}（信息需求清单 E3）" for k in _RULE_KEYS if _is_todo(cfg.get(k))]
    if missing:
        return LoadingAdvice(status=SectionStatus(ok=False, missing=missing))

    gap = float(cfg["min_gap_mm"])
    bed_len, bed_wid = float(cfg["bed_length_mm"]), float(cfg["bed_width_mm"])
    assert glass_length_mm is not None and glass_width_mm is not None  # missing 已排除
    n_rows = math.floor((bed_len + gap) / (glass_length_mm + gap))    # 沿传送方向
    n_cols = math.floor((bed_wid + gap) / (glass_width_mm + gap))
    if n_rows < 1 or n_cols < 1:
        return LoadingAdvice(
            status=SectionStatus(ok=False, missing=["玻璃尺寸超出炉床有效尺寸，无法排布（核对输入）"])
        )
    return LoadingAdvice(
        status=SectionStatus(ok=True),
        sheets_per_bed=n_rows * n_cols,
        layout=f"{n_rows}行×{n_cols}列",
        gap_mm=gap,
    )

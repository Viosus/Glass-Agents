"""应力斑分布打分包：按斑的深浅与位置给玻璃打分（淡且均匀 → 高分）。

公共 API：
- score_fringe_distribution(image, config=None) → FringeScoreResult（单片）
- score_sheets(image, config=None) → SheetsScoreResult（整床多片：切分+逐片打分）
- 参数全部在 config/fringe_scoring.yaml，调参指南见 docs/应力斑分布打分_需求与设计.md §4。
"""

from fringe_scoring.border import BorderBands, band_widths_px, border_mask
from fringe_scoring.indicators import (
    INDICATOR_KEYS,
    SheetIndicators,
    compute_sheet_indicators,
    score_from_raw,
    select_refs,
)
from fringe_scoring.quad import detect_glass_quad, order_corners, quad_from_hull, warp_to_rect
from fringe_scoring.score import FringeScoreResult, load_config, score_fringe_distribution
from fringe_scoring.sheets import SheetsScoreResult, detect_sheet_quads, score_sheets

__all__ = [
    "BorderBands",
    "band_widths_px",
    "border_mask",
    "detect_glass_quad",
    "order_corners",
    "quad_from_hull",
    "warp_to_rect",
    "FringeScoreResult",
    "load_config",
    "score_fringe_distribution",
    "SheetsScoreResult",
    "detect_sheet_quads",
    "score_sheets",
    "INDICATOR_KEYS",
    "SheetIndicators",
    "compute_sheet_indicators",
    "score_from_raw",
    "select_refs",
]

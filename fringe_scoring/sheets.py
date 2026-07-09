"""整床多片切分：一张照片含多块玻璃 → 检出每片四角 → 逐片矫正打分 → 整床汇总。

前提（2026-07 真实产线照片实测确认）：
- 暗场偏光成像：无玻璃处为暗背景，玻璃片边缘因边部应力必然偏亮；
- 暗背景占比 ≥ bg_quantile（默认 25%）→ 全图低分位数即背景水平（边界一圈不可靠：
  实测照片首末行各有 1px 亮线 artifact，导出软件所加，检测前按 edge_trim_frac 裁掉外圈）；
- 前景阈值 = 背景 + max(z 项, 相对亮参考项)：z 项用暗半侧稳健尺度；相对项防
  背景过净（MAD≈0）时阈值塌陷。
流程：裁边 → 阈值出前景 → 闭运算连补断点 → 外轮廓连通域 → 面积过滤
→ 逐片凸包拟 4 角（复用 quad.quad_from_hull）→ 阅读序排序
→ 逐片 score_fringe_distribution(quad_corners=...) 复用既有矫正打分管线。
参数由 config/fringe_scoring.yaml 的 sheets 段驱动，限值不硬编码。
失败 = 抛 ValueError（检不出片 / 片数超上限 / 缺配置段），绝不返回瞎猜的分数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fringe_scoring.indicators import SheetIndicators, compute_sheet_indicators
from fringe_scoring.quad import quad_from_hull
from fringe_scoring.score import (
    FringeScoreResult,
    compute_pipeline,
    load_config,
    score_from_pipeline,
)
from fringe_scoring.segment import robust_scale


def _sort_reading_order(quads: list[np.ndarray]) -> list[np.ndarray]:
    """按阅读序排序：先按行（质心 y 聚簇，簇间距 > 片高中位数一半即换行），行内按 x。"""
    centers = np.array([q.mean(axis=0) for q in quads])  # (n, 2) 每片质心 (x, y)
    heights = np.array([float(q[:, 1].max() - q[:, 1].min()) for q in quads])
    row_tol = 0.5 * float(np.median(heights))

    by_cy = np.argsort(centers[:, 1], kind="stable")
    row_id = np.zeros(len(quads), dtype=int)
    current_row, last_cy = 0, float(centers[by_cy[0], 1])
    for i in by_cy:
        cy = float(centers[i, 1])
        if cy - last_cy > row_tol:
            current_row += 1
        row_id[i] = current_row
        last_cy = cy

    order = np.lexsort((centers[:, 0], row_id))  # 主键=行号，次键=质心 x
    return [quads[int(i)] for i in order]


def detect_sheet_quads(image: np.ndarray, config: dict | None = None) -> list[np.ndarray]:
    """整床照片 → 每片玻璃的 (4,2) 角点列表（原图坐标，阅读序）。

    检测异常一律抛 ValueError：一片都检不出 / 片数超 max_sheets / 某片拟不出 4 角。
    """
    import cv2

    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("detect_sheet_quads: 需要二维灰度图像数组")
    cfg = config if config is not None else load_config()
    sheets_cfg = cfg.get("sheets")
    if not sheets_cfg:
        raise ValueError("detect_sheet_quads: 配置缺少 sheets 段（多片切分参数）")

    rows, cols = arr.shape
    # 裁掉外圈条带：实测照片首末行有 1px 亮线 artifact；至少裁 1px
    trim = max(1, int(round(float(sheets_cfg["edge_trim_frac"]) * min(rows, cols))))
    if rows - 2 * trim < 8 or cols - 2 * trim < 8:
        raise ValueError("detect_sheet_quads: 图像过小（裁边后不足 8×8 像素）")
    inner = arr[trim: rows - trim, trim: cols - trim]

    # 背景水平 = 低分位数（前提：背景占比 ≥ bg_quantile）；尺度只在暗侧估计（不被玻璃污染）
    bg_level = float(np.percentile(inner, 100.0 * float(sheets_cfg["bg_quantile"])))
    bg_scale = robust_scale(inner[inner <= bg_level])
    p_bright = float(np.percentile(inner, 100.0 * float(sheets_cfg["bright_ref_quantile"])))
    if p_bright <= bg_level:
        raise ValueError("detect_sheet_quads: 图像近恒定，未检出任何玻璃片")
    threshold = bg_level + max(
        float(sheets_cfg["fg_min_z"]) * bg_scale,
        float(sheets_cfg["fg_min_rel"]) * (p_bright - bg_level),
    )
    foreground = inner > threshold

    close_px = max(3, int(round(float(sheets_cfg["close_frac"]) * min(rows, cols))))
    close_px += (close_px + 1) % 2  # 结构元取奇数边长
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px))
    closed = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(sheets_cfg["min_area_frac"]) * inner.shape[0] * inner.shape[1]
    min_side = float(sheets_cfg["min_side_frac"]) * min(inner.shape)
    epsilon_frac = float(sheets_cfg["poly_epsilon_frac"])
    quads = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        rect_w, rect_h = cv2.minAreaRect(c)[1]
        if min(rect_w, rect_h) < min_side:
            continue  # 细长条（亮线纹/反光）不可能是玻璃片，按干扰剔除
        quads.append(quad_from_hull(cv2.convexHull(c), epsilon_frac) + float(trim))  # 还原到原图坐标
    if not quads:
        raise ValueError(
            "detect_sheet_quads: 未检出任何玻璃片（前景为空或全部小于 min_area_frac），"
            "确认为暗背景整床照片，或调低 fg_min_z / min_area_frac"
        )
    max_sheets = int(sheets_cfg["max_sheets"])
    if len(quads) > max_sheets:
        raise ValueError(
            f"detect_sheet_quads: 检出 {len(quads)} 片 > max_sheets={max_sheets}，"
            "疑似过分割，检测异常拒绝打分（可调大 close_frac / min_area_frac）"
        )
    return _sort_reading_order(quads)


@dataclass
class SheetsScoreResult:
    """整床打分结果：逐片 FringeScoreResult（阅读序）+ 六指标 + 汇总便捷量。

    每片角点在各自 sheet_results[i].quad_corners_px（原图坐标）。
    汇总口径：score_min = 最差片（验收最相关）；score_mean = 简单平均。
    sheet_indicators：config 含 indicators 段时逐片计算（阅读序对齐），否则 None。
    """

    sheet_results: list[FringeScoreResult]
    sheet_indicators: list[SheetIndicators] | None = None

    @property
    def n_sheets(self) -> int:
        """检出的玻璃片数。"""
        return len(self.sheet_results)

    @property
    def scores(self) -> list[float]:
        """逐片分数列表（阅读序，0–100 越高越好）。"""
        return [r.score_0_100 for r in self.sheet_results]

    @property
    def score_min(self) -> float:
        """最差片分数（整床验收最相关的汇总量）。"""
        return min(self.scores)

    @property
    def score_mean(self) -> float:
        """全床简单平均分。"""
        return float(np.mean(self.scores))

    @property
    def total_scores(self) -> list[float]:
        """逐片六指标加权总分（阅读序）；未配 indicators 段时报错提示。"""
        if self.sheet_indicators is None:
            raise ValueError("SheetsScoreResult: 未计算六指标（config 缺 indicators 段）")
        return [ind.total_score for ind in self.sheet_indicators]

    @property
    def total_min(self) -> float:
        """最差片的六指标加权总分。"""
        return min(self.total_scores)

    @property
    def total_mean(self) -> float:
        """全床六指标加权总分均值。"""
        return float(np.mean(self.total_scores))


def score_sheets(image: np.ndarray, config: dict | None = None) -> SheetsScoreResult:
    """整床照片 → 切片 → 逐片矫正打分 → SheetsScoreResult。

    config=None 时读 config/fringe_scoring.yaml；打分管线与单片入口完全一致
    （score_fringe_distribution + 显式角点），多片模式不引入任何新打分口径。
    """
    arr = np.asarray(image, dtype=float)
    cfg = config if config is not None else load_config()
    quads = detect_sheet_quads(arr, cfg)
    # 每片先算共享流水线（⓪–③），绝对口径打分与六指标的均匀度口径都从它分叉——
    # 背景拟合等重活只跑一遍（数值与各自重跑逐位相同，见 score.FringePipeline）
    pipes = [compute_pipeline(arr, config=cfg, quad_corners=q) for q in quads]
    results = [score_from_pipeline(p, cfg) for p in pipes]
    indicators = None
    if cfg.get("indicators"):  # 配了 indicators 段才算六指标（旧配置兼容）
        indicators = [
            compute_sheet_indicators(p.arr, ~r.border_mask_arr, cfg, pipeline=p)
            for p, r in zip(pipes, results)
        ]
    return SheetsScoreResult(sheet_results=results, sheet_indicators=indicators)

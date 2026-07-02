"""应力斑分布打分：位置加权惩罚 → 0–100 分（越高越好）+ 原始惩罚值。

公式（docs/应力斑分布打分_需求与设计.md §3）：
- 长宽分别归一化，像素定位在 [-1,1]×[-1,1]，中心=图像几何中心（输入已裁切、玻璃占满整图）；
- 位置权重 w(r) 随离中心距离递减（中心的斑罚最重）；
- penalty_raw = Σ_{斑像素∉边框带} s·w(r)；penalty_max = Σ_{非边框像素} w(r)（最差情形）；
- score = 100 × (1 − penalty_raw / penalty_max)，纯几何归一化，不依赖现场真值。
参数在 config/fringe_scoring.yaml，每次调用按需读取（改 yaml 即生效）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from fringe_scoring.border import BorderBands, band_widths_px, border_mask
from fringe_scoring.quad import detect_glass_quad, order_corners, warp_to_rect
from fringe_scoring.segment import (
    background_with_bands,
    deviation_map,
    fringe_mask_and_intensity,
    robust_scale,
    robust_z_map,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fringe_scoring.yaml"
_DISTANCE_NORMS = ("euclidean", "chebyshev")
_WEIGHT_KINDS = ("linear", "gaussian")


def load_config(config_path: Path | str | None = None) -> dict:
    """读取打分配置（默认 config/fringe_scoring.yaml），每次调用重新读盘。"""
    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("segment", "border", "weight"):
        if section not in cfg:
            raise ValueError(f"load_config: 配置缺少 {section} 段（{path}）")
    return cfg


def normalized_coords(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """图像 shape → 归一化坐标网格 (u, v)，各维像素中心落在 (-1,1) 内。"""
    rows, cols = shape
    u = (2.0 * (np.arange(cols) + 0.5) / cols - 1.0).reshape(1, -1)  # 沿列（长）
    v = (2.0 * (np.arange(rows) + 0.5) / rows - 1.0).reshape(-1, 1)  # 沿行（宽）
    return np.broadcast_to(u, shape).copy(), np.broadcast_to(v, shape).copy()


def center_distance(u: np.ndarray, v: np.ndarray, distance_norm: str) -> np.ndarray:
    """归一化坐标 → 离中心距离 r∈[0,1]：euclidean 圆形等罚线（角点=1），chebyshev 方形等罚线。"""
    if distance_norm not in _DISTANCE_NORMS:
        raise ValueError(f"center_distance: 未知 distance_norm {distance_norm!r}，应为 {_DISTANCE_NORMS}")
    if distance_norm == "euclidean":
        return np.sqrt(u * u + v * v) / np.sqrt(2.0)
    return np.maximum(np.abs(u), np.abs(v))


def position_weight(r: np.ndarray, weight_cfg: dict) -> np.ndarray:
    """距离 r → 惩罚权重 w(r)∈(0,1]：linear=1−r；gaussian=exp(−r²/2σ²)。中心最重。"""
    kind = weight_cfg.get("kind")
    if kind not in _WEIGHT_KINDS:
        raise ValueError(f"position_weight: 未知 kind {kind!r}，应为 {_WEIGHT_KINDS}")
    if kind == "linear":
        return 1.0 - r
    sigma = float(weight_cfg["gaussian_sigma"])
    if sigma <= 0.0:
        raise ValueError("position_weight: gaussian_sigma 须为正")
    return np.exp(-(r * r) / (2.0 * sigma * sigma))


@dataclass
class FringeScoreResult:
    """打分结果：主输出 score_0_100（越高越好）与 penalty_raw；其余为诊断量与可视化用图层。"""

    score_0_100: float
    penalty_raw: float
    penalty_max: float
    fringe_area_frac: float          # 斑面积 / 非边框有效区面积
    centrality: float | None         # 斑深浅加权的平均 w（越大越集中于中心）；无斑为 None
    border_bands: BorderBands
    fringe_mask: np.ndarray = field(repr=False)      # 斑掩膜（含边框剔除）
    intensity_s: np.ndarray = field(repr=False)      # 深浅强度 s∈[0,1]
    border_mask_arr: np.ndarray = field(repr=False)  # 边框带掩膜（True=剔除）
    weight_map: np.ndarray = field(repr=False)       # 位置权重 w(r)
    quad_corners_px: np.ndarray | None = field(repr=False, default=None)  # 原图角点(x,y)×4；None=整图即玻璃
    warped_image: np.ndarray | None = field(repr=False, default=None)     # 实际被打分的（矫正后）图


def score_fringe_distribution(
    image: np.ndarray, config: dict | None = None, quad_corners: np.ndarray | None = None
) -> FringeScoreResult:
    """一张玻璃图 → 应力斑分布打分（流水线见模块 docstring 与 docs §2）。

    config=None 时读 config/fringe_scoring.yaml；测试/调参可直接注入同构 dict。
    第 0 步·四边形矫正：quad_corners 显式给角点则直接矫正；否则 config 的
    quad.auto_detect 开启时自动检测（大图裁切的玻璃可能是平行四边形/一般四边形，
    四边形外的填充不得进入任何统计）；检测出"整图即玻璃"则零开销走原路径。
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("score_fringe_distribution: 需要二维图像数组（单片玻璃裁切图）")
    cfg = config if config is not None else load_config()
    seg_cfg, border_cfg, weight_cfg = cfg["segment"], cfg["border"], cfg["weight"]

    # 第 0 步：四边形检测与透视矫正（矫正域同心矩形等高线 = 原图同心四边形）
    quad_cfg = cfg.get("quad")
    corners = None
    if quad_corners is not None:
        corners = order_corners(np.asarray(quad_corners, dtype=float))
    elif quad_cfg and bool(quad_cfg.get("auto_detect")):
        corners = detect_glass_quad(arr, quad_cfg)
    if corners is not None:
        arr = warp_to_rect(arr, corners)

    block_frac = float(seg_cfg["background_block_frac"])
    poly_degree = int(seg_cfg["background_poly_degree"])
    rows, cols = arr.shape

    # ① 定边框带：按带宽上限扣除四边 → 内部拟合背景曲面（整图求值）→ |残差| 剖面扫描。
    #    边框不参与背景估计，在残差里原样保留；照度渐变被曲面吸收，不会误判成边框。
    max_frac = float(border_cfg["max_band_frac"])
    cap_rows, cap_cols = int(max_frac * rows), int(max_frac * cols)
    pre_background = background_with_bands(
        arr, block_frac, poly_degree, cap_rows, cap_rows, cap_cols, cap_cols
    )
    bands = band_widths_px(np.abs(arr - pre_background), border_cfg)
    border = border_mask(arr.shape, bands)
    interior = ~border
    if not bool(interior.any()):
        raise ValueError("score_fringe_distribution: 边框带占满整图，无有效像素")

    # ② 终版背景 → 残差：只按实测带宽扣除（内部信息最大化利用）
    background = background_with_bands(
        arr, block_frac, poly_degree, bands.top_px, bands.bottom_px, bands.left_px, bands.right_px
    )
    residual = arr - background

    # ③ 稳健 z（median/MAD 只在内部估计；对比度门槛防噪声放大）
    reference_scale = robust_scale(arr[interior])
    z = robust_z_map(residual, interior, float(seg_cfg["min_contrast_frac"]), reference_scale)

    # ④ 斑分割 + 深浅强度
    dev = deviation_map(z, str(seg_cfg["polarity"]))
    fringe, intensity_s = fringe_mask_and_intensity(dev, interior, seg_cfg)

    # ⑤ 位置加权惩罚 → 0–100 分
    u, v = normalized_coords(arr.shape)
    r = center_distance(u, v, str(weight_cfg["distance_norm"]))
    w = position_weight(r, weight_cfg)

    penalty_raw = float((intensity_s * w).sum())           # s 在掩膜外恒为 0
    penalty_max = float(w[interior].sum())                 # 最差情形：有效区全是最深斑
    score = 100.0 * (1.0 - penalty_raw / penalty_max)

    s_total = float(intensity_s.sum())
    centrality = penalty_raw / s_total if s_total > 0.0 else None
    fringe_area_frac = float(fringe.sum()) / float(interior.sum())

    return FringeScoreResult(
        score_0_100=float(np.clip(score, 0.0, 100.0)),
        penalty_raw=penalty_raw,
        penalty_max=penalty_max,
        fringe_area_frac=fringe_area_frac,
        centrality=centrality,
        border_bands=bands,
        fringe_mask=fringe,
        intensity_s=intensity_s,
        border_mask_arr=border,
        weight_map=w,
        quad_corners_px=corners,
        warped_image=arr,
    )

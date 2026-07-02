"""玻璃四边形检测与透视矫正：大图裁切的单片玻璃可能是平行四边形/一般四边形。

前提（现场确认）：裁切图 = 玻璃四边形的外接矩形，四边形之外是无信息填充（近恒定值）。
流程：边界一圈找"恒定值尖峰"估填充 → 玻璃掩膜 → 最大轮廓凸包 → approxPolyDP 拟合 4 角
→ 透视矫正成矩形。矫正域的同心矩形等高线映射回原图即"同心四边形"，
配合 weight.distance_norm=chebyshev 实现按玻璃形状的等罚线。
坐标约定：角点为 (x=列, y=行)，顺序 左上→右上→右下→左下。
参数由 config/fringe_scoring.yaml 的 quad 段驱动，限值不硬编码。
"""

from __future__ import annotations

import numpy as np

from fringe_scoring.segment import robust_scale

# approxPolyDP 收 4 点的 epsilon 递增倍率梯（在 config 基准上放大，收不到即报错）
_EPSILON_LADDER = (1.0, 1.5, 2.0, 3.0, 4.0)


def order_corners(corners: np.ndarray) -> np.ndarray:
    """四角点排序为 左上→右上→右下→左下：按质心极角排环序，再把 x+y 最小者转到首位。"""
    pts = np.asarray(corners, dtype=float).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]  # 图像坐标(y 向下)下按极角升序 = 左上→右上→右下→左下 的环序
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _fill_value_from_border(border_vals: np.ndarray, tol: float) -> tuple[float, float]:
    """边界一圈像素里找"恒定填充"尖峰：按 tol 宽分箱取众数箱。

    返回 (填充值估计, 尖峰占边界像素的比例)。填充近恒定 → 一个箱聚集大量像素；
    整图皆玻璃时边界值分散 → 尖峰占比低，调用方据此判"无填充"。
    """
    vals = np.asarray(border_vals, dtype=float)
    lo = float(vals.min())
    bin_idx = np.floor((vals - lo) / max(tol, 1e-12)).astype(np.int64)
    uniq, counts = np.unique(bin_idx, return_counts=True)
    top = int(np.argmax(counts))
    in_peak = bin_idx == uniq[top]
    return float(np.median(vals[in_peak])), float(counts[top] / vals.size)


def quad_from_hull(hull: np.ndarray, poly_epsilon_frac: float) -> np.ndarray:
    """凸包轮廓 → approxPolyDP 拟合 4 角（epsilon 逐级放大），返回排序后 (4,2) 角点。

    单片检测（detect_glass_quad）与多片切分（sheets.detect_sheet_quads）共用。
    收不到 4 角抛 ValueError（规则 > AI，不硬凑）。
    """
    import cv2

    perimeter = cv2.arcLength(hull, True)
    base_eps = float(poly_epsilon_frac) * perimeter
    for mult in _EPSILON_LADDER:  # epsilon 逐级放大直到收敛为 4 角
        approx = cv2.approxPolyDP(hull, base_eps * mult, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2))
    raise ValueError(
        f"quad_from_hull: 轮廓无法拟合为四边形（最终 {len(approx)} 点），"
        "确认玻璃为凸四边形或调大 poly_epsilon_frac"
    )


def detect_glass_quad(image: np.ndarray, quad_cfg: dict) -> np.ndarray | None:
    """检测玻璃四边形角点；返回 (4,2) 角点（x,y，左上起顺时针）或 None（整图即玻璃）。

    返回 None 的情形（都走"整图即玻璃"原路径）：
    - 边界无恒定填充尖峰（正矩形裁切，玻璃占满整图）；
    - 玻璃占比 ≥ full_frame_frac（填充残余可忽略）；
    - 玻璃占比 ≈ 0（近恒定图 = 无纹理玻璃，按无瑕处理，不当"全是填充"拒绝）。
    玻璃占比落在 (≈0, min_glass_frac) 之间 → 检测异常，报错拒绝打分（规则 > AI）。
    """
    import cv2

    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("detect_glass_quad: 需要二维图像数组")

    tol = float(quad_cfg["fill_tol_frac"]) * robust_scale(arr)
    border_vals = np.concatenate([arr[0], arr[-1], arr[1:-1, 0], arr[1:-1, -1]])
    fill_value, peak_frac = _fill_value_from_border(border_vals, max(tol, 1e-12))
    if peak_frac < float(quad_cfg["min_fill_border_frac"]):
        return None  # 边界没有恒定填充 → 整图即玻璃

    glass_mask = np.abs(arr - fill_value) > tol
    glass_frac = float(glass_mask.mean())
    full_frame = float(quad_cfg["full_frame_frac"])
    if glass_frac >= full_frame:
        return None
    if glass_frac <= 1.0 - full_frame:
        return None  # 近恒定图：视为无纹理玻璃整图（用户确认图内只有玻璃）
    if glass_frac < float(quad_cfg["min_glass_frac"]):
        raise ValueError(
            f"detect_glass_quad: 玻璃占比 {glass_frac:.2%} < min_glass_frac，检测异常，拒绝打分"
        )

    contours, _ = cv2.findContours(
        glass_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("detect_glass_quad: 未找到玻璃轮廓")
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    return quad_from_hull(hull, float(quad_cfg["poly_epsilon_frac"]))


def warp_to_rect(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """按角点把四边形玻璃透视矫正成矩形；目标宽/高取对边平均长度（保像素尺度）。"""
    import cv2

    arr = np.asarray(image, dtype=float)
    src = order_corners(corners).astype(np.float32)
    tl, tr, br, bl = src
    width = max(2, int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0)))
    height = max(2, int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0)))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        arr.astype(np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR
    )
    return np.asarray(warped, dtype=float)

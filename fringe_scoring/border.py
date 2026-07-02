"""边框重应力斑带检测：四边各自自适应定带宽，剔出惩罚统计之外。

需求背景：玻璃边框一定有很重的应力斑、宽窄不一 → 不能用固定比例。
输入是非负偏离图（|图像 − 预背景|，由 score.py 用奇反射外延先消掉照度渐变与
边框吸收），按行/列取分位数得 1D 剖面，从每条边向内扫描：
剖面回落到"中段基线 + 内部波动 × ratio"以下即为带的内边界——纯局部判据，
不做任何拟合外推（拟合外推的杠杆臂会把微小斜率误差放大成边缘系统偏移）。
带宽再 clip 到 config 给的 [min_band_frac, max_band_frac] × 边长保护区间。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fringe_scoring.segment import robust_scale


@dataclass
class BorderBands:
    """四边边框带宽（px）：top/bottom 沿行方向，left/right 沿列方向。"""

    top_px: int
    bottom_px: int
    left_px: int
    right_px: int


def _scan_band(profile: np.ndarray, ratio: float, min_px: int, max_px: int) -> int:
    """从剖面起点向内扫描边框带宽（px）：值 > 中段基线 + 内部波动×ratio 视为仍在带内。

    - 中段（1/4~3/4）不含边框：基线 = 中段中位数，内部波动 = 中段稳健尺度（1.4826×MAD）；
      中心有斑只会抬高两者 → 边框判得更保守，不会把内部误判成边框带；
    - 无噪声合成图两者皆 0，判据退化为"严格 > 0"，仍能精确停在带边界。
    """
    n = profile.size
    lo, hi = n // 4, max(n // 4 + 2, (3 * n) // 4)
    central = profile[lo:hi]
    threshold = float(np.median(central)) + ratio * robust_scale(central)
    i = 0
    while i < max_px and profile[i] > threshold:
        i += 1
    return int(np.clip(i, min_px, max_px))


def band_widths_px(dev_abs: np.ndarray, border_cfg: dict) -> BorderBands:
    """非负偏离图（|图像−预背景|）→ 四边各自的边框带宽（px），由 config 的 border 段驱动。"""
    arr = np.asarray(dev_abs, dtype=float)
    if arr.ndim != 2:
        raise ValueError("band_widths_px: 需要二维数组")
    rows, cols = arr.shape

    q = float(border_cfg["profile_quantile"])
    ratio = float(border_cfg["return_to_background_ratio"])
    min_frac = float(border_cfg["min_band_frac"])
    max_frac = float(border_cfg["max_band_frac"])
    if not 0.0 <= min_frac <= max_frac <= 0.5:
        raise ValueError("band_widths_px: 须满足 0 ≤ min_band_frac ≤ max_band_frac ≤ 0.5")

    row_profile = np.quantile(arr, q, axis=1)  # 每行一个值，扫 top/bottom
    col_profile = np.quantile(arr, q, axis=0)  # 每列一个值，扫 left/right

    row_min, row_max = int(np.ceil(min_frac * rows)), int(np.floor(max_frac * rows))
    col_min, col_max = int(np.ceil(min_frac * cols)), int(np.floor(max_frac * cols))

    return BorderBands(
        top_px=_scan_band(row_profile, ratio, row_min, row_max),
        bottom_px=_scan_band(row_profile[::-1], ratio, row_min, row_max),
        left_px=_scan_band(col_profile, ratio, col_min, col_max),
        right_px=_scan_band(col_profile[::-1], ratio, col_min, col_max),
    )


def border_mask(shape: tuple[int, int], bands: BorderBands) -> np.ndarray:
    """带宽 → 布尔掩膜（True=边框带，不计惩罚；但归一化坐标仍按整图算）。"""
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    if bands.top_px > 0:
        mask[: bands.top_px, :] = True
    if bands.bottom_px > 0:
        mask[rows - bands.bottom_px:, :] = True
    if bands.left_px > 0:
        mask[:, : bands.left_px] = True
    if bands.right_px > 0:
        mask[:, cols - bands.right_px:] = True
    return mask

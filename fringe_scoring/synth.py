"""合成应力斑图生成：背景（水平+渐变）+ 高斯斑 + 宽窄不一的重边框 + 噪声。

用途：单测的性质验证与 run_score.py --demo 演示。**不是真实标定数据**，
不得当真值喂判级（规则 > AI）；真实图放 data/images/stress_fringe/。
"""

from __future__ import annotations

import numpy as np


def make_glass_image(
    rows: int = 200,
    cols: int = 300,
    background_level: float = 100.0,
    gradient_amp: float = 10.0,
    noise_sigma: float = 1.0,
    blobs: list[tuple[float, float, float, float]] | None = None,
    frame_widths_frac: tuple[float, float, float, float] | None = None,
    frame_amp: float = 40.0,
    seed: int = 0,
) -> np.ndarray:
    """合成一张"玻璃"灰度图（float 数组，值≈灰度强度，无物理单位）。

    - blobs: [(row_frac, col_frac, sigma_frac, amp), ...]，高斯斑：中心按占比定位，
      sigma_frac 相对短边，amp 为峰值幅度（正=偏亮斑）；
    - frame_widths_frac: (top, bottom, left, right) 四边边框带宽占比（宽窄可不一），
      None=无边框；边框加 frame_amp 的重应力斑；
    - 背景 = background_level + 沿列的线性渐变(gradient_amp) + 高斯噪声(noise_sigma)。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)

    img = background_level + gradient_amp * (xx / max(cols - 1, 1))
    if noise_sigma > 0:
        img = img + rng.normal(0.0, noise_sigma, size=(rows, cols))

    short = float(min(rows, cols))
    for row_frac, col_frac, sigma_frac, amp in blobs or []:
        cy, cx = row_frac * (rows - 1), col_frac * (cols - 1)
        sigma = max(sigma_frac * short, 1.0)
        img = img + amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))

    if frame_widths_frac is not None:
        top, bottom, left, right = frame_widths_frac
        frame = np.zeros((rows, cols), dtype=bool)
        if top > 0:
            frame[: max(1, int(top * rows)), :] = True
        if bottom > 0:
            frame[rows - max(1, int(bottom * rows)):, :] = True
        if left > 0:
            frame[:, : max(1, int(left * cols))] = True
        if right > 0:
            frame[:, cols - max(1, int(right * cols)):] = True
        img = img + frame_amp * frame  # 边框带整体压上重应力斑

    return img

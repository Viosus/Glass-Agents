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


def make_bed_image(
    sheets: list[tuple[np.ndarray, np.ndarray]],
    out_shape: tuple[int, int],
    fill_value: float = 30.0,
    noise_sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """多片玻璃合成整床照片：逐片透视贴图，片外为恒定填充，最后整图加噪声。

    sheets: [(玻璃图, 目标角点(4,2)), ...]，角点约定同 warp_into_quad；
    片与片不应重叠（后贴覆盖先贴）。用于多片切分（sheets.py）的性质测试与演示。
    """
    import cv2

    canvas = np.full(out_shape, float(fill_value), dtype=float)
    for image, corners in sheets:
        arr = np.asarray(image, dtype=np.float32)
        rows, cols = arr.shape
        src = np.array(
            [[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]], dtype=np.float32
        )
        dst = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            arr, matrix, (out_shape[1], out_shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        ones = np.ones((rows, cols), dtype=np.float32)
        inside = cv2.warpPerspective(  # 玻璃区域掩膜：同一变换贴一张全 1 图
            ones, matrix, (out_shape[1], out_shape[0]),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        ) > 0.5
        canvas[inside] = np.asarray(warped, dtype=float)[inside]

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        canvas = canvas + rng.normal(0.0, noise_sigma, size=out_shape)
    return canvas


def warp_into_quad(
    image: np.ndarray,
    corners: np.ndarray,
    out_shape: tuple[int, int],
    fill_value: float = 30.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """把矩形"玻璃"图正向透视贴进画布上的指定四边形，外部用恒定值填充。

    模拟"大图裁切出的非正矩形玻璃"：corners 为 (4,2) 目标角点（x=列, y=行，
    左上→右上→右下→左下），out_shape=(rows, cols) 为裁切图画布尺寸。
    noise_sigma>0 时在贴图**之后**只往玻璃区域内加噪声——真实传感器噪声长在
    大图上、是清晰的（不经过贴图插值），源图应传噪声为零的版本以免双重平滑。
    """
    import cv2

    arr = np.asarray(image, dtype=np.float32)
    rows, cols = arr.shape
    src = np.array(
        [[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]], dtype=np.float32
    )
    dst = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(
        arr,
        matrix,
        (out_shape[1], out_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill_value),
    )
    out = np.asarray(out, dtype=float)

    if noise_sigma > 0:
        ones = np.ones((rows, cols), dtype=np.float32)
        inside = cv2.warpPerspective(  # 玻璃区域掩膜：同一变换贴一张全 1 图
            ones, matrix, (out_shape[1], out_shape[0]),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        ) > 0.5
        rng = np.random.default_rng(seed)
        out[inside] = out[inside] + rng.normal(0.0, noise_sigma, size=int(inside.sum()))
    return out

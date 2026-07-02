"""应力斑 / 背景分割：背景照度估计 → 稳健 z（median/MAD）→ 斑掩膜 + 深浅强度。

口径 = **相对检测、绝对惩罚**（2026-07-02 拍板，见设计文档决策 #11）：
- 找斑（掩膜/质控门）用逐片稳健 z——"这个像素相对本片噪声地板是不是显著偏离"；
- 深浅强度 s 默认用**绝对灰度锚**（s_scale_mode=absolute）——跨片可比的严重度。
  旧的逐片 MAD 锚（per_sheet）在"整片重斑"上排序反转：斑本身撑大自身尺度估计
  （超出 MAD 稳健性的稀疏污染假设），z 被压扁 → 均匀地差被判成均匀地好。
参数由 config/fringe_scoring.yaml 的 segment 段驱动，禁止硬编码限值。
"""

from __future__ import annotations

import numpy as np

# MAD → 正态一致性尺度因子（1/Φ^{-1}(3/4)，统计常数，非工艺限值）
MAD_TO_SIGMA = 1.4826
# Tukey bisquare 截断常数（95% 正态效率，统计常数）与 IRLS 迭代数
TUKEY_C = 4.685
_IRLS_ITERS = 5

_POLARITIES = ("both", "bright", "dark")
_METHODS = ("quantile", "robust_z")
_S_SCALE_MODES = ("absolute", "per_sheet")


def robust_scale(values: np.ndarray) -> float:
    """稳健尺度：1.4826 × MAD（中位绝对偏差），对 <50% 污染不敏感。"""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return MAD_TO_SIGMA * float(np.median(np.abs(arr - med)))


def _block_medians(image: np.ndarray, block_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """分块中位数网格：返回 (块中心行坐标, 块中心列坐标, 块中位值)，各为一维数组。

    块内取中位数——对块内 <50% 的斑污染稳健；整块都是斑时交给上层 IRLS 当离群点。
    """
    rows, cols = image.shape
    n_row_blocks, n_col_blocks = max(1, rows // block_px), max(1, cols // block_px)
    r_centers, c_centers, values = [], [], []
    for i in range(n_row_blocks):
        r0 = i * block_px
        r1 = rows if i == n_row_blocks - 1 else r0 + block_px  # 末块吃掉余数
        for j in range(n_col_blocks):
            c0 = j * block_px
            c1 = cols if j == n_col_blocks - 1 else c0 + block_px
            values.append(float(np.median(image[r0:r1, c0:c1])))
            r_centers.append((r0 + r1 - 1) / 2.0)
            c_centers.append((c0 + c1 - 1) / 2.0)
    return np.asarray(r_centers), np.asarray(c_centers), np.asarray(values)


def _poly_terms(u, v, degree: int) -> list:
    """二维多项式基 u^i·v^j（i+j ≤ degree）；u/v 可为一维（设计矩阵）或网格（求值）。"""
    return [u**i * v**j for i in range(degree + 1) for j in range(degree + 1 - i)]


def background_with_bands(
    image: np.ndarray,
    block_frac: float,
    poly_degree: int,
    top_px: int = 0,
    bottom_px: int = 0,
    left_px: int = 0,
    right_px: int = 0,
) -> np.ndarray:
    """扣除四边带宽后，用内部分块中位 + Tukey-IRLS 拟合低次多项式背景曲面，整图求值。

    为什么是全局低次曲面而不是局部中值滤波：
    - 曲面只吸收照度渐变等大尺度趋势，**吸收不了紧凑的应力斑**（局部中值滤波在
      "斑比滤波窗宽"时会把斑吸进背景，且鼓包斜坡在边缘外推时被放大成假偏离）；
    - 斑/边框所在的块被 Tukey 双平方权重当离群点压零，不拽弯曲面；
    - 整图求值天然覆盖边缘带，无外推杠杆；
    - 全程对 a·x+b 仿射等变（IRLS 权重基于稳健标准化残差，尺度不变）。
    次数默认 1（平面）：平面没有"穹顶"模式，宽斑最多带来微小平移/倾斜（有界）；
    degree=2 只在照度确有弯曲时用——宽斑肩部会污染曲率项（中央隆起、四角下沉），
    再被 IRLS 的角部误拒正反馈放大，需目检确认。
    """
    arr = np.asarray(image, dtype=float)
    rows, cols = arr.shape
    if rows < 2 or cols < 2:
        raise ValueError("background_with_bands: 图像至少 2×2")
    top, bottom, left, right = int(top_px), int(bottom_px), int(left_px), int(right_px)
    core = arr[top: rows - bottom, left: cols - right]
    if core.size == 0:
        raise ValueError("background_with_bands: 扣除边框带后无内部像素")

    block_px = max(2, int(round(block_frac * min(core.shape))))
    r_centers, c_centers, z = _block_medians(core, block_px)

    # 坐标统一归一到全图 [-1,1]（拟合与求值同一组基，数值条件好）
    u = 2.0 * (c_centers + left) / (cols - 1) - 1.0
    v = 2.0 * (r_centers + top) / (rows - 1) - 1.0
    design = np.stack(_poly_terms(u, v, poly_degree), axis=-1)
    if design.shape[0] < design.shape[1]:
        raise ValueError("background_with_bands: 块数少于多项式系数，调小 background_block_frac")

    weights = np.ones(z.size)
    coef = np.zeros(design.shape[1])
    for _ in range(_IRLS_ITERS):
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        coef, *_ = np.linalg.lstsq(design * sqrt_w, z * sqrt_w.ravel(), rcond=None)
        resid = z - design @ coef
        scale = robust_scale(resid)
        if scale <= 0.0:
            break  # 拟合已精确（如常量图/纯渐变），权重不必再迭代
        t = resid / (TUKEY_C * scale)
        weights = np.where(np.abs(t) < 1.0, (1.0 - t * t) ** 2, 0.0)

    # 整图求值：广播逐项累加，不建全尺寸设计矩阵
    u_grid = (2.0 * np.arange(cols) / (cols - 1) - 1.0).reshape(1, -1)
    v_grid = (2.0 * np.arange(rows) / (rows - 1) - 1.0).reshape(-1, 1)
    bg = np.zeros((rows, cols))
    for c_k, term in zip(coef, _poly_terms(u_grid, v_grid, poly_degree)):
        bg = bg + c_k * term
    return bg


def robust_z_map(
    residual: np.ndarray,
    region_mask: np.ndarray,
    min_contrast_frac: float,
    reference_scale: float,
) -> np.ndarray:
    """残差 → 稳健 z 图：median/MAD 只在 region_mask（非边框内部）内估计。

    残差尺度 ≤ reference_scale × min_contrast_frac 时判"无斑"（返回全 0），
    防止把平滑玻璃的数值噪声当尺度放大成斑；常量图（MAD=0）同样安全退化。
    """
    res = np.asarray(residual, dtype=float)
    inside = res[region_mask]
    if inside.size == 0:
        raise ValueError("robust_z_map: 内部区域无有效像素")
    med = float(np.median(inside))
    scale = robust_scale(inside)
    if scale <= min_contrast_frac * reference_scale or scale <= 0.0:
        return np.zeros_like(res)  # 无斑：残差起伏低于对比度门槛
    return (res - med) / scale


def deviation_map(z: np.ndarray, polarity: str) -> np.ndarray:
    """按极性把 z 折成非负偏离 dev：both→|z|，bright→只算偏亮，dark→只算偏暗。"""
    if polarity not in _POLARITIES:
        raise ValueError(f"deviation_map: 未知 polarity {polarity!r}，应为 {_POLARITIES}")
    if polarity == "both":
        return np.abs(z)
    if polarity == "bright":
        return np.maximum(z, 0.0)
    return np.maximum(-z, 0.0)


def fringe_mask_and_intensity(
    dev: np.ndarray,
    region_mask: np.ndarray,
    seg_cfg: dict,
    dev_gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """偏离图 → (斑掩膜, 深浅强度 s∈[0,1])，只在 region_mask 内判定。

    按 s_scale_mode 整体切换判定域（掩膜与强度必须同域，否则"整片重斑"会在
    z 域检测里隐身——z 被自身尺度压扁到 min_z 以下，掩膜为空 → 假满分）：
    - absolute（默认）：掩膜阈值与质控门、强度 s 全在**灰度域**（需传 dev_gray）——
      quantile 取 dev_gray 最深 top_fraction 且 ≥ min_dev_gray；
      s = clip(dev_gray / s_saturation_gray, 0, 1)。跨片可比（前提：同批曝光固定）。
    - per_sheet（旧行为）：全在逐片 z 域——quantile 取 dev 最深 top_fraction 且 ≥ min_z；
      s = clip(dev / z_saturation, 0, 1)。仅曝光不可控时退用；整片重斑会撑大自身
      尺度估计 → 漏罚（排序反转，决策 #11）。
    - robust_z 法的掩膜固定在 z 域（该方法按定义是 z 阈值），s 仍按模式取锚。
    好玻璃即使被硬选 top-q，其偏离≈0 → s≈0 → 惩罚≈0（两种模式同理）。
    """
    method = seg_cfg.get("method")
    if method not in _METHODS:
        raise ValueError(f"fringe_mask_and_intensity: 未知 method {method!r}，应为 {_METHODS}")
    mode = str(seg_cfg.get("s_scale_mode", "per_sheet"))  # 缺省取旧行为，兼容已注入的旧配置
    if mode not in _S_SCALE_MODES:
        raise ValueError(f"fringe_mask_and_intensity: 未知 s_scale_mode {mode!r}，应为 {_S_SCALE_MODES}")

    dev = np.asarray(dev, dtype=float)
    if dev[region_mask].size == 0:
        raise ValueError("fringe_mask_and_intensity: 内部区域无有效像素")
    if mode == "absolute":
        if dev_gray is None:
            raise ValueError("fringe_mask_and_intensity: absolute 模式需要 dev_gray（灰度域偏离图）")
        dev_gray = np.asarray(dev_gray, dtype=float)

    # 掩膜判定域：absolute+quantile 用灰度域（配 min_dev_gray），其余用 z 域（配 min_z）
    if mode == "absolute" and method == "quantile":
        base, floor = dev_gray, float(seg_cfg["min_dev_gray"])
    else:
        base, floor = dev, float(seg_cfg["min_z"])
    assert base is not None  # absolute 分支已确保 dev_gray 非 None

    if method == "quantile":
        top_fraction = float(seg_cfg["top_fraction"])
        if not 0.0 < top_fraction < 1.0:
            raise ValueError("fringe_mask_and_intensity: top_fraction 须在 (0,1) 内")
        threshold = float(np.quantile(base[region_mask], 1.0 - top_fraction))
        threshold = max(threshold, floor)  # 质控下限：top-q 且至少中等偏离
        mask = region_mask & (base >= threshold) & (base > 0.0)  # base>0 排除退化全 0
    else:
        threshold = float(seg_cfg["z_threshold"])
        mask = region_mask & (dev >= threshold)

    if mode == "absolute":
        saturation = seg_cfg.get("s_saturation_gray")
        if saturation is None or float(saturation) <= 0.0:
            raise ValueError("fringe_mask_and_intensity: absolute 模式需配置正的 s_saturation_gray")
        assert dev_gray is not None
        intensity_s = np.where(mask, np.clip(dev_gray / float(saturation), 0.0, 1.0), 0.0)
        return mask, intensity_s

    z_saturation = float(seg_cfg["z_saturation"])
    if z_saturation <= 0.0:
        raise ValueError("fringe_mask_and_intensity: z_saturation 须为正")
    intensity_s = np.where(mask, np.clip(dev / z_saturation, 0.0, 1.0), 0.0)
    return mask, intensity_s

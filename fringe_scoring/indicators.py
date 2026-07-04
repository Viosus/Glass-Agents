"""六指标测量与加权总分：X0.95 / 灰度方差 / 梯度均值 / 梯度方差 / CCP / 均匀度。

统计域 = 透视矫正后单片内部（扣细圈边框带），与均匀度评分同域（2026-07-03 拍板）。
- 深浅类五指标（X0.95/灰度方差/梯度均值/梯度方差/CCP）承担"斑有多深/多粗糙"；
- 均匀度只关于斑纹**空间分布**，与整体灰度深浅解耦（同图案加深不改分，
  全白/常量图=100）——实现 = per_sheet z 口径的分布评分（对逐片仿射变换不变）。
- 每指标经 config 好/差参考值线性映射为 0–100 子分（方向统一为越高越好），
  总分 = Σwᵢsᵢ/Σwᵢ，权重默认平权、工厂可调（config indicators 段，运行时读取）。

标定缺口（2026-07-03 拍板：灰度代理+标注，标定值到位后填 config 即切换）：
- X0.95 的"光程差 nm"需 gray_to_nm 标定，未标定时输出灰度域分位（unit="gray"）；
- CCP 需 mm_per_px 空间标定（未标定=按原始像素，跨设备不可比）与参考样品
  Cmax/CPmax（当前为本批实测初值，非国标标定，ccp_reference_is_plant_calibrated=false）。
失败 = 抛 ValueError（缺配置段/键、权重全零、参考值退化），绝不静默给分。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fringe_scoring.score import score_fringe_distribution

# 六指标固定键名（config indicators.weights / indicators.refs 与此对齐）
INDICATOR_KEYS = (
    "x095", "gray_variance", "gradient_mean", "gradient_variance", "ccp", "uniformity",
)
_GLCM_LEVELS = 8  # GLCM 灰度级 Ng（草案 §4.4 口径，结构常数非限值）
# GLCM 四方向 d=1 的 (行偏移, 列偏移)：0° / 45° / 90° / 135°（与 skimage 口径一致）
_GLCM_OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))


@dataclass
class SheetIndicators:
    """一片玻璃的六指标结果：原始值 + 0–100 子分 + 加权总分 + 标定标注。"""

    x095: float                      # 内部灰度(或 nm)的 95% 分位
    x095_unit: str                   # "gray"=未标定灰度代理 | "nm"=已按 gray_to_nm 标定
    gray_variance: float             # 内部灰度方差
    gradient_mean: float             # Sobel 梯度幅值均值
    gradient_variance: float         # Sobel 梯度幅值方差
    ccp_value: float                 # CCP = 0.5(√(Ca/Cmax) + (CPa/CPmax)^0.25)
    ccp_ca: float                    # GLCM 对比度（四方向平均）
    ccp_cpa: float                   # GLCM 聚类突出（四方向平均）
    ccp_reference_is_plant_calibrated: bool  # False=参考值为开发期批内初值，非国标标定
    uniformity: float                # 斑纹空间分布均匀度 0–100（与深浅解耦）
    sub_scores: dict[str, float] = field(repr=False)   # 六指标 0–100 子分（越高越好）
    weights: dict[str, float] = field(repr=False)      # 当次生效权重（快照，便于追溯）
    total_score: float = 0.0         # 加权总分 0–100

    @property
    def raw_values(self) -> dict[str, float]:
        """六项原始值（键=INDICATOR_KEYS）：换加权方案时喂 score_from_raw 免重跑图像。"""
        return {
            "x095": self.x095,
            "gray_variance": self.gray_variance,
            "gradient_mean": self.gradient_mean,
            "gradient_variance": self.gradient_variance,
            "ccp": self.ccp_value,
            "uniformity": self.uniformity,
        }


def _interior_crop(image: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """内部掩膜（构造上为矩形）→ 裁出内部矩形子图（供 GLCM 等需要 2D 连续域的指标）。"""
    rows_any = np.flatnonzero(interior.any(axis=1))
    cols_any = np.flatnonzero(interior.any(axis=0))
    if rows_any.size == 0 or cols_any.size == 0:
        raise ValueError("_interior_crop: 内部区域为空")
    return np.asarray(image, dtype=float)[
        rows_any[0]: rows_any[-1] + 1, cols_any[0]: cols_any[-1] + 1
    ]


def _gradient_stats(image: np.ndarray, interior: np.ndarray) -> tuple[float, float]:
    """Sobel(3×3) 梯度幅值 |∇I| = √(Gx²+Gy²) 在内部的 (均值, 方差)。"""
    import cv2

    arr = np.asarray(image, dtype=float)
    gx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)[interior]
    return float(mag.mean()), float(mag.var())


def _resample_to_1px_per_mm(image: np.ndarray, mm_per_px: float) -> np.ndarray:
    """图像标准化重采样到 1 px/mm（草案 §4.4 b 口径），保证 CCP 跨尺寸可比。"""
    import cv2

    arr = np.asarray(image, dtype=np.float32)
    if mm_per_px <= 0:
        raise ValueError("indicators: mm_per_px 须为正")
    if abs(mm_per_px - 1.0) < 1e-12:
        return np.asarray(arr, dtype=float)
    rows = max(2, int(round(arr.shape[0] * mm_per_px)))
    cols = max(2, int(round(arr.shape[1] * mm_per_px)))
    # 降采样（分辨率高于 1px/mm）用 INTER_AREA 抗混叠，升采样用线性
    interp = cv2.INTER_AREA if mm_per_px < 1.0 else cv2.INTER_LINEAR
    return np.asarray(cv2.resize(arr, (cols, rows), interpolation=interp), dtype=float)


def _glcm_ca_cpa(image: np.ndarray, ng: int = _GLCM_LEVELS) -> tuple[float, float]:
    """灰度共生矩阵（d=1，四方向，对称+归一）→ (Ca 对比度, CPa 聚类突出)，四方向平均。

    与 tools/metrics.py 的 skimage 口径公式一致（numpy 自实现，零 skimage 依赖）：
    Ca = Σ P(i,j)(i−j)²；CP = Σ P(i,j)(i+j−μx−μy)⁴。
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("indicators: GLCM 需要二维图像")
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        q = np.zeros(arr.shape, dtype=np.intp)  # 常量图：无纹理
    else:
        q = np.floor((arr - lo) / (hi - lo) * (ng - 1) + 0.5).astype(np.intp)
    q = np.clip(q, 0, ng - 1)
    rows, cols = arr.shape

    ca_list, cpa_list = [], []
    i_idx = np.arange(ng).reshape(-1, 1)
    j_idx = np.arange(ng).reshape(1, -1)
    for dr, dc in _GLCM_OFFSETS:
        # 依偏移取像素对 (a, b)，计数入 ng×ng 矩阵；对称化 = 两个方向都计
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        a = q[r0:r1, c0:c1].ravel()
        b = q[r0 + dr: r1 + dr, c0 + dc: c1 + dc].ravel()
        counts = np.zeros((ng, ng), dtype=float)
        np.add.at(counts, (a, b), 1.0)
        counts = counts + counts.T  # symmetric=True
        total = counts.sum()
        if total <= 0:
            raise ValueError("indicators: 图像过小，GLCM 无像素对")
        p = counts / total
        ca_list.append(float((p * (i_idx - j_idx) ** 2).sum()))
        mu_i = float((i_idx * p).sum())
        mu_j = float((j_idx * p).sum())
        cpa_list.append(float((p * (i_idx + j_idx - mu_i - mu_j) ** 4).sum()))
    return float(np.mean(ca_list)), float(np.mean(cpa_list))


def _uniformity_config(cfg: dict, ind_cfg: dict) -> dict:
    """构造均匀度专用配置：per_sheet z 口径（对深浅仿射不变）+ 专属刻度锚。

    输入已是矫正后的单片 → 关四边形检测；gray_threshold 只支持 absolute →
    均匀度掩膜走 quantile z 路径（top_fraction + min_z 沿用主配置）。
    """
    seg = dict(cfg["segment"])
    seg["s_scale_mode"] = "per_sheet"
    seg["method"] = "quantile"
    quad = dict(cfg.get("quad") or {})
    quad["auto_detect"] = False
    return {
        **cfg,
        "segment": seg,
        "quad": quad,
        "scoring": {"penalty_ratio_at_zero": float(ind_cfg["uniformity_ratio_at_zero"])},
    }


def _sub_score(value: float, ref: dict, key: str) -> float:
    """原始值 → 0–100 子分：s = clip(100×(worst−v)/(worst−best))，越高越好。"""
    best, worst = float(ref["best"]), float(ref["worst"])
    if best == worst:
        raise ValueError(f"indicators: refs.{key} 的 best 与 worst 不能相等")
    return float(np.clip(100.0 * (worst - value) / (worst - best), 0.0, 100.0))


def _validate_weights(weights: dict) -> dict[str, float]:
    """权重校验：键恰为六指标、非负、总和为正；返回 float 化副本。"""
    w = {k: float(v) for k, v in (weights or {}).items()}
    if set(w) != set(INDICATOR_KEYS):
        raise ValueError(f"indicators: weights 键须恰为 {INDICATOR_KEYS}")
    if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("indicators: 权重须非负且总和为正")
    return w


def score_from_raw(raw: dict, refs: dict, weights: dict) -> tuple[dict[str, float], float]:
    """六项原始指标值 + 评分方案（refs/weights）→ (六子分, 加权总分)。

    与 compute_sheet_indicators 的评分段同源（后者内部调本函数）：切换加权方案时
    直接用缓存的 raw_values 重算，**无需重跑图像**（毫秒级）。uniformity 已是
    0–100 越高越好，直通；其余五项经 refs 线性映射。
    """
    w = _validate_weights(weights)
    refs = refs or {}
    sub_scores = {k: _sub_score(float(raw[k]), refs[k], k)
                  for k in INDICATOR_KEYS if k != "uniformity" and k in refs}
    missing = set(INDICATOR_KEYS) - {"uniformity"} - set(sub_scores)
    if missing:
        raise ValueError(f"indicators: refs 缺指标 {sorted(missing)}")
    sub_scores["uniformity"] = float(raw["uniformity"])
    total = sum(w[k] * sub_scores[k] for k in INDICATOR_KEYS) / sum(w.values())
    return sub_scores, float(total)


def compute_sheet_indicators(
    warped: np.ndarray, interior: np.ndarray, config: dict
) -> SheetIndicators:
    """矫正后单片 + 内部掩膜 → 六指标 + 加权总分（config 须含 indicators 段）。"""
    arr = np.asarray(warped, dtype=float)
    interior = np.asarray(interior, dtype=bool)
    ind_cfg = config.get("indicators")
    if not ind_cfg:
        raise ValueError("compute_sheet_indicators: 配置缺少 indicators 段（六指标参数）")
    calib = ind_cfg.get("calibration") or {}
    refs = ind_cfg.get("refs") or {}
    weights = _validate_weights(ind_cfg.get("weights"))

    inside = arr[interior]
    if inside.size == 0:
        raise ValueError("compute_sheet_indicators: 内部区域无有效像素")

    # ── 深浅类五指标 ──
    x095 = float(np.percentile(inside, 95))
    gray_to_nm = calib.get("gray_to_nm")
    x095_unit = "gray"
    if gray_to_nm is not None:  # 灰度→nm 标定到位后输出真光程差
        x095 *= float(gray_to_nm)
        x095_unit = "nm"
    gray_variance = float(np.var(inside))
    gradient_mean, gradient_variance = _gradient_stats(arr, interior)

    ccp_img = _interior_crop(arr, interior)
    mm_per_px = calib.get("mm_per_px")
    if mm_per_px is not None:  # 空间标定到位后按草案重采样 1px/mm，跨设备可比
        ccp_img = _resample_to_1px_per_mm(ccp_img, float(mm_per_px))
    ca, cpa = _glcm_ca_cpa(ccp_img)
    c_max, cp_max = float(calib["ccp_c_max"]), float(calib["ccp_cp_max"])
    if c_max <= 0 or cp_max <= 0:
        raise ValueError("compute_sheet_indicators: ccp_c_max / ccp_cp_max 须为正")
    ccp_value = 0.5 * ((ca / c_max) ** 0.5 + (cpa / cp_max) ** 0.25)

    # ── 均匀度（只关于斑纹空间分布，与深浅解耦）──
    uniformity = score_fringe_distribution(
        arr, config=_uniformity_config(config, ind_cfg)
    ).score_0_100

    # ── 子分与加权总分（与 score_from_raw 同源，保证方案快路径重算一致） ──
    raw = {
        "x095": x095, "gray_variance": gray_variance, "gradient_mean": gradient_mean,
        "gradient_variance": gradient_variance, "ccp": ccp_value, "uniformity": uniformity,
    }
    sub_scores, total = score_from_raw(raw, refs, weights)

    return SheetIndicators(
        x095=x095, x095_unit=x095_unit,
        gray_variance=gray_variance,
        gradient_mean=gradient_mean, gradient_variance=gradient_variance,
        ccp_value=ccp_value, ccp_ca=ca, ccp_cpa=cpa,
        ccp_reference_is_plant_calibrated=bool(
            calib.get("ccp_reference_is_plant_calibrated", False)
        ),
        uniformity=uniformity,
        sub_scores=sub_scores, weights=weights, total_score=float(total),
    )

"""六指标测量与加权总分：X0.95 / 灰度方差 / 梯度均值 / 梯度方差 / CCP / 位置评分。

统计域 = 透视矫正后单片内部（扣细圈边框带），与位置指标同域（2026-07-03 拍板）。
- 深浅类五指标（X0.95/灰度方差/梯度均值/梯度方差/CCP）承担"斑有多深/多粗糙"；
- 位置指标只关于斑纹**空间分布**：指标层主量 = 应力斑中心集中度 ρu（越大越差），
  评分层按刻度锚 ρ0 换算为 0–100 位置评分（position_score，越高越好）。与整体
  灰度深浅解耦（同图案加深不改分，全白/常量图=100）——实现 = per_sheet z 口径
  的分布评分（对逐片仿射变换不变）。（2026-07-19 改名，原名"均匀度/uniformity"。）
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

from fringe_scoring.score import (
    FringePipeline,
    compute_pipeline,
    measure_center_concentration,
)

# 六指标固定键名（config indicators.weights / indicators.refs 与此对齐）
INDICATOR_KEYS = (
    "x095", "gray_variance", "gradient_mean", "gradient_variance", "ccp", "position_score",
)
# 改名兼容（2026-07-19："uniformity"→"position_score"）：旧配置/旧调参载荷的键迁移
# 单一事实源——config 迁移、服务器远程调参归一、序列化双写都引用此表
LEGACY_INDICATOR_ALIASES = {"uniformity": "position_score"}
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
    center_concentration: float      # 应力斑中心集中度 ρu ∈ [0,1]（指标层主量，越大越差）
    position_score: float            # 位置评分 0–100（评分层按 ρ0 换算，越高越好；原"均匀度"）
    sub_scores: dict[str, float] = field(repr=False)   # 六指标 0–100 子分（越高越好）
    weights: dict[str, float] = field(repr=False)      # 当次生效权重（快照，便于追溯）
    total_score: float = 0.0         # 加权总分 0–100
    # 位置指标诊断量（任务3，2026-07-19）：spot_area_ratio / p_dark / baseline_flipped /
    # baseline_flip_aborted / threshold_T / quantile_bound / m_R / sigma_R / sigma_ref
    position_diagnostics: dict = field(default_factory=dict, repr=False)

    @property
    def raw_values(self) -> dict[str, float]:
        """六项原始值（键=INDICATOR_KEYS）：换加权方案时喂 score_from_raw 免重跑图像。"""
        return {
            "x095": self.x095,
            "gray_variance": self.gray_variance,
            "gradient_mean": self.gradient_mean,
            "gradient_variance": self.gradient_variance,
            "ccp": self.ccp_value,
            "position_score": self.position_score,
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
    # 量化级存 uint8（ng²−1 ≤ 255 时配对索引也留在 uint8 域）：数值与 intp 完全相同，
    # 只为削内存流量（本函数是带宽热点，级数固定 _GLCM_LEVELS=8 → 索引最大 63）
    idx_dtype = np.uint8 if ng * ng <= 256 else np.intp
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        q = np.zeros(arr.shape, dtype=idx_dtype)  # 常量图：无纹理
    else:
        q = np.clip(np.floor((arr - lo) / (hi - lo) * (ng - 1) + 0.5), 0, ng - 1)
        q = q.astype(idx_dtype)
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
        # bincount 按扁平索引 i·Ng+j 计数 == 逐对累加（np.add.at 的等价快路径）；
        # 索引算术留在 q 的 dtype（uint8 时 ≤63 不溢出），计数结果与 intp 域逐一相同
        pair_index = a * idx_dtype(ng) + b
        counts = np.bincount(pair_index, minlength=ng * ng).astype(float).reshape(ng, ng)
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


# 位置指标掩膜级参数的固定口径（=2026-07 历史默认；config indicators.position_segment 可覆写）。
# 结构口径常数而非工艺限值：位置指标须与"斑纹判定"彻底解耦（2026-07-08 需求）——
# 工厂调判斑灵敏度 min_dev_gray / top_fraction / min_z 等主判定参数时，位置指标不许被牵连。
_POSITION_SEG_DEFAULTS = {
    "top_fraction": 0.15, "min_z": 2.5, "z_saturation": 8.0, "polarity": "bright",
}


def _position_indicator_config(cfg: dict, ind_cfg: dict) -> dict:
    """构造位置指标（中心集中度 ρu）专用配置：per_sheet z 口径（对深浅仿射不变）。

    输入已是矫正后的单片 → 关四边形检测；gray_threshold 只支持 absolute →
    位置指标掩膜走 quantile z 路径。掩膜级参数（top_fraction/min_z/z_saturation/
    polarity）**不继承主配置**而取固定口径（indicators.position_segment 可覆写；
    旧键 uniformity_segment 仍识别，deprecated）：斑纹判定参数怎么调都不影响
    位置指标（min_dev_gray 属 absolute 域，本就不进此路径）。
    背景估计/边框带/对比度门槛仍与主配置共享——那是"背景"不是"判斑"，
    且与绝对口径共用一条流水线（见 score.compute_pipeline）的前提就是同参。
    指标层不注入、不读取刻度锚 ρ0——ρu→位置评分的换算在评分层
    （compute_sheet_indicators 的评分段）完成。
    """
    seg = dict(cfg["segment"])
    seg.update(_POSITION_SEG_DEFAULTS)
    seg_override = ind_cfg.get("position_segment")
    if seg_override is None:  # deprecated：改名前旧键回退
        seg_override = ind_cfg.get("uniformity_segment")
    seg.update(dict(seg_override or {}))
    seg["s_scale_mode"] = "per_sheet"
    seg["method"] = "quantile"
    quad = dict(cfg.get("quad") or {})
    quad["auto_detect"] = False
    # 基准翻转门闩（2026-07-19）：默认启用；config indicators.baseline_flip 段可覆写/
    # 关闭（与历史结果对拍用）。只挂位置指标路径——主分布评分不受影响
    flip = {"enabled": True, **(ind_cfg.get("baseline_flip") or {})}
    return {**cfg, "segment": seg, "quad": quad, "baseline_flip": flip}


# 子分溢出口径（2026-07-13 拍板）：ccp 标尺 [0.2, 1.4]——劣于 worst 仍封 0，
# 优于 best 允许 >100（溢出，随加权传导进总分）；其余五项保持 0–100 双向封顶。
# 三端同步：Dart fringe_algo.dart kOverflowKeys / 小程序 algo.js OVERFLOW_KEYS。
_OVERFLOW_KEYS = frozenset({"ccp"})


def _sub_score(value: float, ref: dict, key: str) -> float:
    """原始值 → 子分：s = 100×(worst−v)/(worst−best)，越高越好。

    常规指标 clip 到 [0,100]；_OVERFLOW_KEYS 只封下界 0（优于 best 溢出 >100）。
    """
    best, worst = float(ref["best"]), float(ref["worst"])
    if best == worst:
        raise ValueError(f"indicators: refs.{key} 的 best 与 worst 不能相等")
    s = 100.0 * (worst - value) / (worst - best)
    if key in _OVERFLOW_KEYS:
        return float(max(0.0, s))
    return float(np.clip(s, 0.0, 100.0))


def _validate_weights(weights: dict) -> dict[str, float]:
    """权重校验：键恰为六指标、非负、总和为正；返回 float 化副本。"""
    w = {k: float(v) for k, v in (weights or {}).items()}
    if set(w) != set(INDICATOR_KEYS):
        raise ValueError(f"indicators: weights 键须恰为 {INDICATOR_KEYS}")
    if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("indicators: 权重须非负且总和为正")
    return w


def select_refs(ind_cfg: dict, thickness_mm: float | None) -> dict:
    """按玻璃厚度选参考值档（厚玻璃天生斑重，须与同厚度参考比才公平）。

    机制与国标分级表同构（表按公称厚度分档，行值=适用"≤该厚度"）：
    refs_by_thickness = [{max_thickness_mm, refs}, ...]，取 max_thickness_mm ≥ 厚度
    的最小档；无厚度 / 未配分档 / 厚度超出全部档 → 回退默认 refs（全厚度口径）。
    各档参考值须由工厂用**同厚度**好/差样片标定，本模块不提供臆造默认。
    """
    bands = ind_cfg.get("refs_by_thickness") or []
    if thickness_mm is not None and bands:
        eligible = [b for b in bands if float(b["max_thickness_mm"]) >= float(thickness_mm)]
        if eligible:
            band = min(eligible, key=lambda b: float(b["max_thickness_mm"]))
            # 档内给的指标覆盖默认，未给的（如 position_score）继承默认参考
            return {**(ind_cfg.get("refs") or {}), **band["refs"]}
    return ind_cfg["refs"]


def score_from_raw(raw: dict, refs: dict, weights: dict) -> tuple[dict[str, float], float]:
    """六项原始指标值 + 评分方案（refs/weights）→ (六子分, 加权总分)。

    与 compute_sheet_indicators 的评分段同源（后者内部调本函数）：切换加权方案时
    直接用缓存的 raw_values 重算，**无需重跑图像**（毫秒级）。
    六项参考值均可调：线性映射 s=clip(100×(worst−v)/(worst−best)) 对两个方向都成立
    （深浅类 worst>best=越小越好；position_score best>worst=越大越好，默认 {best:100,
    worst:0} 等价直通）。refs 缺 position_score 时仍直通（旧配置向后兼容）。
    例外：_OVERFLOW_KEYS（ccp）子分上不封顶，见 _sub_score。
    """
    w = _validate_weights(weights)
    refs = refs or {}
    sub_scores = {k: _sub_score(float(raw[k]), refs[k], k)
                  for k in INDICATOR_KEYS if k in refs}
    missing = set(INDICATOR_KEYS) - {"position_score"} - set(sub_scores)
    if missing:
        raise ValueError(f"indicators: refs 缺指标 {sorted(missing)}")
    if "position_score" not in sub_scores:
        sub_scores["position_score"] = float(raw["position_score"])
    total = sum(w[k] * sub_scores[k] for k in INDICATOR_KEYS) / sum(w.values())
    return sub_scores, float(total)


def compute_sheet_indicators(
    warped: np.ndarray,
    interior: np.ndarray,
    config: dict,
    pipeline: FringePipeline | None = None,
) -> SheetIndicators:
    """矫正后单片 + 内部掩膜 → 六指标 + 加权总分（config 须含 indicators 段）。

    pipeline：该片已算好的共享流水线（score.compute_pipeline 的产物，须与 warped
    同一片、同背景参数）——位置指标直接复用其背景/残差/z，免整条流水线重跑
    （数值与重跑逐位相同，纯提速）；不传则原样重算（外部单独调用向后兼容）。
    """
    arr = np.asarray(warped, dtype=float)
    interior = np.asarray(interior, dtype=bool)
    ind_cfg = config.get("indicators")
    if not ind_cfg:
        raise ValueError("compute_sheet_indicators: 配置缺少 indicators 段（六指标参数）")
    calib = ind_cfg.get("calibration") or {}
    # 参考值按厚度选档（calibration.thickness_mm 为空=未知厚度 → 默认全厚度口径）
    refs = select_refs(ind_cfg, calib.get("thickness_mm")) or {}
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

    # ── 位置指标·指标层（只关于斑纹空间分布，与深浅解耦；判定参数解耦见
    #    _position_indicator_config）：主量 ρu = penalty_raw / penalty_max，不读 ρ0；
    #    含基准翻转门闩与诊断量（score.measure_center_concentration）──
    ucfg = _position_indicator_config(config, ind_cfg)
    pipe_u = pipeline if pipeline is not None else compute_pipeline(arr, ucfg)
    u_res = measure_center_concentration(pipe_u, ucfg)
    penalty_raw, penalty_max = u_res.penalty_raw, u_res.penalty_max
    center_concentration = u_res.center_concentration
    position_diagnostics = {
        "spot_area_ratio": u_res.spot_area_ratio,
        "p_dark": u_res.p_dark,
        "baseline_flipped": u_res.baseline_flipped,
        "baseline_flip_aborted": u_res.baseline_flip_aborted,
        "threshold_T": u_res.threshold_t,
        "quantile_bound": u_res.quantile_bound,
        "m_R": u_res.m_r,
        "sigma_R": u_res.sigma_r,
        "sigma_ref": u_res.sigma_ref,
    }

    # ── 位置指标·评分层：刻度锚 ρ0 只在此处读取（indicators.scoring 段，
    #    旧键 uniformity_ratio_at_zero 回退，deprecated）。表达式与旧实现逐算子
    #    同序同型（100·(1−praw/(pmax·ρ0)) 再 clip），保证改名前后位级相同——
    #    不得改写成 100·(1−ρu/ρ0)（除法结合次序不同，舍入可差 1 ulp）──
    sc_cfg = ind_cfg.get("scoring") or {}
    rho0 = sc_cfg.get("position_ratio_at_zero", ind_cfg.get("uniformity_ratio_at_zero"))
    if rho0 is None:
        raise ValueError(
            "compute_sheet_indicators: 缺位置评分刻度锚 "
            "indicators.scoring.position_ratio_at_zero（旧键 uniformity_ratio_at_zero 亦可）"
        )
    rho0 = float(rho0)
    if not 0.0 < rho0 <= 1.0:
        raise ValueError("compute_sheet_indicators: position_ratio_at_zero 须在 (0,1] 内")
    position_score = float(
        np.clip(100.0 * (1.0 - penalty_raw / (penalty_max * rho0)), 0.0, 100.0)
    )

    # ── 子分与加权总分（与 score_from_raw 同源，保证方案快路径重算一致） ──
    raw = {
        "x095": x095, "gray_variance": gray_variance, "gradient_mean": gradient_mean,
        "gradient_variance": gradient_variance, "ccp": ccp_value,
        "position_score": position_score,
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
        center_concentration=center_concentration,
        position_score=position_score,
        sub_scores=sub_scores, weights=weights, total_score=float(total),
        position_diagnostics=position_diagnostics,
    )

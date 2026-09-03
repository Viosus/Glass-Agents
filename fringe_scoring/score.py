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

from fringe_scoring.border import BorderBands, band_widths_px, border_mask
from fringe_scoring.quad import detect_glass_quad, order_corners, warp_to_rect
from fringe_scoring.segment import (
    background_with_bands,
    baseline_flip_z,
    deviation_map,
    fringe_mask_and_intensity,
    fringe_mask_intensity_info,
    robust_scale,
    robust_z_stats,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fringe_scoring.yaml"
_DISTANCE_NORMS = ("euclidean", "chebyshev")
_WEIGHT_KINDS = ("linear", "gaussian")


def load_config(config_path: Path | str | None = None) -> dict:
    """读取打分配置（默认 config/fringe_scoring.yaml），每次调用重新读盘。

    yaml 延迟导入：嵌入外部软件、用 dict 注入 config 的场景不必安装 pyyaml。
    """
    import yaml

    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("segment", "border", "weight"):
        if section not in cfg:
            raise ValueError(f"load_config: 配置缺少 {section} 段（{path}）")
    return cfg


def normalized_coords(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """图像 shape → 归一化坐标网格 (u, v)，各维像素中心落在 (-1,1) 内。

    返回只读广播视图（零拷贝；下游 center_distance 只读不写——需可写时自行 copy）。
    """
    rows, cols = shape
    u = (2.0 * (np.arange(cols) + 0.5) / cols - 1.0).reshape(1, -1)  # 沿列（长）
    v = (2.0 * (np.arange(rows) + 0.5) / rows - 1.0).reshape(-1, 1)  # 沿行（宽）
    return np.broadcast_to(u, shape), np.broadcast_to(v, shape)


def center_distance(u: np.ndarray, v: np.ndarray, distance_norm: str) -> np.ndarray:
    """归一化坐标 → 离中心距离 r∈[0,1]：euclidean 圆形等罚线（角点=1），chebyshev 方形等罚线。"""
    if distance_norm not in _DISTANCE_NORMS:
        raise ValueError(f"center_distance: 未知 distance_norm {distance_norm!r}，应为 {_DISTANCE_NORMS}")
    if distance_norm == "euclidean":
        return np.sqrt(u * u + v * v) / np.sqrt(2.0)
    return np.maximum(np.abs(u), np.abs(v))


def position_weight(r: np.ndarray, weight_cfg: dict) -> np.ndarray:
    """距离 r → 惩罚权重 w(r)∈(0,1]：linear=1−r；gaussian=exp(−r²/2σ²)。中心最重。

    floor（默认 0）为边缘权重下限：w = max(形状值, floor)——边缘重斑不免罚
    （2026-07-03 人工锚点：边缘集中重斑的片人工判不及格，决策 #12），
    中心 > 边缘的次序保持不变。
    """
    kind = weight_cfg.get("kind")
    if kind not in _WEIGHT_KINDS:
        raise ValueError(f"position_weight: 未知 kind {kind!r}，应为 {_WEIGHT_KINDS}")
    floor = float(weight_cfg.get("floor", 0.0))
    if not 0.0 <= floor < 1.0:
        raise ValueError("position_weight: floor 须在 [0,1) 内")
    if kind == "linear":
        w = 1.0 - r
    else:
        sigma = float(weight_cfg["gaussian_sigma"])
        if sigma <= 0.0:
            raise ValueError("position_weight: gaussian_sigma 须为正")
        w = np.exp(-(r * r) / (2.0 * sigma * sigma))
    return np.maximum(w, floor)


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


@dataclass
class FringePipeline:
    """步骤⓪–③的共享中间结果：绝对口径与位置指标口径只在掩膜/刻度（④⑤）分叉，
    矫正/边框带/背景/残差/z 完全同参 → 算一次两处复用（数值与各自重算逐位相同）。"""

    arr: np.ndarray                      # 矫正后被打分图
    corners: np.ndarray | None           # 原图角点；None=整图即玻璃
    bands: BorderBands                   # 实测边框带宽
    border: np.ndarray                   # 免罚边框掩膜（True=剔除）
    interior: np.ndarray                 # 惩罚域内部
    residual: np.ndarray                 # 图 − 背景曲面
    z: np.ndarray                        # 稳健 z 图
    residual_floor: float                # quantile(residual[interior], block_q)——dev_gray 暗地板锚
    weight_map: np.ndarray               # 位置权重 w(r)
    reference_scale: float               # σref = robust_scale(灰度[interior])（无斑退化门参照）
    z_med: float                         # m_R = med(残差[interior])（z 的基准，诊断量）
    z_scale: float                       # σR = robust_scale(残差[interior])（z 的尺度，诊断量）


def compute_pipeline(
    image: np.ndarray, config: dict | None = None, quad_corners: np.ndarray | None = None
) -> FringePipeline:
    """一张玻璃图 → 共享流水线（步骤⓪–③ + 位置权重）。

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
    block_q = float(seg_cfg.get("background_block_quantile", 0.5))  # 0.5=中位数旧口径；低分位=暗地板锚
    rows, cols = arr.shape

    # ① 定边框带：按带宽上限扣除四边 → 内部拟合背景曲面（整图求值）→ |残差| 剖面扫描。
    #    边框不参与背景估计，在残差里原样保留；照度渐变被曲面吸收，不会误判成边框。
    max_frac = float(border_cfg["max_band_frac"])
    cap_rows, cap_cols = int(max_frac * rows), int(max_frac * cols)
    pre_background = background_with_bands(
        arr, block_frac, poly_degree, cap_rows, cap_rows, cap_cols, cap_cols, block_q
    )
    bands = band_widths_px(np.abs(arr - pre_background), border_cfg)

    # 免罚域边框带 = min(实测带宽, 正常允许宽度)：细圈是物理正常（免罚），
    # 超宽段是质量缺陷 → 留在惩罚域按普通斑计罚（决策 #13，人工语义确认）。
    # 背景/残差估计仍用实测全带宽（带内像素不进背景拟合，与免罚多宽无关）。
    normal_frac = border_cfg.get("normal_band_frac")
    if normal_frac is None:
        penalty_bands = bands  # 兼容旧配置：不设允许值 = 整带免罚
    else:
        normal_frac = float(normal_frac)
        if not 0.0 <= normal_frac <= 1.0:
            raise ValueError("score_fringe_distribution: normal_band_frac 须在 [0,1] 内")
        allow_r, allow_c = int(round(normal_frac * rows)), int(round(normal_frac * cols))
        penalty_bands = BorderBands(
            top_px=min(bands.top_px, allow_r),
            bottom_px=min(bands.bottom_px, allow_r),
            left_px=min(bands.left_px, allow_c),
            right_px=min(bands.right_px, allow_c),
        )
    border = border_mask(arr.shape, penalty_bands)
    interior = ~border
    if not bool(interior.any()):
        raise ValueError("score_fringe_distribution: 边框带占满整图，无有效像素")

    # ② 终版背景 → 残差：只按实测带宽扣除（内部信息最大化利用）
    background = background_with_bands(
        arr, block_frac, poly_degree,
        bands.top_px, bands.bottom_px, bands.left_px, bands.right_px, block_q,
    )
    residual = arr - background

    # ③ 稳健 z（median/MAD 只在内部估计；对比度门槛防噪声放大）
    reference_scale = robust_scale(arr[interior])
    z, z_med, z_scale = robust_z_stats(
        residual, interior, float(seg_cfg["min_contrast_frac"]), reference_scale
    )

    # dev_gray 的中心与背景块统计取同一分位（block_q）：暗地板锚（低分位）下，
    # 整片亮斑的残差中位数会落在斑的水平上，用中位数居中会把斑重新归零——
    # 必须同样锚到暗侧（0.5 时即旧口径中位数）。极性无关 → 缓存进流水线。
    residual_floor = float(np.quantile(residual[interior], block_q))

    # 位置权重 w(r)：只依赖形状与 weight 段（两口径共享）
    u, v = normalized_coords(arr.shape)
    r = center_distance(u, v, str(weight_cfg["distance_norm"]))
    w = position_weight(r, weight_cfg)

    return FringePipeline(
        arr=arr, corners=corners, bands=bands, border=border, interior=interior,
        residual=residual, z=z, residual_floor=residual_floor, weight_map=w,
        reference_scale=reference_scale, z_med=z_med, z_scale=z_scale,
    )


def score_from_pipeline(pipe: FringePipeline, config: dict) -> FringeScoreResult:
    """共享流水线 → 打分（步骤④⑤）。config 只取掩膜级 segment 参数与 scoring 刻度锚。

    前提：config 的背景估计/边框带/weight 参数与生成 pipe 时一致（掩膜级参数
    method/s_scale_mode/polarity/top_fraction/min_z/min_dev_gray/饱和值可不同）——
    这正是绝对口径与位置指标口径的分叉面。
    """
    cfg = config
    seg_cfg = cfg["segment"]
    arr, interior = pipe.arr, pipe.interior

    # ④ 斑分割 + 深浅强度
    polarity = str(seg_cfg["polarity"])
    dev = deviation_map(pipe.z, polarity)
    dev_gray = deviation_map(pipe.residual - pipe.residual_floor, polarity)
    fringe, intensity_s = fringe_mask_and_intensity(dev, interior, seg_cfg, dev_gray=dev_gray)

    # ⑤ 位置加权惩罚 → 0–100 分
    w = pipe.weight_map
    penalty_raw = float((intensity_s * w).sum())           # s 在掩膜外恒为 0
    penalty_max = float(w[interior].sum())                 # 理论最差：有效区全是最深斑
    # 输出刻度锚（人工校准）：罚分比 ρ 达 penalty_ratio_at_zero → 0 分。
    # 理论最差在分位数法下不可达（掩膜 ≤ top_fraction 面积 → ρ 有 ~0.15 量级上限，
    # 分数被压在高位）；该锚把"人工判不及格"的罚分比映射到低分段，纯单调不改排序。
    ratio_at_zero = float(cfg.get("scoring", {}).get("penalty_ratio_at_zero", 1.0))
    if not 0.0 < ratio_at_zero <= 1.0:
        raise ValueError("score_fringe_distribution: penalty_ratio_at_zero 须在 (0,1] 内")
    score = 100.0 * (1.0 - penalty_raw / (penalty_max * ratio_at_zero))

    s_total = float(intensity_s.sum())
    centrality = penalty_raw / s_total if s_total > 0.0 else None
    fringe_area_frac = float(fringe.sum()) / float(interior.sum())

    return FringeScoreResult(
        score_0_100=float(np.clip(score, 0.0, 100.0)),
        penalty_raw=penalty_raw,
        penalty_max=penalty_max,
        fringe_area_frac=fringe_area_frac,
        centrality=centrality,
        border_bands=pipe.bands,
        fringe_mask=fringe,
        intensity_s=intensity_s,
        border_mask_arr=pipe.border,
        weight_map=w,
        quad_corners_px=pipe.corners,
        warped_image=arr,
    )


# 覆盖支路的覆盖判斑灰度阈 G0 默认值（2026-07-22，位置口径专属固定工程口径）：
# 独立于工厂判斑灵敏度 segment.min_dev_gray——判斑灵敏度解耦要求（2026-07-08）对
# 覆盖支路同样成立；config indicators.scoring.position_coverage.min_dev_gray 可覆写。
POSITION_COVERAGE_MIN_DEV_GRAY = 30.0


@dataclass
class CenterConcentrationResult:
    """位置指标（应力斑中心集中度）逐片测量结果：主量 ρu + 诊断量（任务3 全项）。

    指标层产物——不含 0–100 评分、不读刻度锚 ρ0；位置评分由评分层
    （indicators.compute_sheet_indicators 的评分段）按原表达式另行换算。
    weighted_coverage 为评分层覆盖度支路的测量输入（2026-07-22）：固定灰度阈
    G0 下的位置加权判斑覆盖度——逐片自标准化的 ρu 对"整片皆斑"自吞（尺度
    等变必然除掉整体纹理强度），该量在绝对灰度域补上这一维（同批曝光一致前提，
    与主口径绝对评分同前提；对逐片仿射变换**不**具备 ρu 的不变性）。
    """

    center_concentration: float   # ρu = penalty_raw / penalty_max ∈ [0,1]，越大越差
    penalty_raw: float            # Σ_{Ωu} s·w（评分层按原表达式换算位置评分用）
    penalty_max: float            # Σ_M w（理论最差总罚）
    spot_area_ratio: float        # |Ωu| / |M|——区分"斑少且靠边"与"斑多顶翻基准"
    weighted_coverage: float      # A_w = Σ_{devgray≥G0} w / Σ_M w（覆盖支路测量输入）
    threshold_t: float            # 判斑阈值 T（复算核对；序列化键仍为 threshold_T）
    quantile_bound: str           # T 由谁决定："quantile" | "min_z"
    p_dark: float                 # 暗向尾部占比（翻转判据本身）
    baseline_flipped: bool        # 基准翻转门闩是否触发且成功
    baseline_flip_aborted: bool   # 门闩触发但回退（暗侧样本不足/尺度退化）
    m_r: float                    # 实际用于 z 的稳健基准（序列化键 m_R；翻转成功时 = med(R_D)）
    sigma_r: float                # 实际用于 z 的稳健尺度（序列化键 sigma_R；退化门时为原始 σR）
    sigma_ref: float              # σref = 1.4826·MAD(I_M)（退化门参照）


def measure_center_concentration(
    pipe: FringePipeline, config: dict
) -> CenterConcentrationResult:
    """共享流水线 → 位置指标测量（④⑤ 的位置口径分叉，含基准翻转门闩）。

    config 须为 _position_indicator_config 构造的位置口径配置（per_sheet +
    quantile）。门闩由 config["baseline_flip"] 段控制（缺省=启用）；未触发时
    对 pipe.z 零操作，praw/pmax 与 score_from_pipeline 同一输入逐算子同序
    ——无翻转片与旧实现位级相同。
    """
    cfg = config
    seg_cfg = cfg["segment"]
    interior = pipe.interior

    # 基准翻转门闩（Pass1 = pipe.z 现行逻辑；触发才用暗侧子总体重算 z）
    flip = baseline_flip_z(
        pipe.residual, interior, pipe.z, float(seg_cfg["min_z"]),
        float(seg_cfg["min_contrast_frac"]), pipe.reference_scale,
        pipe.z_med, pipe.z_scale, cfg.get("baseline_flip"),
    )

    # ④ 斑分割 + 强度（与 score_from_pipeline 同式；T 在翻转后的 dev 上重新求分位）
    polarity = str(seg_cfg["polarity"])
    dev = deviation_map(flip.z, polarity)
    dev_gray = deviation_map(pipe.residual - pipe.residual_floor, polarity)
    mask, intensity_s, info = fringe_mask_intensity_info(
        dev, interior, seg_cfg, dev_gray=dev_gray
    )

    # ⑤ 位置加权罚分比（不换算分数——ρ0 属评分层）
    w = pipe.weight_map
    penalty_raw = float((intensity_s * w).sum())           # s 在掩膜外恒为 0
    penalty_max = float(w[interior].sum())                 # 理论最差：有效区全是最深斑

    # 位置加权覆盖度（评分层覆盖支路的测量输入）：固定灰度阈 G0 判"绝对域斑"，
    # 位置权重加权求占比——斑压中心计得重；G0 独立于工厂判斑灵敏度（见常数注释）
    cov_cfg = cfg.get("position_coverage") or {}
    g0 = float(cov_cfg.get("min_dev_gray", POSITION_COVERAGE_MIN_DEV_GRAY))
    cov_mask = interior & (dev_gray >= g0)
    weighted_coverage = float(w[cov_mask].sum()) / penalty_max

    return CenterConcentrationResult(
        center_concentration=penalty_raw / penalty_max,
        penalty_raw=penalty_raw,
        penalty_max=penalty_max,
        spot_area_ratio=float(mask.sum()) / float(interior.sum()),
        weighted_coverage=weighted_coverage,
        threshold_t=float(info["threshold"]),
        quantile_bound=str(info["bound"]),
        p_dark=flip.p_dark,
        baseline_flipped=flip.flipped,
        baseline_flip_aborted=flip.aborted,
        m_r=flip.m_r,
        sigma_r=flip.sigma_r,
        sigma_ref=pipe.reference_scale,
    )


def score_fringe_distribution(
    image: np.ndarray, config: dict | None = None, quad_corners: np.ndarray | None = None
) -> FringeScoreResult:
    """一张玻璃图 → 应力斑分布打分（流水线见模块 docstring 与 docs §2）。

    = compute_pipeline（⓪–③）+ score_from_pipeline（④⑤）；对外签名与行为不变。
    需要在同一片上叠加第二套掩膜口径（如六指标的位置指标）时，持有 pipeline
    分别打分即可，免整条流水线重跑。
    """
    cfg = config if config is not None else load_config()
    return score_from_pipeline(compute_pipeline(image, cfg, quad_corners), cfg)

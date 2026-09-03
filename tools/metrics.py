"""钢化玻璃应力斑质量指标：评估区域掩膜 + X0.95 / IsoT / CCP。

严格依据 docs/《钢化玻璃应力斑分级及检测方法》草案2026.5.18(1).pdf（GB/T 草案，
即原挂账的"docs/01"主体；正式版发布后须复核）。分级限值在 config/grading.yaml，
CCP 参考常量在 config/ccp_reference.yaml，禁止把限值硬编码进逻辑。

另含**提案指标** W_w（位置加权组合纹理指数，2026-07-30，见 texture_w()）——非现行
草案条文。零标定常数：保留 CCP 的公式形状与两个纹理分量，把需要参考样标定的分母
Cmax、CPmax 换成各自的纯数学理论上界（Ca ≤ (Ng−1)²、CPa ≤ (2(Ng−1))⁴/12），
并注入纯几何位置权重 w(r)=1−r。（前身 CCP_pos 工程已于 2026-07-30 整体废弃删除；CP_pos 之名亦于 2026-08-04 更为 W_w。）

约定：
- 几何单位 mm，光程差/延迟量单位 nm。
- 掩膜 M = 整板 扣除 边缘带 E 与 孔洞 H；所有指标仅在 M 内统计。
- 图像 shape=(rows, cols)；约定 cols 沿长度 L、rows 沿宽度 W。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_GRADING = _CONFIG_DIR / "grading.yaml"
_CCP_REF = _CONFIG_DIR / "ccp_reference.yaml"


def _is_todo(v) -> bool:
    """判断 config 值是否为缺值占位（None 或 TODO 开头字符串）→ 按"无法判定"处理。"""
    return v is None or (isinstance(v, str) and v.strip().upper().startswith("TODO"))


def _load_yaml(path: Path) -> dict:
    """按需读取 yaml 配置（每次调用重新读盘，改 yaml 即时生效）。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------- #
# 评估区域几何（docs/01 §2）
# --------------------------------------------------------------------------- #
def edge_band_mm(L_mm: float, W_mm: float, thickness_mm: float) -> tuple[float, float]:
    """边缘带宽度 (La 沿长度, Wa 沿宽度)，单位 mm。

    La=10%L、Wa=10%W；下限 50mm；上限随厚度：≤8mm→200mm，≥10mm→350mm。
    8<t<10（如 9mm）暂按 350 处理，真实分界待 docs/01 确认 → TODO(plant)。
    """
    cap = 200.0 if thickness_mm <= 8 else 350.0   # TODO(plant): 8<t<10 分界待确认
    la = min(max(0.10 * L_mm, 50.0), cap)
    wa = min(max(0.10 * W_mm, 50.0), cap)
    return la, wa


def hole_exclusion_radius_mm(thickness_mm: float, hole_radius_mm: float) -> float:
    """孔洞扣除半径 = 6×厚度 + 孔半径（mm）。"""
    return 6.0 * thickness_mm + hole_radius_mm


def build_mask(
    shape: tuple[int, int],
    mm_per_px: float,
    thickness_mm: float,
    L_mm: float | None = None,
    W_mm: float | None = None,
    holes: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """生成评估区域布尔掩膜 M（True=纳入统计）。

    holes: [(cx_mm, cy_mm, hole_radius_mm), ...]，按 hole_exclusion_radius_mm 扣除。
    """
    rows, cols = shape
    L = L_mm if L_mm is not None else cols * mm_per_px
    W = W_mm if W_mm is not None else rows * mm_per_px
    la, wa = edge_band_mm(L, W, thickness_mm)

    mask = np.ones(shape, dtype=bool)
    la_px = int(round(la / mm_per_px))
    wa_px = int(round(wa / mm_per_px))
    if wa_px > 0:
        mask[:wa_px, :] = False
        mask[rows - wa_px:, :] = False
    if la_px > 0:
        mask[:, :la_px] = False
        mask[:, cols - la_px:] = False

    if holes:
        yy, xx = np.ogrid[:rows, :cols]
        for cx_mm, cy_mm, r_mm in holes:
            R_px = hole_exclusion_radius_mm(thickness_mm, r_mm) / mm_per_px
            cx_px = cx_mm / mm_per_px
            cy_px = cy_mm / mm_per_px
            mask[(xx - cx_px) ** 2 + (yy - cy_px) ** 2 <= R_px ** 2] = False
    return mask


# --------------------------------------------------------------------------- #
# 三指标
# --------------------------------------------------------------------------- #
def x0_95(retardation_nm_masked: np.ndarray) -> float:
    """M 内光程差的 95% 分位（nm）。值越小越好。"""
    arr = np.asarray(retardation_nm_masked, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("x0_95: 掩膜内无有效像素")
    return float(np.percentile(arr, 95))


def iso_t(retardation_nm_masked: np.ndarray, T: float = 75.0) -> float:
    """M 内光程差 < T 的面积占比（%）。占比越大越好（判级为'≥'方向）。"""
    arr = np.asarray(retardation_nm_masked, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("iso_t: 掩膜内无有效像素")
    return float(np.mean(arr < T) * 100.0)


@dataclass
class CcpResult:
    """CCP 计算结果。is_calibrated=False 表示用的是合成/注入参考，未现场标定，不得当真值下发。"""

    value: float
    ca: float            # GLCM 对比度（四方向平均）
    cpa: float           # GLCM 聚类突出 cluster prominence（四方向平均）
    is_calibrated: bool


def _standardize_to_1px_per_mm(image: np.ndarray, mm_per_px: float) -> np.ndarray:
    """草案 §4.4 b) 图像标准化：重采样到 1 px/mm（像素距离归一到真实距离）。

    参考面积 10000 px² ↔ 100 mm×100 mm，即标准化后 1 px = 1 mm——保证不同尺寸/
    分辨率玻璃的 CCP 可横向比较；此后 GLCM 步距 d=1 即对应真实 1 mm。
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("ccp: 需要二维光程差图像（评估区域），非一维数组")
    if mm_per_px <= 0:
        raise ValueError("ccp: mm_per_px 须为正")
    if abs(mm_per_px - 1.0) < 1e-12:
        return arr
    from skimage.transform import resize

    rows = max(2, int(round(arr.shape[0] * mm_per_px)))
    cols = max(2, int(round(arr.shape[1] * mm_per_px)))
    # mm_per_px<1（分辨率高于 1px/mm）时是降采样，需抗混叠
    return resize(arr, (rows, cols), order=1, preserve_range=True, anti_aliasing=mm_per_px < 1.0)


def _require_finite(arr: np.ndarray, who: str) -> None:
    """输入必须全为有限值（NaN/Inf → 抛错）。

    缺值不放行：NaN 会让 min/max 量化的 lo/hi 变 NaN，astype(uint8) 把 NaN 静默变 0，
    结果是"完美无斑"的最优值——掩膜后的光程差图一旦带 NaN 就会命中（2026-07-30 复核发现）。
    """
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{who}: 输入含 NaN/Inf，拒绝出值（掩膜请用布尔索引而非填 NaN）")


def _quantize_minmax(arr: np.ndarray, ng: int) -> np.ndarray:
    """逐片 min/max 线性量化到 Ng 级（uint8）——CCP 血统口径（草案 §4.4）。

    CCP 与提案指标 W_w 共用本量化（同 lo/hi、同 Ng）→ 建材同口径、数值血统可比。
    ⚠️ 该口径对逐片仿射变换不变（幅度盲，深浅由 X0.95/IsoT 分工承担）；近平坦图的
    "微幅平滑渐变"会被放大——由 texture_w 的最小动态范围条款（可选）兜底，iid 噪声
    则由 CP 特征自身钝感兜住（ccp() 保持草案原状不动）。
    """
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        q = np.zeros(arr.shape, dtype=np.uint8)           # 常量图：无纹理
    else:
        q = np.floor((arr - lo) / (hi - lo) * (ng - 1) + 0.5).astype(np.uint8)
    return np.clip(q, 0, ng - 1)


def _glcm_ca_cpa(image: np.ndarray, ng: int = 8) -> tuple[float, float]:
    """二维光程差图像 → (Ca, CPa)：GLCM 四方向(0/45/90/135°)平均的对比度与聚类突出。

    步距 d=1（输入应已标准化到 1 px/mm，故 d=1 ≡ 真实 1 mm）；线性量化到 Ng 级。
    """
    from skimage.feature import graycomatrix, graycoprops

    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("ccp: 需要二维光程差图像（评估区域），非一维数组")
    q = _quantize_minmax(arr, ng)

    angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]    # 0 / 45 / 90 / 135°
    glcm = graycomatrix(q, distances=[1], angles=angles, levels=ng, symmetric=True, normed=True)

    ca = float(graycoprops(glcm, "contrast").mean())      # 四向平均
    cpa = float(_cluster_prominence(glcm).mean())          # 四向平均
    return ca, cpa


def _cluster_prominence(glcm: np.ndarray) -> np.ndarray:
    """各 (距离,角度) 的聚类突出 CP = Σ (i+j-μx-μy)^4 · P(i,j)；返回展平后的逐方向值。"""
    levels = glcm.shape[0]
    i = np.arange(levels).reshape(-1, 1, 1, 1)
    j = np.arange(levels).reshape(1, -1, 1, 1)
    mu_i = (i * glcm).sum(axis=(0, 1))                     # μx，shape=(n_dist, n_angle)
    mu_j = (j * glcm).sum(axis=(0, 1))                     # μy
    cp = (((i + j - mu_i - mu_j) ** 4) * glcm).sum(axis=(0, 1))
    return cp.reshape(-1)


def ccp(retardation_nm_masked: np.ndarray, mm_per_px: float, ref: dict | None = None) -> CcpResult:
    """组合纹理特征 CCP（草案 §4.4）。

    - 步骤 b) 图像标准化：先重采样到 1 px/mm（10000 px² ↔ 100 mm 口径），跨尺寸可比；
    - GLCM 灰度级 Ng=8、四方向(0/45/90/135°)平均；
    - 公式(1)：CCP = 0.5 * ( sqrt(Ca / Cmax) + (CPa / CPmax) ** 0.25 )；
    - Cmax / CPmax 取自"参考最差样品"——**草案未给数值，标定值仍缺**。

    ``ref``：可注入参考 ``{"c_max":…, "cp_max":…}``（测试用合成参考，结果 is_calibrated=False）。
    ``ref=None`` 时读 config/ccp_reference.yaml；其值为 TODO(plant) 未标定 → 抛 NotImplementedError，
    **不当真值下发**（规则 > AI）。
    """
    standardized = _standardize_to_1px_per_mm(retardation_nm_masked, mm_per_px)
    ca, cpa = _glcm_ca_cpa(standardized, ng=8)

    if ref is None:
        ref = _load_yaml(_CCP_REF)
        if _is_todo(ref.get("c_max")) or _is_todo(ref.get("cp_max")):
            raise NotImplementedError(
                "CCP 未标定：config/ccp_reference.yaml 的 Cmax/CPmax 为 TODO(plant)，不当真值下发。"
                "测试可注入合成 ref={'c_max':…, 'cp_max':…}（结果 is_calibrated=False）。"
            )
        is_calibrated = True
    else:
        is_calibrated = False

    c_max = float(ref["c_max"])
    cp_max = float(ref["cp_max"])
    value = 0.5 * ((ca / c_max) ** 0.5 + (cpa / cp_max) ** 0.25)
    return CcpResult(value=value, ca=ca, cpa=cpa, is_calibrated=is_calibrated)


# --------------------------------------------------------------------------- #
# W_w（提案指标：位置加权聚类突出指数 —— 非现行草案条文，零标定常数）
# --------------------------------------------------------------------------- #
# 设计（2026-07-30 用户拍板）：彻底抛弃 CCP 中依赖参考样标定的统计量（Cmax/CPmax
# 及同类），但**保留两个纹理分量**：Ca、CPa 各以自身的纯数学上界归一
# （(Ng−1)²=49 与 (2(Ng−1))⁴/12≈3201.33），位置权重 w(r)=1−r 纯几何零常数。
# 全链路除结构常数（Ng=8、d=1、四方向、几何定义）外无任何可调/待标定数值。
# ⚠️ 2026-07-30 对抗复核更正两处早期错判：
#   ① "Ca 单独 Spearman 仅 -0.635" 系张冠李戴（-0.649 实为**未加权** CPa）；
#      同口径实测 Ca_w 单独 -0.865，与 CPa_w 的 -0.878 相当，**并不弱**；
#   ② 因而"为零标定常数必须弃 Ca"不成立——Ca 也有纯数学上界。按实效评比 7 种
#      候选形式后改回双分量组合（-0.886，显著优于仅 CPa）。详见 texture_w docstring。

# GLCM 四方向 d=1 的 (行偏移, 列偏移)：0°/45°/90°/135°——与 skimage graycomatrix 的
# angles=[0, π/4, π/2, 3π/4] 四角度均值同口径（全 1 权重下与 skimage 路径的 parity
# 由 tests/test_metrics.py::test_texture_w_uniform_weight_parity_with_skimage_path 钉死）
_GLCM_POS_OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))


def _ca_sup(ng: int) -> float:
    """对比度 Ca 的纯数学理论上界 = (ng−1)²（替代参考样 Cmax 的归一化分母）。

    推导：Ca = E[(i−j)²]，i、j ∈ {0,…,ng−1} ⇒ |i−j| ≤ ng−1，故 Ca ≤ (ng−1)²，
    等号由全部共生对落在 (0, ng−1) / (ng−1, 0) 时取得（逐列交替图的 0° 方向即达此值；
    四方向平均后严格小于上界）。ng=8 → 49。
    """
    return float((ng - 1) ** 2)


def _cp_sup(ng: int) -> float:
    """聚类突出 CP 的纯数学理论上界 = (2(ng−1))⁴ / 12（替代参考样 CPmax 的归一化分母）。

    推导：CP = E[(S−E S)⁴]，S = i+j 支撑于 [0, 2(ng−1)]。有界区间 [a,b] 上四阶
    中心矩的上确界 = (b−a)⁴/12，由端点两点分布取得：质量 p 置于 b、(1−p) 置于 a，
    m4 = (b−a)⁴·p(1−p)·[1−3p(1−p)]，令 t=p(1−p) 得 g(t)=t(1−3t) 在 t=1/6 取最大 1/12，
    对应 p=(1−√(1/3))/2≈0.2113（数值扫描验证吻合，2026-07-30）。加权 GLCM 归一化后
    仍是同支撑上的概率分布 → 上界同样适用；真实连通图像只能渐近逼近 → 值域 [0,1)。
    """
    return float((2 * (ng - 1)) ** 4) / 12.0


def _position_weight_map(shape: tuple[int, int]) -> np.ndarray:
    """评估区图像 shape → 位置权重图 w(r) = 1 − r（W_w 提案定义：纯几何零常数）。

    归一化坐标：u = 2(col+0.5)/cols − 1、v = 2(row+0.5)/rows − 1（像素中心口径，
    各维独立归一 → 等权线为与评估区同形的同心矩形，与纵横比无关）；
    r = max(|u|,|v|)（chebyshev）。评估区中心 w=1，向边界线性降至 ~0——边缘深斑
    由位置盲的 X0.95/IsoT 分工兜底（三指标分工），此处不设下限、不引入任何常数。
    像素中心离散化使 r < 1 严格成立 → w > 0，无全零权重风险。
    （工厂位置评分的 w 带 0.3 下限，是另一层的工程口径，二者自 2026-07-30 起分离。）
    """
    rows, cols = shape
    u = (2.0 * (np.arange(cols) + 0.5) / cols - 1.0).reshape(1, -1)
    v = (2.0 * (np.arange(rows) + 0.5) / rows - 1.0).reshape(-1, 1)
    r = np.maximum(np.abs(u), np.abs(v))
    return 1.0 - r


def _glcm_ca_cpa_weighted(
    image: np.ndarray, weight_map: np.ndarray, ng: int = 8
) -> tuple[float, float]:
    """位置加权 GLCM → (Ca_w, CPa_w)，四方向平均（W_w 只消费 CPa_w）。

    与 _glcm_ca_cpa 同量化（_quantize_minmax）、同特征公式；唯一区别：每个共生对
    (p, p+Δ) 不再计 1，而按端点权重的算术平均 pair_w = ½(w_p + w_{p+Δ}) 累加。
    - pair_w 对 p↔p+Δ 交换不变 → 与 counts+counts.T 对称化相容；
    - 均匀权重 w≡c 时归一化把 c 抵消 → 退化回未加权 GLCM（Ca_w=Ca、CPa_w=CPa）；
      空间均匀纹理下同理 W_w 退化为未加权 CP 指数（兼容性性质，单测钉死）。
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("texture_w: 需要二维光程差图像（评估区域），非一维数组")
    w = np.asarray(weight_map, dtype=float)
    if w.shape != arr.shape:
        raise ValueError("texture_w: weight_map 形状须与图像一致")
    q = _quantize_minmax(arr, ng)
    rows, cols = arr.shape

    i_idx = np.arange(ng).reshape(-1, 1)
    j_idx = np.arange(ng).reshape(1, -1)
    ca_list, cpa_list = [], []
    for dr, dc in _GLCM_POS_OFFSETS:
        # 依偏移取像素对 (a, b) 与端点权重 (wa, wb)；bincount(weights=pair_w) 按
        # 扁平索引 i·Ng+j 加权累加（逐对累加的向量化等价）
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        a = q[r0:r1, c0:c1].ravel().astype(np.intp)
        b = q[r0 + dr: r1 + dr, c0 + dc: c1 + dc].ravel().astype(np.intp)
        wa = w[r0:r1, c0:c1].ravel()
        wb = w[r0 + dr: r1 + dr, c0 + dc: c1 + dc].ravel()
        pair_w = 0.5 * (wa + wb)
        counts = np.bincount(a * ng + b, weights=pair_w, minlength=ng * ng).reshape(ng, ng)
        counts = counts + counts.T                        # symmetric=True，同 graycomatrix 口径
        total = counts.sum()
        if total <= 0:
            raise ValueError("texture_w: 图像过小或权重全零，加权 GLCM 无有效像素对")
        p = counts / total
        ca_list.append(float((p * (i_idx - j_idx) ** 2).sum()))
        mu_i = float((i_idx * p).sum())
        mu_j = float((j_idx * p).sum())
        cpa_list.append(float((p * (i_idx + j_idx - mu_i - mu_j) ** 4).sum()))
    return float(np.mean(ca_list)), float(np.mean(cpa_list))


@dataclass
class TextureWResult:
    """W_w 计算结果。零标定常数指标——无 is_calibrated 字段（无可标定项）。"""

    value: float             # W = ½(√(Ca_w/49) + ⁴√(CPa_w/3201.33)) ∈ [0,1)，越大越差
    ca_w: float              # 位置加权 GLCM 对比度（四方向平均，归一化前原始值）
    cpa_w: float             # 位置加权 GLCM 聚类突出（四方向平均，归一化前原始值）
    dynamic_range_nm: float  # 标准化后 max−min（最小动态范围条款的判据，恒输出供诊断）
    degenerate: bool         # True=触发最小动态范围条款（判"无纹理"，value=0）


def texture_w(
    retardation_nm_masked: np.ndarray, mm_per_px: float, ref: dict | None = None
) -> TextureWResult:
    """位置加权组合纹理指数 W_w（**提案指标，非现行草案条文；零标定常数**）。

    曾名 CP_pos，2026-08-04 更名：多字母名无法写进公式，且易误读为
    "聚类突出 CP 的位置变体"——实为 Ca+CPa 组合指数。下标 w=位置加权，
    与分量记号 Ca,w / CPa,w 同一家族签名。旧引用兼容别名见文件尾。

    设计定位（2026-07-30）：保留现行 CCP 的公式形状与两个纹理分量，**只把两个需要
    参考样标定的分母 Cmax、CPmax 替换为各自的纯数学理论上界**，并注入位置权重。
    除结构常数（Ng=8、d=1、四方向、几何定义）外无任何可调/待标定数值——任何实验室
    对同一输入算出同一个数。

    形式选定依据（2026-07-30 按实效评比 7 种零标定候选形式，115 片人工标注集；
    2026-08-04 全部秩相关改**并列平均秩**口径重算——人工分 5 分一档、99.1% 样本在
    并列组内，旧 argsort 口径数字作废。权威数值=make_texture_w_assets.py 每次运行
    真复算落盘的 data/derived/texture_w_doc/values.json form_comparison）：
    - 本形式（双上界组合）vs 人工分 -0.894，显著优于仅 CPa 的 -0.883
      （bootstrap 2000 次配对 Δ|ρ|=+0.0102，95% CI [+0.0033, +0.0206] 不含 0）；
    - 仅 Ca 点估计最高（-0.923）但与仅 CPa 的差异不显著（Δ|ρ|=+0.0393，
      CI [-0.0142, +0.1016] 含 0）且有病态（合成条纹图上顺序反转），不采用；
    - 位置加权是最大贡献项：未加权 CPa 仅 -0.641 → 加权 -0.883（+0.24），
      远超任意两种特征组合形式之间的差异；
    - 两分量在病态上互补：Ca 抗"平滑全幅渐变"但对随机噪声敏感，CPa 反之。
      组合后两类极端图案均落在产线分布下三分之一（近平坦微噪 23 分位、
      iid 强噪 32 分位），且近平坦一类另有最小动态范围条款兜底。

    链路：
    - 图像标准化：重采样 1 px/mm（复用 _standardize_to_1px_per_mm）；
    - 最小动态范围条款（可选物理守卫）：标准化后 max−min < ε → 判"无纹理"，
      value=0（最优方向）。ε 为仪器精度性质常数（非参考样统计量，不在弃用之列），
      读 ref/config 的 min_dynamic_range_nm；未定值（TODO(plant)）→ 条款不激活；
    - 量化：逐片 min/max 线性 Ng=8（_quantize_minmax，仿射不变=零灰度标定依赖）；
    - 位置权重：w(r)=1−r 纯几何（_position_weight_map，零常数）；
    - 加权 GLCM：pair_w=½(w_p+w_{p+Δ}) 累加 → 取 Ca_w 与 CPa_w；
    - 归一：value = ½(√(Ca_w/_ca_sup(8)) + ⁴√(CPa_w/_cp_sup(8)))——与现行 CCP 公式
      逐算子同形（½、√、⁴√ 及结合次序全同），仅两个分母换成理论上界。
      值域 [0,1)：两分支各 ≤1 故半和 ≤1；四方向平均使各分支严格 <1（某方向达上界
      要求该方向所有相邻对同色/极端配对，四方向同时成立即退化为常量图，矛盾）。

    ``ref``：仅可注入 ``{"min_dynamic_range_nm": …}``（测试/演算用）；``ref=None``
    读 config/ccp_reference.yaml。**本函数不因缺常数拒绝出值——它没有待标定常数**。
    分级限值（config/grading.yaml 的 texture_w 表）仍属经验划线，未建立前 grade 返回 None。
    """
    _require_finite(np.asarray(retardation_nm_masked, dtype=float), "texture_w")
    standardized = _standardize_to_1px_per_mm(retardation_nm_masked, mm_per_px)
    if ref is None:
        ref = _load_yaml(_CCP_REF)

    # 最小动态范围条款（可选守卫；仅挂 W_w，ccp() 保持草案原状）
    dyn = float(np.max(standardized) - np.min(standardized))
    eps = ref.get("min_dynamic_range_nm")
    if not _is_todo(eps) and dyn < float(eps):
        return TextureWResult(value=0.0, ca_w=0.0, cpa_w=0.0,
                           dynamic_range_nm=dyn, degenerate=True)

    w_map = _position_weight_map(standardized.shape)
    ca_w, cpa_w = _glcm_ca_cpa_weighted(standardized, w_map, ng=8)
    # 与 ccp() 逐算子同形同序（0.5 * (√ + ⁴√)），仅分母为理论上界
    value = 0.5 * ((ca_w / _ca_sup(8)) ** 0.5 + (cpa_w / _cp_sup(8)) ** 0.25)
    return TextureWResult(value=value, ca_w=ca_w, cpa_w=cpa_w,
                       dynamic_range_nm=dyn, degenerate=False)


# --------------------------------------------------------------------------- #
# 分级
# --------------------------------------------------------------------------- #
def _pick_row(table: list, thickness_mm: float, required_key: str):
    """取 thickness_mm ≥ 查询厚度的最小一行（行值=公称厚度档"适用 ≤该值"）。

    查无适用行（如 >15mm，草案注明"由供需双方商定"）或该行限值为 TODO → None。
    """
    rows = sorted((r for r in table if not _is_todo(r.get(required_key))),
                  key=lambda r: r["thickness_mm"])
    return next((r for r in rows if r["thickness_mm"] >= thickness_mm), None)


def grade(value: float, thickness_mm: float, method: str, grading: dict | None = None):
    """按 config/grading.yaml 给指标值判级，返回 'A'/'B'/'C'。

    判级方向（草案表1/2/3）：x0_95 / ccp / texture_w 越小越好（≤A_max→A）；
    iso_t 占比越大越好（≥A_min→A）。
    缺值（TODO(plant) 或查不到适用行）→ 返回 None（无法判级），不猜测。
    method: 'x0_95' | 'iso_t' | 'ccp' | 'texture_w'（提案指标，限值表待工作组建立）。
    """
    g = grading if grading is not None else _load_yaml(_GRADING)
    key = {"x0_95": "x0_95_nm", "iso_t": "iso_t_pct", "ccp": "ccp",
           "texture_w": "texture_w"}.get(method)
    if key is None:
        raise ValueError(f"grade: 未知 method {method!r}")
    table = g.get(key)
    if _is_todo(table):
        return None     # 该方法分级表缺失
    assert table is not None  # _is_todo 已排除 None/TODO

    if method in ("x0_95", "ccp", "texture_w"):
        # 越小越好："≤"方向（草案表1/表3；texture_w 同 ccp 方向）
        row = _pick_row(table, thickness_mm, "A_max")
        if row is None:
            return None
        if value <= row["A_max"]:
            return "A"
        if value <= row["B_max"]:
            return "B"
        return "C"

    # iso_t：占比越大越好，"≥"方向（草案表2，阈值 T=75nm）
    row = _pick_row(table, thickness_mm, "A_min")
    if row is None:
        return None
    if value >= row["A_min"]:
        return "A"
    if value >= row["B_min"]:
        return "B"
    return "C"


# ── 更名兼容（2026-08-04：CP_pos → W_w / texture_w）：旧调用点不断链 ──
CpPosResult = TextureWResult
cp_pos = texture_w

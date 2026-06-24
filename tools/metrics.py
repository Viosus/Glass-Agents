"""钢化玻璃应力斑质量指标：评估区域掩膜 + X0.95 / IsoT / CCP。

严格依据 docs/01-objective-and-metrics.md（暂缺）。分级限值在 config/grading.yaml，
CCP 参考常量在 config/ccp_reference.yaml，禁止把限值硬编码进逻辑。

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
    return v is None or (isinstance(v, str) and v.strip().upper().startswith("TODO"))


def _load_yaml(path: Path) -> dict:
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


def _glcm_ca_cpa(image: np.ndarray, ng: int = 8) -> tuple[float, float]:
    """二维光程差图像 → (Ca, CPa)：GLCM 四方向(0/45/90/135°)平均的对比度与聚类突出。

    步距 d=1（真实步距对应物理尺度待 docs/01 → TODO(plant)）；线性量化到 Ng 级。
    """
    from skimage.feature import graycomatrix, graycoprops

    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("ccp: 需要二维光程差图像（评估区域），非一维数组")

    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        q = np.zeros(arr.shape, dtype=np.uint8)           # 常量图：无纹理
    else:
        q = np.floor((arr - lo) / (hi - lo) * (ng - 1) + 0.5).astype(np.uint8)
    q = np.clip(q, 0, ng - 1)

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
    """聚类对比度参数 CCP（方法 4.4）。

    - GLCM 灰度级 Ng=8、四方向(0/45/90/135°)平均；
    - 公式 CCP = 0.5 * ( sqrt(Ca / Cmax) + (CPa / CPmax) ** 0.25 )；
    - Cmax / CPmax 取自"参考最差样品"（归一化基准 10000 px² / 100 mm）。

    ``ref``：可注入参考 ``{"c_max":…, "cp_max":…}``（测试用合成参考，结果 is_calibrated=False）。
    ``ref=None`` 时读 config/ccp_reference.yaml；其值为 TODO(plant) 未标定 → 抛 NotImplementedError，
    **不当真值下发**（规则 > AI）。归一化基准的精确口径待 docs/01 → TODO(plant)。
    """
    ca, cpa = _glcm_ca_cpa(retardation_nm_masked, ng=8)

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
# 分级
# --------------------------------------------------------------------------- #
def grade(value: float, thickness_mm: float, method: str, grading: dict | None = None):
    """按 config/grading.yaml 给指标值判级，返回 'A'/'B'/'C'。

    缺值（TODO(plant) 或查不到适用行）→ 返回 None（无法判级），不猜测。
    method: 'x0_95' | 'iso_t' | 'ccp'。
    """
    g = grading if grading is not None else _load_yaml(_GRADING)
    key = {"x0_95": "x0_95_nm", "iso_t": "iso_t_pct", "ccp": "ccp"}.get(method)
    if key is None:
        raise ValueError(f"grade: 未知 method {method!r}")
    table = g.get(key)
    if _is_todo(table):
        return None     # 该方法分级表缺失（待 docs/01）
    assert table is not None  # _is_todo 已排除 None/TODO

    if method == "x0_95":
        # 取 thickness_mm >= 查询厚度的最小一行
        rows = sorted((r for r in table if not _is_todo(r.get("A_max"))),
                      key=lambda r: r["thickness_mm"])
        row = next((r for r in rows if r["thickness_mm"] >= thickness_mm), None)
        if row is None:
            return None
        if value <= row["A_max"]:
            return "A"
        if value <= row["B_max"]:
            return "B"
        return "C"

    # iso_t / ccp 分级表当前为 TODO(plant)，到此说明结构已就绪但数值待补
    return None

"""tools/metrics.py 的指标/掩膜单测（覆盖临界值）。"""

import numpy as np
import pytest

from tools import metrics


# --------------------------- X0.95 + 判级 --------------------------- #
def test_x0_95_value():
    assert metrics.x0_95(np.full(100, 65.0)) == pytest.approx(65.0)


def test_x0_95_grade_boundaries_6mm():
    # 草案表1 6mm 档：A_max=70, B_max=95
    assert metrics.grade(70.0, 6, "x0_95") == "A"      # 边界 ≤70 → A
    assert metrics.grade(65.0, 6, "x0_95") == "A"
    assert metrics.grade(95.0, 6, "x0_95") == "B"      # 70<95≤95 → B
    assert metrics.grade(95.01, 6, "x0_95") == "C"     # >95 → C


def test_x0_95_grade_boundaries_8mm():
    # 草案表1 8mm 档：A_max=80, B_max=120（表已由草案 PDF 补全）
    assert metrics.grade(80.0, 8, "x0_95") == "A"
    assert metrics.grade(120.0, 8, "x0_95") == "B"
    assert metrics.grade(120.01, 8, "x0_95") == "C"


def test_grade_over_15mm_returns_none():
    # >15mm 草案注"由供需双方商定" → 查无适用行 → 无法判级
    assert metrics.grade(50.0, 16, "x0_95") is None
    assert metrics.grade(0.5, 16, "ccp") is None


# --------------------------- IsoT --------------------------- #
def test_iso_t_value():
    # IsoT 数值：30% 像素低于阈值 75nm → 30.0%
    arr = np.concatenate([np.full(30, 50.0), np.full(70, 100.0)])
    assert metrics.iso_t(arr, T=75.0) == pytest.approx(30.0)


def test_iso_t_grade_boundaries_6mm():
    # 草案表2 6mm 档（"≥"方向）：A_min=95, B_min=85
    assert metrics.grade(95.0, 6, "iso_t") == "A"      # 边界 ≥95 → A
    assert metrics.grade(94.9, 6, "iso_t") == "B"
    assert metrics.grade(85.0, 6, "iso_t") == "B"      # 边界 ≥85 → B
    assert metrics.grade(84.9, 6, "iso_t") == "C"


def test_iso_t_grade_15mm_returns_none():
    # 表2 "≥15mm 由供需双方商定" → 无 15 行 → 无法判级
    assert metrics.grade(90.0, 15, "iso_t") is None


# --------------------------- 评估区域几何 --------------------------- #
def test_edge_band_upper_cap_by_thickness():
    # 10%*10000=1000 → 受厚度上限钳制：≤8→200，≥10→350
    assert metrics.edge_band_mm(10000, 10000, 8) == (200.0, 200.0)
    assert metrics.edge_band_mm(10000, 10000, 10) == (350.0, 350.0)
    assert metrics.edge_band_mm(10000, 10000, 15) == (350.0, 350.0)


def test_edge_band_lower_floor():
    # 10%*100=10 → 受下限 50 钳制
    assert metrics.edge_band_mm(100, 100, 6) == (50.0, 50.0)


def test_edge_band_within_range():
    # 10%*800=80，介于 50~200 之间 → 取 80
    assert metrics.edge_band_mm(800, 800, 6) == (80.0, 80.0)


def test_hole_exclusion_radius():
    # 孔洞排除半径 = 6×厚度 + 孔半径（mm）
    assert metrics.hole_exclusion_radius_mm(6, 5) == 41.0   # 6*6+5
    assert metrics.hole_exclusion_radius_mm(10, 0) == 60.0  # 6*10+0


# --------------------------- 掩膜构建 --------------------------- #
def test_build_mask_excludes_border():
    # 200x200px，1mm/px，厚 6mm，10% 边带=20mm=20px（介于 50? 不，10%*200=20 < 50 → 取下限 50）
    mask = metrics.build_mask((200, 200), mm_per_px=1.0, thickness_mm=6)
    assert mask.shape == (200, 200)
    assert not mask[0, 0]             # 角落在边带内
    assert mask[100, 100]             # 中心保留
    assert mask.sum() < mask.size     # 确有扣除


# --------------------------- CCP（方法 4.4） --------------------------- #
def test_ccp_default_config_uncalibrated_raises():
    # 默认 config 的 Cmax/CPmax 为 TODO(plant) → 不当真值下发，抛 NotImplementedError（安全保证）
    rng = np.random.default_rng(0)
    img = rng.integers(0, 100, size=(32, 32)).astype(float)
    with pytest.raises(NotImplementedError):
        metrics.ccp(img, mm_per_px=1.0)


def test_ccp_synthetic_ref_computes_value():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 100, size=(32, 32)).astype(float)
    ref = {"c_max": 10.0, "cp_max": 1000.0}     # 合成参考（未标定）
    res = metrics.ccp(img, mm_per_px=1.0, ref=ref)
    assert res.is_calibrated is False           # 显式标注未标定，不当真值
    assert res.ca >= 0.0 and res.cpa >= 0.0
    # 公式自洽：CCP = 0.5*(sqrt(Ca/Cmax)+(CPa/CPmax)**0.25)
    expected = 0.5 * ((res.ca / 10.0) ** 0.5 + (res.cpa / 1000.0) ** 0.25)
    assert res.value == pytest.approx(expected)


def test_ccp_flat_image_has_zero_contrast():
    # 常量图无纹理 → Ca=0（四方向平均）
    res = metrics.ccp(np.full((16, 16), 42.0), mm_per_px=1.0, ref={"c_max": 10.0, "cp_max": 1000.0})
    assert res.ca == pytest.approx(0.0)


def test_ccp_requires_2d():
    # CCP 需要二维光程差图像：一维数组应被拒绝
    with pytest.raises(ValueError):
        metrics.ccp(np.zeros(100), mm_per_px=1.0, ref={"c_max": 1.0, "cp_max": 1.0})


def test_ccp_standardization_makes_resolutions_comparable():
    # 草案 §4.4 b)：同一块玻璃在不同扫描分辨率下 CCP 应基本一致（标准化到 1px/mm）
    yy, xx = np.mgrid[0:60, 0:80].astype(float)         # 60×80mm @ 1px/mm
    base = 100.0 + 50.0 * np.sin(2 * np.pi * xx / 10.0) * np.cos(2 * np.pi * yy / 8.0)
    from skimage.transform import resize

    hi_res = resize(base, (120, 160), order=1, preserve_range=True)  # 同一玻璃 @ 2px/mm
    ref = {"c_max": 10.0, "cp_max": 1000.0}
    v_base = metrics.ccp(base, mm_per_px=1.0, ref=ref).value
    v_hi = metrics.ccp(hi_res, mm_per_px=0.5, ref=ref).value
    assert v_hi == pytest.approx(v_base, abs=0.03)


def test_ccp_grade_boundaries_8mm():
    # 草案表3 8mm 档：A_max=0.49, B_max=0.71
    assert metrics.grade(0.49, 8, "ccp") == "A"
    assert metrics.grade(0.71, 8, "ccp") == "B"
    assert metrics.grade(0.72, 8, "ccp") == "C"

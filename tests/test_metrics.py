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


# --------------------------- W_w（提案：位置加权聚类突出指数，零标定常数） --------------------------- #
def test_texture_w_zero_constants_no_refusal():
    # 零标定常数：默认 config（ε 为 TODO(plant)）下直接出值，不因缺常数拒绝
    rng = np.random.default_rng(0)
    img = rng.integers(0, 100, size=(32, 32)).astype(float)
    res = metrics.texture_w(img, mm_per_px=1.0)
    assert 0.0 <= res.value < 1.0
    assert res.degenerate is False


def test_texture_w_formula_selfconsistent_and_bounds():
    # 公式自洽：value = ½(√(Ca_w/(Ng−1)²) + ⁴√(CPa_w/((2(Ng−1))⁴/12)))
    # 两个上界常数：Ca → 49；CPa → 38416/12
    rng = np.random.default_rng(1)
    img = rng.normal(100.0, 25.0, size=(60, 80))
    res = metrics.texture_w(img, mm_per_px=1.0)
    ca_sup, cp_sup = metrics._ca_sup(8), metrics._cp_sup(8)
    assert ca_sup == pytest.approx(7 ** 2)
    assert cp_sup == pytest.approx((2 * 7) ** 4 / 12.0)
    expected = 0.5 * ((res.ca_w / ca_sup) ** 0.5 + (res.cpa_w / cp_sup) ** 0.25)
    assert res.value == pytest.approx(expected)
    # 与现行 CCP 逐算子同形：仅分母不同（同一图、同一 GLCM 口径下可互相换算）
    assert 0.0 <= res.value < 1.0


def test_texture_w_near_extremal_image_approaches_bound():
    # 构造近极值图（约 21% 面积为满值大块、其余为 0）→ 未加权 CPa 接近理论上界；
    # 任何图像的 texture_w 值恒 < 1（上界为上确界，真实连通图像仅渐近可达）
    img = np.zeros((300, 400))
    img[:, :84] = 255.0     # 左侧 21% 竖条（近极值两点分布 p≈0.21）
    cpa = metrics._glcm_ca_cpa(img, ng=8)[1]
    sup = metrics._cp_sup(8)
    assert 0.9 * sup < cpa < sup
    assert metrics.texture_w(img, mm_per_px=1.0).value < 1.0
    # 两分支各自 <1（值域保证来自两个上界，而非只有 CPa 一支）
    r = metrics.texture_w(img, mm_per_px=1.0)
    assert (r.ca_w / metrics._ca_sup(8)) ** 0.5 < 1.0
    assert (r.cpa_w / metrics._cp_sup(8)) ** 0.25 < 1.0


def test_texture_w_affine_invariance():
    # 逐片仿射变换（×a+b）→ min/max 量化归一 → 值不变（零灰度标定依赖）。
    # 无量化平局时逐位相同；带平局的情形另见 test_texture_w_affine_quantization_tie
    rng = np.random.default_rng(2)
    img = rng.normal(100.0, 20.0, size=(80, 100))
    v0 = metrics.texture_w(img, mm_per_px=1.0).value
    v1 = metrics.texture_w(img * 1.7 + 13.0, mm_per_px=1.0).value
    assert v1 == pytest.approx(v0, rel=1e-12)


def test_texture_w_affine_quantization_tie():
    # 仿射不变的边界：_quantize_minmax 用 floor(·+0.5)，恰落在量化平局点(.5)的像素
    # 在仿射后可能翻档 → W 有极小偏差（非逐位）。此处钉死"量级 ≤1e-5"这一如实口径，
    # 防止文档把仿射不变写成无条件逐位相等（2026-07-30 对抗复核发现）
    yy, xx = np.mgrid[0:60, 0:80].astype(float)
    img = 100.0 + 50.0 * np.sin(2 * np.pi * xx / 10.0) * np.cos(2 * np.pi * yy / 8.0)
    v0 = metrics.texture_w(img, mm_per_px=1.0).value
    v1 = metrics.texture_w(img + 13.0, mm_per_px=1.0).value
    rel = abs(v1 - v0) / v0
    assert rel < 1e-5, f"平局引起的偏差超出预期量级：rel={rel:.2e}"


def test_texture_w_rejects_non_finite():
    # 缺值不放行：含 NaN/Inf 的输入必须抛错，绝不静默返回"完美无斑"的最优值
    img = np.full((40, 50), 100.0)
    img[10, 10] = np.nan
    with pytest.raises(ValueError):
        metrics.texture_w(img, mm_per_px=1.0)
    img[10, 10] = np.inf
    with pytest.raises(ValueError):
        metrics.texture_w(img, mm_per_px=1.0)


def test_texture_w_contrast_theoretical_bound():
    # 对比度 Ca 的纯数学上界 (Ng−1)²=49（|i−j| ≤ Ng−1）——这是双分量组合仍能
    # 保持"零标定常数"的原因（2026-07-30 按实效改回双分量）
    rng = np.random.default_rng(11)
    for shape in ((60, 80), (120, 100)):
        img = rng.normal(100.0, 30.0, size=shape)
        w = metrics._position_weight_map(shape)
        ca_w, _ = metrics._glcm_ca_cpa_weighted(img, w, ng=8)
        assert 0.0 <= ca_w < (8 - 1) ** 2
    # 逐列交替图使 0° 方向全为 (0,Ng−1) 对 → 该方向达上界，四向平均后仍 <1
    alt = np.zeros((200, 200))
    alt[:, ::2] = 255.0
    ca_alt, _ = metrics._glcm_ca_cpa_weighted(alt, np.ones((200, 200)), ng=8)
    assert 0.5 * (8 - 1) ** 2 < ca_alt < (8 - 1) ** 2


def test_texture_w_uniform_texture_degenerates_to_unweighted():
    # 空间均匀纹理（iid 噪声场）→ 位置加权退化 → 与未加权 CP 指数一致（rel 2%）
    rng = np.random.default_rng(3)
    img = rng.normal(100.0, 20.0, size=(200, 300))
    v_w = metrics.texture_w(img, mm_per_px=1.0).value
    ca_p, cpa_p = metrics._glcm_ca_cpa(img, ng=8)
    v_plain = 0.5 * ((ca_p / metrics._ca_sup(8)) ** 0.5
                     + (cpa_p / metrics._cp_sup(8)) ** 0.25)
    assert v_w == pytest.approx(v_plain, rel=0.02)


def test_texture_w_center_vs_corner_direction():
    # 位置方向性：同幅度高斯斑放中心 → 值更高（更差）；放角落 → 更低
    yy, xx = np.mgrid[0:120, 0:160].astype(float)

    def blob(cy, cx):
        """单高斯斑合成图：斑心 (cy,cx)，σ=15px，峰值 60（背景 100）。"""
        return 100.0 + 60.0 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 15.0 ** 2)))

    v_center = metrics.texture_w(blob(60, 80), mm_per_px=1.0).value
    v_corner = metrics.texture_w(blob(12, 16), mm_per_px=1.0).value
    assert v_center > v_corner


def test_texture_w_iid_noise_naturally_low():
    # CP 对 iid 噪声天然钝感（弃 Ca 的收益）：近平坦+微噪片 → 低值，无 ε 也不误判
    rng = np.random.default_rng(4)
    flat = 100.0 + rng.normal(0.0, 0.5, size=(200, 300))
    assert metrics.texture_w(flat, mm_per_px=1.0).value < 0.3


def test_texture_w_min_dynamic_range_clause():
    # ε 条款（可选物理守卫）：注入 ε → 近平坦片判无纹理记 0；未注入 → 不触发
    rng = np.random.default_rng(5)
    flat = 100.0 + rng.normal(0.0, 0.5, size=(80, 100))
    res = metrics.texture_w(flat, mm_per_px=1.0, ref={"min_dynamic_range_nm": 5.0})
    assert res.degenerate is True and res.value == 0.0
    assert res.ca_w == 0.0 and res.cpa_w == 0.0
    assert res.dynamic_range_nm < 5.0
    res2 = metrics.texture_w(flat, mm_per_px=1.0, ref={})
    assert res2.degenerate is False


def test_texture_w_requires_2d():
    # 一维数组应被拒绝（须为二维光程差图像），同 ccp 口径
    with pytest.raises(ValueError):
        metrics.texture_w(np.zeros(100), mm_per_px=1.0)


def test_texture_w_standardization_makes_resolutions_comparable():
    # 草案 §4.4 b) 口径沿用：同一玻璃不同扫描分辨率 → W_w 基本一致（1px/mm 标准化）
    yy, xx = np.mgrid[0:60, 0:80].astype(float)
    base = 100.0 + 50.0 * np.sin(2 * np.pi * xx / 10.0) * np.cos(2 * np.pi * yy / 8.0)
    from skimage.transform import resize

    hi_res = resize(base, (120, 160), order=1, preserve_range=True)
    v_base = metrics.texture_w(base, mm_per_px=1.0).value
    v_hi = metrics.texture_w(hi_res, mm_per_px=0.5).value
    assert v_hi == pytest.approx(v_base, abs=0.03)


def test_texture_w_weight_map_pure_linear():
    # w(r)=1−r 零常数：中心=1、边缘趋 0、无下限平台；全图权重 >0（像素中心离散化）
    w = metrics._position_weight_map((101, 101))
    assert w[50, 50] == pytest.approx(1.0)
    assert w.min() > 0.0
    assert w.min() == pytest.approx(1.0 / 101, rel=1e-6)


def test_texture_w_grade_todo_table_returns_none():
    # grading.yaml 的 texture_w 表为 TODO(plant) → 无法判级，返回 None（不猜测）
    assert metrics.grade(0.5, 8, "texture_w") is None


def test_texture_w_grade_with_injected_table():
    # 注入限值表（工作组建立后的形态）→ A/B/C 边界同"≤"方向；超档 → None
    g = {"texture_w": [{"thickness_mm": 8, "A_max": 0.50, "B_max": 0.70}]}
    assert metrics.grade(0.50, 8, "texture_w", grading=g) == "A"
    assert metrics.grade(0.70, 8, "texture_w", grading=g) == "B"
    assert metrics.grade(0.71, 8, "texture_w", grading=g) == "C"
    assert metrics.grade(0.50, 16, "texture_w", grading=g) is None


def test_texture_w_deterministic_repeat():
    # 确定性：同图重复计算结果逐位一致
    rng = np.random.default_rng(6)
    img = rng.normal(100.0, 25.0, size=(60, 70))
    v0 = metrics.texture_w(img, mm_per_px=1.0).value
    for _ in range(5):
        assert metrics.texture_w(img, mm_per_px=1.0).value.hex() == v0.hex()

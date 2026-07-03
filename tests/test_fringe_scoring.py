"""应力斑分布打分 · 性质测试（合成图，不依赖现场真值）。

覆盖性质：
① 稳健性：背景整体仿射变换（+c、×k）不改分——逐片稳健标准化的核心承诺；
② 单调性：斑更靠中心 / 更深 / 面积更大 → 分更低；
③ 边框剔除：宽窄不一的重边框不计惩罚；
④ 退化边界：常量图 / 纯渐变图 → 无斑满分，非法枚举报错；
⑤ 四边形玻璃：角点检测精度 / 矫正打分一致性 / 填充抗干扰 / 整图旁路 / 异常拒绝。
"""

import numpy as np
import pytest

from fringe_scoring import load_config, score_fringe_distribution
from fringe_scoring.border import band_widths_px
from fringe_scoring.quad import detect_glass_quad, order_corners
from fringe_scoring.synth import make_glass_image, warp_into_quad


def base_config() -> dict:
    """测试用固定配置（不读仓库 yaml，避免调参影响性质断言）。"""
    return {
        "quad": {
            "auto_detect": True,
            "fill_tol_frac": 0.05,
            "min_fill_border_frac": 0.2,
            "full_frame_frac": 0.99,
            "min_glass_frac": 0.5,
            "poly_epsilon_frac": 0.02,
        },
        "segment": {
            "method": "quantile",
            "top_fraction": 0.05,
            "min_z": 2.5,
            "polarity": "both",
            "z_threshold": 3.0,
            "s_scale_mode": "per_sheet",  # 本文件既有性质（含仿射不变）针对 per_sheet 锚；绝对锚见 ⑥
            "z_saturation": 8.0,
            "background_block_frac": 0.05,
            "background_poly_degree": 1,
            "min_contrast_frac": 0.05,
        },
        "border": {
            "min_band_frac": 0.0,
            "max_band_frac": 0.25,
            "profile_quantile": 0.5,
            "return_to_background_ratio": 2.0,
        },
        "weight": {"kind": "linear", "gaussian_sigma": 0.5, "distance_norm": "chebyshev"},
    }


# 通用四边形角点（凸，非平行四边形）：画布 260×380，(x=列, y=行)，左上起顺时针
QUAD_CORNERS = np.array([[60.0, 10.0], [355.0, 25.0], [320.0, 245.0], [25.0, 230.0]])
QUAD_CANVAS = (260, 380)


# ---------------- ① 稳健性：背景深浅不固定 ----------------
def test_affine_background_invariance():
    # 同一斑型，整图 ×k+c（模拟每片背景深浅/对比度不同）→ 分数不变
    img = make_glass_image(blobs=[(0.5, 0.5, 0.08, 5.0)], seed=3)
    res_a = score_fringe_distribution(img, config=base_config())
    res_b = score_fringe_distribution(1.7 * img + 33.0, config=base_config())
    assert abs(res_a.score_0_100 - res_b.score_0_100) < 1e-6


# ---------------- ② 单调性 ----------------
def test_center_fringe_scores_worse_than_offcenter():
    # 同一颗斑：贴中心 vs 靠角落 → 中心分更低（离中心越远惩罚越低）
    center = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=5)
    corner = make_glass_image(blobs=[(0.72, 0.72, 0.08, 6.0)], seed=5)
    s_center = score_fringe_distribution(center, config=base_config()).score_0_100
    s_corner = score_fringe_distribution(corner, config=base_config()).score_0_100
    assert s_center < s_corner


def test_deeper_fringe_scores_worse():
    # 同位置同大小：更深（幅度×2，未触 z_saturation 封顶）→ 分更低
    shallow = make_glass_image(blobs=[(0.5, 0.5, 0.08, 3.0)], seed=7)
    deep = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=7)
    s_shallow = score_fringe_distribution(shallow, config=base_config()).score_0_100
    s_deep = score_fringe_distribution(deep, config=base_config()).score_0_100
    assert s_deep < s_shallow


def test_more_fringe_area_scores_worse():
    # 斑面积更大 → 分更低；用 robust_z 法（掩膜随面积增长，同时覆盖备选分割路径）
    cfg = base_config()
    cfg["segment"]["method"] = "robust_z"
    one = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=9)
    two = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0), (0.35, 0.65, 0.08, 6.0)], seed=9)
    s_one = score_fringe_distribution(one, config=cfg).score_0_100
    s_two = score_fringe_distribution(two, config=cfg).score_0_100
    assert s_two < s_one


# ---------------- ③ 边框剔除 ----------------
def test_heavy_uneven_frame_does_not_change_score():
    # 加一圈宽窄不一的重边框（若计入惩罚会重挫分数）→ 与无边框基线近似相等
    blob = [(0.5, 0.5, 0.08, 6.0)]
    plain = make_glass_image(blobs=blob, seed=11)
    framed = make_glass_image(
        blobs=blob, frame_widths_frac=(0.05, 0.12, 0.08, 0.03), frame_amp=40.0, seed=11
    )
    res_plain = score_fringe_distribution(plain, config=base_config())
    res_framed = score_fringe_distribution(framed, config=base_config())
    assert abs(res_plain.score_0_100 - res_framed.score_0_100) < 1.0
    # 四边带宽应大致跟随真实边框（宽窄不一），且不超过保护上限
    bands = res_framed.border_bands
    assert bands.bottom_px > bands.right_px  # 0.12 边 vs 0.03 边
    assert bands.top_px <= 0.25 * plain.shape[0]


def test_band_widths_follow_uneven_frame_exactly():
    # 无噪声、内部全 0 的 |残差| 图：检测带宽应精确等于铺设的边框宽度
    dev_abs = np.zeros((200, 300))
    dev_abs[:10, :] = 5.0    # top
    dev_abs[-24:, :] = 5.0   # bottom
    dev_abs[:, :24] = 5.0    # left
    dev_abs[:, -9:] = 5.0    # right
    bands = band_widths_px(dev_abs, base_config()["border"])
    assert (bands.top_px, bands.bottom_px, bands.left_px, bands.right_px) == (10, 24, 24, 9)


# ---------------- ④ 退化与配置 ----------------
def test_constant_image_scores_perfect():
    # 常量图（MAD=0）：安全退化为"无斑"，满分且诊断量一致
    res = score_fringe_distribution(np.full((120, 160), 50.0), config=base_config())
    assert res.score_0_100 == 100.0
    assert res.fringe_area_frac == 0.0
    assert res.centrality is None


def test_smooth_gradient_scores_perfect():
    # 纯渐变无噪声：背景估计吸收渐变，残差为数值噪声 → 对比度门槛判"无斑"
    img = make_glass_image(noise_sigma=0.0, seed=13)
    res = score_fringe_distribution(img, config=base_config())
    assert res.score_0_100 == 100.0


def test_invalid_polarity_rejected():
    # 非法枚举值：报错拒绝，不静默放行（规则 > AI）
    cfg = base_config()
    cfg["segment"]["polarity"] = "up"
    with pytest.raises(ValueError):
        score_fringe_distribution(make_glass_image(seed=1), config=cfg)


def test_repo_yaml_config_is_valid():
    # 仓库 config/fringe_scoring.yaml 可读、枚举合法、能端到端跑通一张图
    cfg = load_config()
    res = score_fringe_distribution(make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=15), config=cfg)
    assert 0.0 <= res.score_0_100 <= 100.0
    assert res.penalty_max > 0.0


# ---------------- ⑤ 四边形玻璃（大图裁切，非正矩形） ----------------
# 合成口径：源图不带噪声，贴进四边形后再往玻璃区域补噪声——真实传感器噪声长在
# 大图上、是清晰的；若源图带噪声会被贴图插值平滑一次，与真实裁切图不符。
def _quad_canvas(blobs=None, seed: int = 0, fill_value: float = 30.0) -> np.ndarray:
    """生成一张"通用四边形玻璃"裁切图（噪声在贴图后补进玻璃区域）。"""
    clean = make_glass_image(noise_sigma=0.0, blobs=blobs, seed=seed)
    return warp_into_quad(
        clean, QUAD_CORNERS, QUAD_CANVAS, fill_value=fill_value, noise_sigma=1.0, seed=seed
    )


def test_quad_detection_accuracy():
    # 已知角点的通用四边形玻璃 → 自动检出角点逐点误差 ≤ 3px
    detected = detect_glass_quad(_quad_canvas(seed=21), base_config()["quad"])
    assert detected is not None
    err = np.abs(detected - order_corners(QUAD_CORNERS)).max()
    assert err <= 3.0, f"角点最大误差 {err:.2f}px"


def test_quad_warp_score_consistency():
    # 同一玻璃内容：矩形直接打分 vs 贴成四边形后自动矫正打分 → 偏移有界（≤3 分）。
    # 矫正插值会轻微平滑噪声 → 稳健 z 锚收紧 → 矫正图系统性偏严 ~2-3 分（docs §7）；
    # 同族排序不受影响（见 test_quad_family_ordering_preserved），真实裁切图全走矫正路径。
    blobs = [(0.5, 0.5, 0.08, 6.0)]
    direct = score_fringe_distribution(make_glass_image(blobs=blobs, seed=23), config=base_config())
    warped = score_fringe_distribution(_quad_canvas(blobs=blobs, seed=23), config=base_config())
    assert warped.quad_corners_px is not None
    assert abs(direct.score_0_100 - warped.score_0_100) <= 3.0


def test_quad_family_ordering_preserved():
    # 同为四边形玻璃（都经矫正）：干净 > 角落斑 > 中心斑 的排序保持
    s_clean = score_fringe_distribution(_quad_canvas(seed=31), config=base_config()).score_0_100
    s_corner = score_fringe_distribution(
        _quad_canvas(blobs=[(0.72, 0.72, 0.08, 6.0)], seed=31), config=base_config()
    ).score_0_100
    s_center = score_fringe_distribution(
        _quad_canvas(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=31), config=base_config()
    ).score_0_100
    assert s_center < s_corner < s_clean


def test_quad_fill_value_does_not_matter():
    # 抗干扰：四边形外的填充值换黑/中灰/亮 → 分数几乎不变（填充不进任何统计）
    blobs = [(0.5, 0.5, 0.08, 6.0)]
    scores = [
        score_fringe_distribution(
            _quad_canvas(blobs=blobs, seed=25, fill_value=fv), config=base_config()
        ).score_0_100
        for fv in (0.0, 128.0, 220.0)
    ]
    assert max(scores) - min(scores) < 0.5


def test_full_frame_bypasses_quad_and_matches():
    # 整图即玻璃（正矩形裁切）：检测返回 None，与关掉 auto_detect 的结果完全一致
    img = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=27)
    cfg_off = base_config()
    cfg_off["quad"]["auto_detect"] = False
    res_on = score_fringe_distribution(img, config=base_config())
    res_off = score_fringe_distribution(img, config=cfg_off)
    assert res_on.quad_corners_px is None
    assert res_on.score_0_100 == res_off.score_0_100


# ---------------- ⑥ 绝对灰度锚（s_scale_mode=absolute，决策 #11） ----------------
def absolute_config() -> dict:
    """绝对锚测试配置：s 用固定灰度饱和值（合成图幅度量级下取 60）。"""
    cfg = base_config()
    cfg["segment"]["s_scale_mode"] = "absolute"
    cfg["segment"]["s_saturation_gray"] = 60.0
    cfg["segment"]["min_dev_gray"] = 4.0  # 合成图噪声 σ=1 → 4σ 质控下限
    return cfg


def test_absolute_mode_fixes_uniform_severe_inversion():
    # 排序反转回归（真实照片实测暴露）：整片布满深斑 vs 干净玻璃+一颗淡斑。
    # per_sheet 锚下前者自身 MAD 被斑撑大 → 漏罚；绝对锚下必须判"整片深斑"更差。
    heavy_blobs = [(r, c, 0.10, 45.0) for r in (0.2, 0.5, 0.8) for c in (0.2, 0.5, 0.8)]
    heavy = make_glass_image(blobs=heavy_blobs, seed=11)
    faint = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=11)
    cfg = absolute_config()
    s_heavy = score_fringe_distribution(heavy, config=cfg).score_0_100
    s_faint = score_fringe_distribution(faint, config=cfg).score_0_100
    assert s_heavy < s_faint - 1.0


def test_absolute_mode_depth_monotonic():
    # 同位置同大小：绝对幅度更深 → 分更低（绝对锚保留深浅信号，不被逐片归一化吃掉）
    shallow = make_glass_image(blobs=[(0.5, 0.5, 0.08, 10.0)], seed=13)
    deep = make_glass_image(blobs=[(0.5, 0.5, 0.08, 30.0)], seed=13)
    cfg = absolute_config()
    s_shallow = score_fringe_distribution(shallow, config=cfg).score_0_100
    s_deep = score_fringe_distribution(deep, config=cfg).score_0_100
    assert s_deep < s_shallow


def test_absolute_mode_is_exposure_sensitive_by_design():
    # 绝对锚有意放弃仿射不变（前提=同批曝光固定）：整体 ×k 加深 → 分数变化
    img = make_glass_image(blobs=[(0.5, 0.5, 0.08, 20.0)], seed=17)
    cfg = absolute_config()
    s_a = score_fringe_distribution(img, config=cfg).score_0_100
    s_b = score_fringe_distribution(1.7 * img, config=cfg).score_0_100
    assert s_b < s_a  # 乘性加深 → 斑更深 → 分更低


def test_absolute_mode_missing_saturation_raises():
    # absolute 模式缺 s_saturation_gray → 报错拒绝，不静默退回旧口径
    cfg = absolute_config()
    del cfg["segment"]["s_saturation_gray"]
    img = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=19)
    with pytest.raises(ValueError, match="s_saturation_gray"):
        score_fringe_distribution(img, config=cfg)


# ---------------- ⑧ gray_threshold 面积敏感 / bright 极性 / 边缘权重下限（决策 #12） ----------------
def gray_threshold_config() -> dict:
    """决策 #12 口径：灰度阈值分割（面积不封顶）+ 只算偏亮。"""
    cfg = absolute_config()
    cfg["segment"]["method"] = "gray_threshold"
    cfg["segment"]["polarity"] = "bright"
    return cfg


def test_gray_threshold_area_sensitive():
    # 同深度、面积更大的斑 → 分更低（quantile 法面积封顶做不到这一点，正是换法动机）
    small = make_glass_image(blobs=[(0.5, 0.5, 0.06, 30.0)], seed=41)
    large = make_glass_image(blobs=[(0.5, 0.5, 0.16, 30.0)], seed=41)
    cfg = gray_threshold_config()
    s_small = score_fringe_distribution(small, config=cfg).score_0_100
    s_large = score_fringe_distribution(large, config=cfg).score_0_100
    assert s_large < s_small - 1.0


def test_gray_threshold_requires_absolute_mode():
    # gray_threshold 定义在灰度域，per_sheet 模式下必须报错而非静默换义
    cfg = base_config()
    cfg["segment"]["method"] = "gray_threshold"
    img = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=43)
    with pytest.raises(ValueError, match="gray_threshold"):
        score_fringe_distribution(img, config=cfg)


def test_bright_polarity_ignores_dark_deviation():
    # bright 极性：偏暗的偏离不算斑（暗场物理=应力必偏亮；防背景漂移时反标干净暗区）
    dark_blob = make_glass_image(blobs=[(0.5, 0.5, 0.08, -30.0)], seed=45)
    cfg = gray_threshold_config()
    s_bright = score_fringe_distribution(dark_blob, config=cfg).score_0_100
    cfg_both = gray_threshold_config()
    cfg_both["segment"]["polarity"] = "both"
    s_both = score_fringe_distribution(dark_blob, config=cfg_both).score_0_100
    assert s_bright > s_both  # both 会罚暗斑，bright 不罚
    assert s_bright > 99.0


def test_weight_floor_narrows_edge_forgiveness_but_keeps_center_worse():
    # floor 的干净性质是相对的：边缘斑/中心斑的罚分比 ρ_edge/ρ_center 随 floor 收窄
    # （floor 同时抬高 penalty_max，单图分数不保证单调；且斑不能贴边，否则被边框带剔除）
    edge = make_glass_image(blobs=[(0.5, 0.8, 0.06, 30.0)], seed=47)  # w≈0.4，避开边框带
    center = make_glass_image(blobs=[(0.5, 0.5, 0.06, 30.0)], seed=47)
    cfg_floor = gray_threshold_config()
    cfg_floor["weight"]["floor"] = 0.5
    cfg_nofloor = gray_threshold_config()
    cfg_nofloor["weight"]["floor"] = 0.0

    def rho(img, cfg):
        """罚分比 ρ = 1 − score/100（与刻度锚无关的单调量）。"""
        return 1.0 - score_fringe_distribution(img, config=cfg).score_0_100 / 100.0

    ratio_floor = rho(edge, cfg_floor) / rho(center, cfg_floor)
    ratio_nofloor = rho(edge, cfg_nofloor) / rho(center, cfg_nofloor)
    assert ratio_floor > ratio_nofloor  # 边缘斑相对中心斑的宽恕变少
    assert rho(center, cfg_floor) > rho(edge, cfg_floor)  # 中心仍然最重罚


def test_weight_floor_invalid_raises():
    # floor 越界 [0,1) → 报错拒绝
    img = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=49)
    for bad in (-0.1, 1.0, 1.5):
        cfg = gray_threshold_config()
        cfg["weight"]["floor"] = bad
        with pytest.raises(ValueError, match="floor"):
            score_fringe_distribution(img, config=cfg)


# ---------------- ⑦ 输出刻度锚（scoring.penalty_ratio_at_zero） ----------------
def test_score_anchor_lowers_scores_but_keeps_order():
    # 刻度锚是纯单调映射：分数整体下移、排序不变；缺省(不配 scoring 段)=旧行为
    shallow = make_glass_image(blobs=[(0.5, 0.5, 0.08, 10.0)], seed=31)
    deep = make_glass_image(blobs=[(0.5, 0.5, 0.08, 30.0)], seed=31)
    cfg = absolute_config()
    base_shallow = score_fringe_distribution(shallow, config=cfg).score_0_100
    base_deep = score_fringe_distribution(deep, config=cfg).score_0_100
    cfg["scoring"] = {"penalty_ratio_at_zero": 0.2}
    anch_shallow = score_fringe_distribution(shallow, config=cfg).score_0_100
    anch_deep = score_fringe_distribution(deep, config=cfg).score_0_100
    assert anch_shallow < base_shallow and anch_deep < base_deep  # 整体下移
    assert anch_deep < anch_shallow  # 排序不变


def test_score_anchor_clips_at_zero():
    # 罚分比超过锚值 → 0 分封底，不出负分
    heavy_blobs = [(r, c, 0.10, 45.0) for r in (0.2, 0.5, 0.8) for c in (0.2, 0.5, 0.8)]
    heavy = make_glass_image(blobs=heavy_blobs, seed=33)
    cfg = absolute_config()
    cfg["scoring"] = {"penalty_ratio_at_zero": 0.001}
    assert score_fringe_distribution(heavy, config=cfg).score_0_100 == 0.0


def test_score_anchor_invalid_raises():
    # 锚值越界 (0,1] → 报错拒绝
    img = make_glass_image(blobs=[(0.5, 0.5, 0.08, 6.0)], seed=35)
    for bad in (0.0, -0.2, 1.5):
        cfg = absolute_config()
        cfg["scoring"] = {"penalty_ratio_at_zero": bad}
        with pytest.raises(ValueError, match="penalty_ratio_at_zero"):
            score_fringe_distribution(img, config=cfg)


def test_quad_low_glass_frac_rejected():
    # 玻璃占比过低（非近恒定图）→ 检测异常，报错拒绝打分，不瞎打
    canvas = np.full((300, 300), 30.0)
    canvas[100:180, 100:180] = make_glass_image(rows=80, cols=80, seed=29)
    with pytest.raises(ValueError):
        score_fringe_distribution(canvas, config=base_config())

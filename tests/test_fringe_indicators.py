"""六指标测量 · 性质测试（合成图，不依赖现场真值）。

覆盖性质：
① 深浅单调：更深斑 → X0.95/灰度方差↑；更花 → 梯度统计↑；
② 均匀度深浅解耦：同图案 ×k 加深 → uniformity 不变（±ε），深浅指标变差；
   全白/常量图 → uniformity=100、方差/梯度≈0、总分≈满分；
③ 子分映射：方向（越差子分越低）与 [0,100] clip；平权总分=六子分均值；权重可调生效；
④ 失败契约：缺 indicators 段 / 权重键不齐 / refs 退化 → ValueError；
⑤ CCP 对拍：numpy 自实现 GLCM 与 tools/metrics.py 的 skimage 口径容差内一致。
"""

import numpy as np
import pytest

from fringe_scoring.indicators import INDICATOR_KEYS, compute_sheet_indicators
from fringe_scoring.synth import make_glass_image


def base_config() -> dict:
    """测试用固定基础配置（自含，不读仓库 yaml；与 test_fringe_scoring 同构）。"""
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
            "method": "gray_threshold",
            "top_fraction": 0.05,
            "min_z": 2.5,
            "polarity": "bright",
            "s_scale_mode": "absolute",
            "s_saturation_gray": 60.0,
            "min_dev_gray": 4.0,
            "z_threshold": 3.0,
            "z_saturation": 8.0,
            "background_block_frac": 0.05,
            "background_block_quantile": 0.15,
            "background_poly_degree": 1,
            "min_contrast_frac": 0.05,
        },
        "border": {
            "min_band_frac": 0.0,
            "max_band_frac": 0.25,
            "profile_quantile": 0.5,
            "return_to_background_ratio": 2.0,
            "normal_band_frac": 0.04,
        },
        "weight": {"kind": "linear", "gaussian_sigma": 0.5, "distance_norm": "chebyshev",
                   "floor": 0.3},
    }


def indicators_config() -> dict:
    """测试用完整配置：base_config + indicators 段（合成图量级的 refs）。"""
    cfg = base_config()
    cfg["indicators"] = {
        "weights": {k: 1.0 for k in INDICATOR_KEYS},
        "uniformity_ratio_at_zero": 0.33,
        "refs": {
            "x095": {"best": 100.0, "worst": 160.0},   # 合成图背景≈100，斑幅≤60
            "gray_variance": {"best": 1.0, "worst": 400.0},
            "gradient_mean": {"best": 3.0, "worst": 40.0},
            "gradient_variance": {"best": 10.0, "worst": 1000.0},
            "ccp": {"best": 0.4, "worst": 1.0},
        },
        "calibration": {
            "gray_to_nm": None, "mm_per_px": None,
            "ccp_c_max": 0.10, "ccp_cp_max": 700.0,
            "ccp_reference_is_plant_calibrated": False, "thickness_mm": None,
        },
    }
    return cfg


def full_interior(img: np.ndarray) -> np.ndarray:
    """测试用内部掩膜：整图（合成图无需扣边框带，指标函数只看掩膜）。"""
    return np.ones(img.shape, dtype=bool)


# ---------------- ① 深浅单调 ----------------
def test_deeper_blob_raises_x095_and_variance():
    # 同位置同大小，幅度×3 → X0.95 与灰度方差都必须升高
    shallow = make_glass_image(blobs=[(0.5, 0.5, 0.15, 15.0)], seed=61)
    deep = make_glass_image(blobs=[(0.5, 0.5, 0.15, 45.0)], seed=61)
    cfg = indicators_config()
    ind_s = compute_sheet_indicators(shallow, full_interior(shallow), cfg)
    ind_d = compute_sheet_indicators(deep, full_interior(deep), cfg)
    assert ind_d.x095 > ind_s.x095
    assert ind_d.gray_variance > ind_s.gray_variance
    assert ind_d.total_score < ind_s.total_score  # 更深 → 总分更低


def test_busier_texture_raises_gradient_stats():
    # 高频强条纹 vs 平滑单斑 → 梯度均值/方差都更高
    yy, xx = np.mgrid[0:200, 0:300]
    busy = make_glass_image(seed=63) + 30.0 * np.sin(2 * np.pi * xx / 12.0)
    smooth = make_glass_image(blobs=[(0.5, 0.5, 0.20, 30.0)], seed=63)
    cfg = indicators_config()
    ind_b = compute_sheet_indicators(busy, full_interior(busy), cfg)
    ind_s = compute_sheet_indicators(smooth, full_interior(smooth), cfg)
    assert ind_b.gradient_mean > ind_s.gradient_mean
    assert ind_b.gradient_variance > ind_s.gradient_variance


# ---------------- ② 均匀度与深浅解耦 ----------------
def test_uniformity_invariant_to_depth():
    # 同一斑纹图案整体 ×k：uniformity 不变（空间分布未变），X0.95/方差变差
    base = make_glass_image(blobs=[(0.5, 0.5, 0.10, 20.0)], seed=65)
    deeper = 1.8 * base  # 乘性加深：图案不变、幅度变大
    cfg = indicators_config()
    ind_a = compute_sheet_indicators(base, full_interior(base), cfg)
    ind_b = compute_sheet_indicators(deeper, full_interior(deeper), cfg)
    assert abs(ind_a.uniformity - ind_b.uniformity) < 1e-6  # 仿射不变（per_sheet 口径）
    assert ind_b.x095 > ind_a.x095
    assert ind_b.gray_variance > ind_a.gray_variance


def test_constant_sheet_uniformity_100_and_near_perfect_total():
    # 全白/常量图（用户锚定语义）：uniformity=100；方差/梯度≈0；总分≈满分
    white = np.full((200, 300), 250.0)
    cfg = indicators_config()
    cfg["indicators"]["refs"]["x095"] = {"best": 250.0, "worst": 400.0}  # 常量图 x095=250
    ind = compute_sheet_indicators(white, full_interior(white), cfg)
    assert ind.uniformity == 100.0
    assert ind.gray_variance < 1e-9
    assert ind.gradient_mean < 1e-9
    assert ind.total_score > 95.0


# ---------------- ③ 子分与加权 ----------------
def test_equal_weights_total_is_mean_of_sub_scores():
    # 平权（需求 7）：总分 = 六子分简单平均
    img = make_glass_image(blobs=[(0.5, 0.5, 0.10, 25.0)], seed=67)
    ind = compute_sheet_indicators(img, full_interior(img), indicators_config())
    assert ind.total_score == pytest.approx(
        float(np.mean([ind.sub_scores[k] for k in INDICATOR_KEYS]))
    )


def test_weight_adjustment_changes_total():
    # 工厂调权生效：把 uniformity 权重拉满、其余归零 → 总分=uniformity 子分
    img = make_glass_image(blobs=[(0.5, 0.5, 0.10, 25.0)], seed=69)
    cfg = indicators_config()
    cfg["indicators"]["weights"] = {k: (1.0 if k == "uniformity" else 0.0) for k in INDICATOR_KEYS}
    ind = compute_sheet_indicators(img, full_interior(img), cfg)
    assert ind.total_score == pytest.approx(ind.sub_scores["uniformity"])


def test_sub_scores_clipped_to_0_100():
    # 超出 worst 参考的极端值 → 子分钳到 0（不出负分）
    deep = make_glass_image(blobs=[(0.5, 0.5, 0.20, 200.0)], seed=71)
    ind = compute_sheet_indicators(deep, full_interior(deep), indicators_config())
    for key, s in ind.sub_scores.items():
        assert 0.0 <= s <= 100.0, key


# ---------------- ④ 失败契约 ----------------
def test_missing_indicators_section_raises():
    # 缺 indicators 段 → 显式报错
    img = make_glass_image(seed=73)
    with pytest.raises(ValueError, match="indicators 段"):
        compute_sheet_indicators(img, full_interior(img), base_config())


def test_incomplete_weights_raises():
    # 权重键不齐 → 报错（六键契约）
    img = make_glass_image(seed=75)
    cfg = indicators_config()
    del cfg["indicators"]["weights"]["ccp"]
    with pytest.raises(ValueError, match="weights"):
        compute_sheet_indicators(img, full_interior(img), cfg)


def test_degenerate_refs_raise():
    # best == worst 的退化参考 → 报错不硬算
    img = make_glass_image(seed=77)
    cfg = indicators_config()
    cfg["indicators"]["refs"]["x095"] = {"best": 5.0, "worst": 5.0}
    with pytest.raises(ValueError, match="x095"):
        compute_sheet_indicators(img, full_interior(img), cfg)


# ---------------- ⑥ 方案快路径：score_from_raw 与首算一致 ----------------
def test_score_from_raw_matches_full_compute():
    # 缓存原始值 + 同方案重算 == 首算（切换加权方案免重跑图像的正确性地基）
    from fringe_scoring.indicators import score_from_raw

    img = make_glass_image(blobs=[(0.5, 0.5, 0.10, 25.0)], seed=81)
    cfg = indicators_config()
    ind = compute_sheet_indicators(img, full_interior(img), cfg)
    sub, total = score_from_raw(
        ind.raw_values, cfg["indicators"]["refs"], cfg["indicators"]["weights"]
    )
    assert total == pytest.approx(ind.total_score)
    for key in INDICATOR_KEYS:
        assert sub[key] == pytest.approx(ind.sub_scores[key])


def test_score_from_raw_new_weights_matches_fresh_compute():
    # 换一套权重：快路径重算 == 用新权重全量重跑（uniformity 等原始值不受权重影响）
    from fringe_scoring.indicators import score_from_raw

    img = make_glass_image(blobs=[(0.5, 0.5, 0.10, 25.0)], seed=83)
    cfg_a = indicators_config()
    ind_a = compute_sheet_indicators(img, full_interior(img), cfg_a)
    new_weights = {k: (2.0 if k == "uniformity" else 0.5) for k in INDICATOR_KEYS}
    _, total_fast = score_from_raw(ind_a.raw_values, cfg_a["indicators"]["refs"], new_weights)
    cfg_b = indicators_config()
    cfg_b["indicators"]["weights"] = new_weights
    ind_b = compute_sheet_indicators(img, full_interior(img), cfg_b)
    assert total_fast == pytest.approx(ind_b.total_score)


def test_uniformity_default_ref_equals_passthrough():
    # uniformity 参考默认 {best:100, worst:0} == 直通（旧配置缺该键也直通）
    from fringe_scoring.indicators import score_from_raw

    img = make_glass_image(blobs=[(0.5, 0.5, 0.10, 25.0)], seed=85)
    cfg = indicators_config()
    ind = compute_sheet_indicators(img, full_interior(img), cfg)
    refs_with = dict(cfg["indicators"]["refs"])
    refs_with["uniformity"] = {"best": 100.0, "worst": 0.0}
    sub_a, total_a = score_from_raw(ind.raw_values, cfg["indicators"]["refs"], cfg["indicators"]["weights"])
    sub_b, total_b = score_from_raw(ind.raw_values, refs_with, cfg["indicators"]["weights"])
    assert sub_b["uniformity"] == pytest.approx(sub_a["uniformity"])
    assert total_b == pytest.approx(total_a)


def test_uniformity_custom_ref_maps_direction():
    # 自定义均匀度参考 {best:100, worst:50}：原始 75 → 子分 50（越大越好方向自动成立）
    from fringe_scoring.indicators import score_from_raw

    raw = {"x095": 20.0, "gray_variance": 1.0, "gradient_mean": 3.0,
           "gradient_variance": 10.0, "ccp": 0.4, "uniformity": 75.0}
    cfg = indicators_config()
    refs = dict(cfg["indicators"]["refs"])
    refs["uniformity"] = {"best": 100.0, "worst": 50.0}
    sub, _ = score_from_raw(raw, refs, cfg["indicators"]["weights"])
    assert sub["uniformity"] == pytest.approx(50.0)


# ---------------- ⑤ CCP 对拍 skimage 口径 ----------------
def test_glcm_matches_skimage_reference():
    # numpy 自实现 GLCM 的 Ca/CPa 与 tools/metrics.py（skimage）容差内一致
    from fringe_scoring.indicators import _glcm_ca_cpa
    from tools.metrics import _glcm_ca_cpa as skimage_ca_cpa

    img = make_glass_image(blobs=[(0.4, 0.6, 0.12, 35.0)], seed=79)
    ca_np, cpa_np = _glcm_ca_cpa(img)
    ca_sk, cpa_sk = skimage_ca_cpa(img, ng=8)
    assert ca_np == pytest.approx(ca_sk, rel=1e-6)
    assert cpa_np == pytest.approx(cpa_sk, rel=1e-6)

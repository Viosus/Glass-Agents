"""应力斑分布打分 · 性质测试（合成图，不依赖现场真值）。

覆盖计划四类性质：
① 稳健性：背景整体仿射变换（+c、×k）不改分——逐片稳健标准化的核心承诺；
② 单调性：斑更靠中心 / 更深 / 面积更大 → 分更低；
③ 边框剔除：宽窄不一的重边框不计惩罚；
④ 退化边界：常量图 / 纯渐变图 → 无斑满分，非法枚举报错。
"""

import numpy as np
import pytest

from fringe_scoring import load_config, score_fringe_distribution
from fringe_scoring.border import band_widths_px
from fringe_scoring.synth import make_glass_image


def base_config() -> dict:
    """测试用固定配置（不读仓库 yaml，避免调参影响性质断言）。"""
    return {
        "segment": {
            "method": "quantile",
            "top_fraction": 0.05,
            "min_z": 2.5,
            "polarity": "both",
            "z_threshold": 3.0,
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
        "weight": {"kind": "linear", "gaussian_sigma": 0.5, "distance_norm": "euclidean"},
    }


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

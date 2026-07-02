"""整床多片切分 · 性质测试（合成图，不依赖现场真值）。

覆盖性质：
① 检出与排序：多片网格全部检出、阅读序（先行后列）、角点定位精度；
② 逐片语义：带中心重斑的片分数严格低于干净片；
③ 稳健性：整床照片仿射变换（×k+c）不改任何一片的分数；
④ 抗粘连：斜放相邻两片不被闭运算合并；
⑤ 失败契约：空床 / 片数超上限 / 缺 sheets 配置段 → ValueError，绝不瞎猜。
"""

import numpy as np
import pytest

from fringe_scoring import detect_sheet_quads, score_sheets
from fringe_scoring.synth import make_bed_image, make_glass_image


def base_config() -> dict:
    """测试用固定配置（不读仓库 yaml，避免调参影响性质断言）。"""
    return {
        "sheets": {
            "edge_trim_frac": 0.002,
            "bg_quantile": 0.25,
            "fg_min_z": 6.0,
            "fg_min_rel": 0.25,
            "bright_ref_quantile": 0.995,
            "close_frac": 0.005,
            "min_area_frac": 0.01,
            "min_side_frac": 0.05,
            "max_sheets": 50,
            "poly_epsilon_frac": 0.02,
        },
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


def tilted_rect(x0: float, y0: float, w: float, h: float, dx: float = 6.0) -> np.ndarray:
    """轻微倾斜的四边形角点（左上→右上→右下→左下），模拟摆放不正的玻璃片。"""
    return np.array(
        [[x0 + dx, y0], [x0 + w, y0 + dx], [x0 + w - dx, y0 + h], [x0, y0 + h - dx]]
    )


BED_SHAPE = (420, 640)
FILL = 30.0
# 2 行 × 3 列网格；(0,1) 位置的片带中心重斑
GRID_CORNERS = [
    tilted_rect(30 + 200 * col, 25 + 200 * row, 170.0, 150.0)
    for row in range(2)
    for col in range(3)
]
BLOB_INDEX = 1  # 阅读序中第 2 片（第一行中间）


def make_grid_bed() -> np.ndarray:
    """六片整床合成照：干净片 × 5 + 中心重斑片 × 1（噪声长在床图上，见 make_bed_image）。"""
    sheets = []
    for i, corners in enumerate(GRID_CORNERS):
        blobs = [(0.5, 0.5, 0.12, 8.0)] if i == BLOB_INDEX else None
        sheets.append((make_glass_image(rows=120, cols=160, noise_sigma=0.0, blobs=blobs, seed=i), corners))
    return make_bed_image(sheets, BED_SHAPE, fill_value=FILL, noise_sigma=1.0, seed=42)


# ---------------- ① 检出与排序 ----------------
def test_grid_bed_all_sheets_detected_in_reading_order():
    bed = make_grid_bed()
    quads = detect_sheet_quads(bed, config=base_config())
    assert len(quads) == len(GRID_CORNERS)
    for detected, truth in zip(quads, GRID_CORNERS):
        # 阅读序一致 + 质心定位误差在闭运算/拟合的像素级容差内
        err = np.linalg.norm(detected.mean(axis=0) - truth.mean(axis=0))
        assert err < 15.0, f"质心偏差 {err:.1f}px 超容差"


# ---------------- ② 逐片语义 ----------------
def test_blob_sheet_scores_strictly_lowest():
    bed = make_grid_bed()
    res = score_sheets(bed, config=base_config())
    assert res.n_sheets == len(GRID_CORNERS)
    scores = res.scores
    blob_score = scores[BLOB_INDEX]
    clean_scores = [s for i, s in enumerate(scores) if i != BLOB_INDEX]
    assert blob_score < min(clean_scores) - 0.5
    assert res.score_min == pytest.approx(blob_score)


# ---------------- ③ 稳健性：整床仿射不变 ----------------
def test_bed_affine_invariance():
    bed = make_grid_bed()
    cfg = base_config()
    scores_a = score_sheets(bed, config=cfg).scores
    scores_b = score_sheets(1.7 * bed + 33.0, config=cfg).scores
    assert np.allclose(scores_a, scores_b, atol=1e-6)


# ---------------- ④ 抗粘连：斜放相邻片 ----------------
def test_adjacent_tilted_sheets_not_merged():
    # 两片平行四边形，最近处间隙 ~12px（> 闭运算核 3px），不得粘成一片
    left = tilted_rect(20.0, 20.0, 180.0, 200.0, dx=25.0)
    right = tilted_rect(212.0, 20.0, 180.0, 200.0, dx=25.0)
    sheets = [
        (make_glass_image(rows=150, cols=140, noise_sigma=0.0, seed=1), left),
        (make_glass_image(rows=150, cols=140, noise_sigma=0.0, seed=2), right),
    ]
    bed = make_bed_image(sheets, (260, 430), fill_value=FILL, noise_sigma=1.0, seed=7)
    quads = detect_sheet_quads(bed, config=base_config())
    assert len(quads) == 2


def test_thin_bright_streak_ignored():
    # 贯穿画幅的细亮线（真实照片实测干扰）：面积过 min_area_frac 但短边过细 → 剔除不当玻璃
    left = tilted_rect(20.0, 20.0, 180.0, 200.0, dx=25.0)
    right = tilted_rect(212.0, 20.0, 180.0, 200.0, dx=25.0)
    sheets = [
        (make_glass_image(rows=150, cols=140, noise_sigma=0.0, seed=1), left),
        (make_glass_image(rows=150, cols=140, noise_sigma=0.0, seed=2), right),
    ]
    bed = make_bed_image(sheets, (260, 430), fill_value=FILL, noise_sigma=1.0, seed=7)
    bed[246:252, :] = 200.0  # 细亮线：6px 高、横贯整幅
    quads = detect_sheet_quads(bed, config=base_config())
    assert len(quads) == 2


# ---------------- ⑤ 失败契约 ----------------
def test_blank_bed_raises():
    rng = np.random.default_rng(0)
    blank = FILL + rng.normal(0.0, 1.0, size=BED_SHAPE)
    with pytest.raises(ValueError, match="未检出任何玻璃片"):
        detect_sheet_quads(blank, config=base_config())


def test_max_sheets_exceeded_raises():
    # 片数超 max_sheets 上限 → 判过分割，报错拒绝打分
    cfg = base_config()
    cfg["sheets"]["max_sheets"] = 3
    with pytest.raises(ValueError, match="max_sheets"):
        detect_sheet_quads(make_grid_bed(), config=cfg)


def test_missing_sheets_config_raises():
    # 配置缺 sheets 段 → 多片入口显式报错，不静默走单片路径
    cfg = base_config()
    del cfg["sheets"]
    with pytest.raises(ValueError, match="sheets 段"):
        score_sheets(make_grid_bed(), config=cfg)

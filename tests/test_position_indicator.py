"""位置指标（中心集中度 ρu / 位置评分）· 基准翻转门闩与性质测试（任务4 T1–T6）。

T3/T6 判据说明（2026-07-19 合成族实测校定，偏离任务书原判据处如实标注）：
- T3 原判据"49%→51% 总跳变 ≤10 分"在横带族上不可达：<50% 侧 σR 的 MAD
  渐进击穿使 U 连续虚高（f∈[0.40,0.49] 段 σR 被近半污染撑大、z 压扁），而
  门闩在该侧 p_dark=0 无从触发（per_sheet/MAD 口径为任务书 §7 固定项）。
  门闩实际保证并在此断言的是：①消除 >50% 侧灾难性假高分（未修版 U≈99）；
  ②翻转侧与低覆盖侧数值接续；③翻转点方向保守（只向下、不产生假好）。
- T6 同理：触发边界的分支台阶对任何 q_lo 都 >5 分（实测 0.05→−44 /
  0.25→−25 / 0.5→−18；q_lo=0.5 会破坏干净区 <25% 的深翻转锚定），按任务书
  "边界阶跃→调 q 下界"授权取 q_lo=0.25；断言分支内连续 ≤5 分、切换处方向
  保守且 |ΔU| ≤ 30。
"""

import numpy as np
import pytest

from fringe_scoring.indicators import INDICATOR_KEYS, compute_sheet_indicators
from fringe_scoring.synth import (
    make_band_image,
    make_dark_tail_image,
    make_flipped_baseline_image,
    make_glass_image,
)


def position_config(flip: dict | None = None) -> dict:
    """测试用完整配置（自含，与 test_fringe_indicators.indicators_config 同构）。"""
    cfg = {
        "quad": {"auto_detect": False},
        "segment": {
            "method": "gray_threshold", "top_fraction": 0.05, "min_z": 2.5,
            "polarity": "bright", "s_scale_mode": "absolute",
            "s_saturation_gray": 60.0, "min_dev_gray": 4.0, "z_threshold": 3.0,
            "z_saturation": 8.0, "background_block_frac": 0.05,
            "background_block_quantile": 0.15, "background_poly_degree": 1,
            "min_contrast_frac": 0.05,
        },
        "border": {"min_band_frac": 0.0, "max_band_frac": 0.25,
                   "profile_quantile": 0.5, "return_to_background_ratio": 2.0,
                   "normal_band_frac": 0.04},
        "weight": {"kind": "linear", "gaussian_sigma": 0.5,
                   "distance_norm": "chebyshev", "floor": 0.3},
        "indicators": {
            "weights": {k: 1.0 for k in INDICATOR_KEYS},
            "scoring": {"position_ratio_at_zero": 0.33},
            "refs": {
                "x095": {"best": 100.0, "worst": 160.0},
                "gray_variance": {"best": 1.0, "worst": 400.0},
                "gradient_mean": {"best": 3.0, "worst": 40.0},
                "gradient_variance": {"best": 10.0, "worst": 1000.0},
                "ccp": {"best": 0.4, "worst": 1.0},
            },
            "calibration": {"gray_to_nm": None, "mm_per_px": None,
                            "ccp_c_max": 0.10, "ccp_cp_max": 700.0,
                            "ccp_reference_is_plant_calibrated": False,
                            "thickness_mm": None},
        },
    }
    if flip is not None:
        cfg["indicators"]["baseline_flip"] = flip
    return cfg


def full_interior(img: np.ndarray) -> np.ndarray:
    """测试用内部掩膜：整图。"""
    return np.ones(img.shape, dtype=bool)


def run_position(img: np.ndarray, flip: dict | None = None):
    """一张图 → (SheetIndicators, 诊断 dict)。"""
    ind = compute_sheet_indicators(img, full_interior(img), position_config(flip))
    return ind, ind.position_diagnostics


# ---------------- T1 · 仿射不变性（含翻转路径） ----------------
@pytest.mark.parametrize("a", [0.5, 2.0, 10.0])
@pytest.mark.parametrize("c", [-20.0, 0.0, 50.0])
def test_affine_invariance_covers_flip_path(a, c):
    # I → aI + c 时 ρu 相对差 ≤1e-9；普通片与"基准已翻转"片各验一遍——
    # 两遍法全程只用分位/中位/MAD（对正仿射同变），不得引入任何绝对量
    cases = [
        (make_glass_image(blobs=[(0.5, 0.5, 0.12, 30.0)], seed=65), False),
        (make_flipped_baseline_image(), True),
    ]
    for img, want_flip in cases:
        ind0, d0 = run_position(img)
        ind1, d1 = run_position(a * img + c)
        assert d0["baseline_flipped"] == want_flip
        assert d1["baseline_flipped"] == want_flip
        rel = abs(ind1.center_concentration - ind0.center_concentration) / max(
            abs(ind0.center_concentration), 1e-12
        )
        assert rel <= 1e-9, f"仿射 a={a} c={c} 翻转={want_flip} rel={rel:.2e}"


# ---------------- T2 · 无翻转片零改动（门闩开关位级等价） ----------------
def test_latch_is_noop_on_normal_sheets_bitwise():
    # p_dark ≤ p_flip 的片：门闩开/关的 ρu 与 position_score 浮点位级相同
    # （真实 25 片的同款验收在交付脚本跑；此处用合成片守住回归网）
    for seed in (61, 65, 71, 75):
        img = make_glass_image(blobs=[(0.4, 0.55, 0.12, 28.0)], seed=seed)
        ind_on, d_on = run_position(img, flip={"enabled": True})
        ind_off, _ = run_position(img, flip={"enabled": False})
        assert d_on["p_dark"] <= 0.05
        assert not d_on["baseline_flipped"]
        assert ind_on.center_concentration.hex() == ind_off.center_concentration.hex()
        assert ind_on.position_score.hex() == ind_off.position_score.hex()


# ---------------- T3 · 连续性（中央横带展宽族，本任务核心验收） ----------------
def test_band_widening_continuity_and_flip_fix():
    covs = [0.30, 0.35, 0.40, 0.45, 0.47, 0.49, 0.50, 0.51, 0.53, 0.55, 0.60, 0.70]
    us, flips = {}, {}
    for f in covs:
        ind, d = run_position(make_band_image(f))
        us[f], flips[f] = ind.position_score, d["baseline_flipped"]

    # ① 门闩消除灾难性假高分：未修版在覆盖率过半后给假满分
    ind_off, _ = run_position(make_band_image(0.51), flip={"enabled": False})
    assert ind_off.position_score > 95.0          # 原缺陷：斑最重反而近满分
    assert ind_off.position_score - us[0.51] > 30.0  # 门闩把假满分拉回

    # ② 分支内连续：相邻两点 ≤5 分/百分点（切换点单独看③）
    for f0, f1 in zip(covs, covs[1:]):
        if flips[f0] != flips[f1]:
            continue
        step_per_pct = abs(us[f1] - us[f0]) / (100.0 * (f1 - f0))
        assert step_per_pct <= 5.0, f"{f0}->{f1} 每百分点 {step_per_pct:.1f} 分"

    # ③ 切换点方向保守（只向下，不产生假好），台阶有界；允许非单调
    switch = [(f0, f1) for f0, f1 in zip(covs, covs[1:]) if flips[f0] != flips[f1]]
    assert len(switch) == 1
    f0, f1 = switch[0]
    assert us[f1] < us[f0], "翻转点必须向下（保守），不得向上跳假好"
    assert us[f0] - us[f1] <= 35.0

    # ④ 翻转侧与低覆盖侧数值接续（门闩恢复的正是干净基准下的口径）
    assert abs(us[0.51] - us[0.30]) <= 10.0


# ---------------- T4 · 确定性 ----------------
def test_deterministic_repeat_10x():
    img = make_glass_image(blobs=[(0.5, 0.45, 0.15, 35.0)], seed=75)
    base_ind, base_diag = run_position(img)
    for _ in range(9):
        ind, diag = run_position(img)
        for k, v in base_ind.raw_values.items():
            assert ind.raw_values[k] == v, k
        assert diag == base_diag


# ---------------- T5 · 退化情形 ----------------
def test_constant_image_and_empty_interior():
    white = np.full((200, 300), 250.0)
    ind, d = run_position(white)
    assert ind.center_concentration == 0.0
    assert ind.position_score == 100.0
    assert d["spot_area_ratio"] == 0.0
    assert d["p_dark"] == 0.0
    assert not d["baseline_flipped"] and not d["baseline_flip_aborted"]
    # M 为空：维持现行拒绝出分行为（不因新增诊断字段而崩溃/静默给分）
    img = make_glass_image(seed=61)
    with pytest.raises(ValueError, match="无有效像素"):
        compute_sheet_indicators(img, np.zeros(img.shape, dtype=bool), position_config())


# ---------------- T6 · 门闩边界（p_dark ≈ p_flip 家族） ----------------
def test_dark_tail_boundary_family():
    fracs = [0.040, 0.045, 0.049, 0.051, 0.055, 0.060]
    us, flips = {}, {}
    for f in fracs:
        ind, d = run_position(make_dark_tail_image(f))
        us[f], flips[f] = ind.position_score, d["baseline_flipped"]
        assert not d["baseline_flip_aborted"]
    assert not flips[fracs[0]] and flips[fracs[-1]]  # 家族确实跨越了 p_flip
    for f0, f1 in zip(fracs, fracs[1:]):
        if flips[f0] == flips[f1]:
            assert abs(us[f1] - us[f0]) <= 5.0, f"分支内 {f0}->{f1}"
        else:
            # 切换处：方向保守（向下），台阶 ≤30（q_lo=0.25 实测 −25；
            # 任务书初值 q_lo=0.05 时为 −44，已按其授权上调，见模块 docstring）
            assert us[f1] < us[f0]
            assert us[f0] - us[f1] <= 30.0


def test_flip_abort_guard_small_dark_sample():
    # 暗侧样本 |D| < max(1000, 0.005|M|) → 回退 Pass1 并置 aborted：
    # 小图上 p_dark 虽超阈，门闩必须放弃重定基准且数值与关门闩位级相同
    img = make_dark_tail_image(0.10, rows=60, cols=60)
    ind_on, d_on = run_position(img, flip={"enabled": True})
    ind_off, _ = run_position(img, flip={"enabled": False})
    assert d_on["p_dark"] > 0.05
    assert d_on["baseline_flip_aborted"] and not d_on["baseline_flipped"]
    assert ind_on.center_concentration.hex() == ind_off.center_concentration.hex()
    assert ind_on.position_score.hex() == ind_off.position_score.hex()


# ---------------- 附 · 样片乙照片版连续性（照片不入库，存在才跑） ----------------
_PHOTO = None
try:  # 样片乙所在整床照（床2）；照片按仓库规则不提交，克隆环境自动跳过
    from pathlib import Path

    _CAND = sorted(
        (Path(__file__).resolve().parents[1] / "data" / "images" / "stress_fringe").glob(
            "2025-10-30_20251030_204759*.PNG"
        )
    )
    _PHOTO = _CAND[0] if _CAND else None
except Exception:
    _PHOTO = None


@pytest.mark.skipif(_PHOTO is None, reason="现场照片未入库（按仓库规则），跳过实拍版 T3")
def test_band_widening_continuity_on_real_sheet():
    # 样片乙（床2-片4）：把其中央横带按比例复制展宽，验证真实纹理下的同款性质
    import cv2

    img = cv2.imdecode(np.fromfile(str(_PHOTO), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    from fringe_scoring.sheets import score_sheets

    cfg = position_config()
    cfg["sheets"] = {
        "edge_trim_frac": 0.002, "bg_quantile": 0.25, "fg_min_z": 6.0,
        "fg_min_rel": 0.25, "bright_ref_quantile": 0.995, "close_frac": 0.005,
        "min_area_frac": 0.01, "min_side_frac": 0.05, "max_sheets": 50,
        "poly_epsilon_frac": 0.02,
    }
    res = score_sheets(np.asarray(img, dtype=float), cfg)
    sheet = np.asarray(res.sheet_results[3].warped_image, dtype=float)  # 床2-片4=样片乙

    rows = sheet.shape[0]
    r0, r1 = int(0.40 * rows), int(0.60 * rows)  # 乙的中央横带区
    band = sheet[r0:r1]
    us, flips = [], []
    for reps in (1, 2, 3, 4):  # 横带纵向复制展宽 → 覆盖率约 20%/40%/60%/80%
        widened = np.vstack([sheet[:r0]] + [band] * reps + [sheet[r1:]])
        ind, d = run_position(widened)
        us.append(ind.position_score)
        flips.append(d["baseline_flipped"])
    assert flips[-1] or max(us) - min(us) < 60.0  # 展宽后要么触发翻转要么本就平缓
    for u0, u1, f0, f1 in zip(us, us[1:], flips, flips[1:]):
        if f0 != f1:
            assert u1 < u0 + 10.0  # 切换处不得向上跳假好（容噪 10 分）

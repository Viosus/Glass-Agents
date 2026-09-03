"""sample26 资产生成器纯函数单测（不依赖照片与 GlassApp 包，缺图环境可跑）。

被测函数全部在模块顶层且不触发 GlassApp 载入（_glassapp_api 是运行期惰性载入，
见该模块 docstring）；本文件覆盖：裁切覆盖度与 aniso（含 13# 错检回归）、
QA 分级边界、分档统计防污染、refs 草案对 n=1 档拒绝出值。
"""

import pytest
import yaml

from fringe_scoring.make_sample26_assets import (
    BAND_REF_KEYS,
    _band_stats,
    _crop_check,
    _qa_check,
    _refs_draft,
    _spearman,
    _stats_eligible,
)

QA = {"aniso_warn": 0.05, "aniso_fail": 0.10, "coverage_lo": 0.85, "coverage_hi": 1.02}


# ---------------------------------------------------------------- 裁切覆盖度
def test_crop_check_square_spec():
    """610×610 规格、605×605 矫正 → 覆盖度 605/610（内咬 ~0.8%），aniso=0。"""
    m = _crop_check([610, 610], 605, 605)
    assert m["coverage_long"] == m["coverage_short"] == pytest.approx(605 / 610, abs=1e-4)
    assert m["aniso"] == 0.0


def test_crop_check_orientation_free():
    """长边配长边：矫正图横放/竖放（504×358 vs 358×504）结果必须相同。"""
    a = _crop_check([510, 360], 504, 358)
    b = _crop_check([510, 360], 358, 504)
    assert a == b
    assert a["coverage_long"] == pytest.approx(504 / 510, abs=1e-4)


def test_crop_check_13_regression():
    """13# 错检回归：610×610 方片检成 643×350 → aniso 巨大，QA 必 fail。"""
    m = _crop_check([610, 610], 643, 350)
    assert m["aniso"] == pytest.approx(0.8371, abs=1e-3)  # 远超 fail 阈 0.10
    qa = _qa_check(1, 1, m, "auto", QA)
    assert qa["level"] == "fail"


# ---------------------------------------------------------------- QA 分级
def test_qa_levels():
    """pass/warn/fail 三级边界：容差取自 manifest（阈值处不含等号）。"""
    ok = {"coverage_long": 0.99, "coverage_short": 0.99, "aniso": 0.0}
    assert _qa_check(1, 1, ok, "auto", QA)["level"] == "pass"
    warn = dict(ok, aniso=0.06)
    assert _qa_check(1, 1, warn, "auto", QA)["level"] == "warn"
    fail = dict(ok, aniso=0.12)
    assert _qa_check(1, 1, fail, "auto", QA)["level"] == "fail"
    # 阈值恰好命中不升级（判据是严格大于）
    assert _qa_check(1, 1, dict(ok, aniso=0.05), "auto", QA)["level"] == "pass"


def test_qa_special_paths():
    """检测失败=fail；片数不符=fail；整图口径=warn；覆盖度出域=fail。"""
    assert _qa_check(0, 1, None, "detection_failed", QA)["level"] == "fail"
    assert _qa_check(2, 1, None, "auto", QA)["level"] == "fail"
    ok = {"coverage_long": 0.99, "coverage_short": 0.99, "aniso": 0.0}
    assert _qa_check(1, 1, ok, "fullframe", QA)["level"] == "warn"
    far = {"coverage_long": 0.5, "coverage_short": 0.5, "aniso": 0.0}
    assert _qa_check(1, 1, far, "auto", QA)["level"] == "fail"


# ---------------------------------------------------------------- 统计防污染
def _rec(sid, cat="core", t=6, qa="pass", prov="auto", n=1, with_ind=True, w=0.2):
    ind = None
    if with_ind:
        ind = {k: 1.0 for k in BAND_REF_KEYS}
        ind.update(texture_w=w, position_score=50.0, total_score=50.0)
    return {"sample_id": sid, "file": f"{sid}.png", "category": cat, "thickness_mm": t,
            "qa": {"level": qa, "reasons": []},
            "detection": {"provenance": prov, "n_sheets": n}, "indicators": ind}


def test_stats_eligible_filters():
    """special/control/QA fail/整图/多片/无指标 六类全部被拒并记录原因。"""
    records = [
        _rec("core-ok"),
        _rec("spec", cat="special"),
        _rec("ctrl", cat="control"),
        _rec("qafail", qa="fail"),
        _rec("full", prov="fullframe"),
        _rec("multi", n=2),
        _rec("noind", with_ind=False),
    ]
    eligible, excluded = _stats_eligible(records)
    assert [r["sample_id"] for r in eligible] == ["core-ok"]
    assert len(excluded) == 6
    assert all(e["reasons"] for e in excluded)


def test_stats_conservation():
    """入围 + 排除 = 总数（守恒，不允许静默丢样）。"""
    records = [_rec(f"s{i}") for i in range(5)] + [_rec("x", cat="special")]
    eligible, excluded = _stats_eligible(records)
    assert len(eligible) + len(excluded) == len(records)


# ---------------------------------------------------------------- 分档与草案
def _band_input(pairs):
    """[(厚度, [W_w…])] → _band_stats 入参（其余指标用 W_w 平移填充，保证 min<max）。"""
    recs = []
    for t, ws in pairs:
        for i, w in enumerate(ws):
            r = _rec(f"t{t}n{i}", t=t, w=w)
            for k in BAND_REF_KEYS:
                r["indicators"][k] = w if k == "texture_w" else w + 0.1 * (i + 1)
            recs.append(r)
    return recs


def test_band_stats_grouping():
    """按厚度分组，n/成员/min/max/mean 正确。"""
    bands = _band_stats(_band_input([(4, [0.17, 0.18]), (6, [0.23, 0.25, 0.27])]))
    assert [b["thickness_mm"] for b in bands] == [4, 6]
    b6 = bands[1]
    assert b6["n"] == 3
    assert b6["texture_w"]["min"] == pytest.approx(0.23)
    assert b6["texture_w"]["max"] == pytest.approx(0.27)


def test_refs_draft_excludes_n1():
    """n=1 档（15mm 不在 DRAFT_BANDS_MM）不出 refs；n≥2 档 best<worst 全键齐。"""
    bands = _band_stats(_band_input([(4, [0.17, 0.18]), (5, [0.15, 0.20]),
                                     (6, [0.23, 0.25]), (8, [0.23, 0.25]),
                                     (15, [0.30])]))
    draft = _refs_draft(bands)
    out = draft["refs_by_thickness"]
    assert [b["max_thickness_mm"] for b in out] == [4.0, 5.0, 6.0, 8.0]
    for band in out:
        assert set(band["refs"]) == set(BAND_REF_KEYS)
        for ref in band["refs"].values():
            assert ref["best"] < ref["worst"]
    # 可被 yaml 回读且结构同 select_refs 消费格式
    back = yaml.safe_load(yaml.safe_dump(draft))
    assert back["refs_by_thickness"][0]["refs"]["texture_w"]["best"] == pytest.approx(0.17)


def test_refs_draft_rejects_degenerate():
    """档内某指标 best==worst（映射退化）必须拒绝出草案，不静默放行。"""
    bands = _band_stats(_band_input([(4, [0.17, 0.17])]))  # W_w 两片同值
    with pytest.raises(AssertionError):
        _refs_draft(bands)


# ---------------------------------------------------------------- 秩相关
def test_spearman_ties_average_rank():
    """并列平均秩口径：厚度大量并列时结果与 scipy 一致（此处用手算锚点）。"""
    # x=[4,4,6,6]（两组并列）, y 单调 → ρ = 0.8944（完全单调但组内并列稀释）
    assert _spearman([4, 4, 6, 6], [1, 2, 3, 4]) == pytest.approx(0.8944, abs=1e-3)
    assert _spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

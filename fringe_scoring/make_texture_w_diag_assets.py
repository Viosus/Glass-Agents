"""W_w 口径敏感性与工程可分辨性诊断资产（开发壳，不随核心包交付）。

回答三个在 V1.2 定稿后提出的问题，全部落盘供技术说明引用（图文同源）：

1. **步距扫描**：GLCM 步距 d 从 1 mm 扫到 100 mm，看两支路的量级平衡与判别力如何变化。
   动机：读法A 定 d=1 mm，而应力斑是几十毫米尺度现象；需要实测这个选择的代价。
2. **可信区间分层**：按饱和占比三分位分组，组内 W 与人工分的秩相关。
   动机：饱和裁剪与判别力高度纠缠，须查明"扣掉伪影后指标在哪个区间还站得住"。
3. **工程可分辨性**：对比度支路（支路1）对 W 的贡献幅度 vs 名次间距、名次重排量、
   报数精度门槛、判级翻转率。动机：支路1 只贡献 0.2~0.4% 的方差，须回答
   "它到底能不能在最终分数里被分辨出来"。
   另附形式选择的乐观偏差估计（对半选择/评估）。

数据源：115 片人工标注集（原生分辨率现算）+ 两份既有资产的逐片记录
（texture_w_doc / texture_w_field 的 values.json）。
用法：venv python fringe_scoring/make_texture_w_diag_assets.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fringe_scoring.sheets import score_sheets  # noqa: E402
from tools import metrics  # noqa: E402

PHOTO_DIR = ROOT / "data" / "images" / "stress_fringe"
CFG_PATH = ROOT / "config" / "fringe_scoring.yaml"
OUT = ROOT / "data" / "derived" / "texture_w_diag"
DOC_ASSET = ROOT / "data" / "derived" / "texture_w_doc" / "values.json"
FIELD_ASSET = ROOT / "data" / "derived" / "texture_w_field" / "values.json"
STEPS = (1, 2, 5, 10, 20, 50, 100)          # GLCM 步距（px = mm，因 mm/px=1）
CA_SUP, CP_SUP = metrics._ca_sup(8), metrics._cp_sup(8)

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _avg_ranks(v: np.ndarray) -> np.ndarray:
    """并列取**平均秩**（scipy rankdata 'average' 口径，纯 numpy）。"""
    order = np.argsort(v, kind="mergesort")
    pos = np.empty(v.size, dtype=float)
    pos[order] = np.arange(v.size, dtype=float)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=pos) / cnt)[inv]


def _spearman(x, y) -> float:
    """Spearman 秩相关，**并列取平均秩**（2026-08-04 更正，与人工分档并列相容）；
    任一侧为常量返回 nan。原 argsort 口径对 5 分一档的人工分依赖枚举顺序，已作废。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(_avg_ranks(x), _avg_ranks(y))[0, 1])


def _glcm_at_step(img: np.ndarray, wmap: np.ndarray, d: int,
                  ng: int = 8) -> tuple[float, float] | None:
    """位置加权 GLCM，步距 d（像素≡毫米），四方向平均 → (Ca_w, CPa_w)。

    与 tools.metrics._glcm_ca_cpa_weighted 同口径，仅把固定的 d=1 偏移改为参数化，
    用于步距敏感性扫描（不改变生产实现）。
    """
    arr = np.asarray(img, float)
    q = metrics._quantize_minmax(arr, ng).astype(np.intp)
    w = np.asarray(wmap, float)
    rows, cols = arr.shape
    i_idx = np.arange(ng).reshape(-1, 1)
    j_idx = np.arange(ng).reshape(1, -1)
    cas, cps = [], []
    for dr, dc in ((0, d), (-d, d), (-d, 0), (-d, -d)):
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        if r1 <= r0 or c1 <= c0:
            return None
        a = q[r0:r1, c0:c1].ravel()
        b = q[r0 + dr: r1 + dr, c0 + dc: c1 + dc].ravel()
        wa = w[r0:r1, c0:c1].ravel()
        wb = w[r0 + dr: r1 + dr, c0 + dc: c1 + dc].ravel()
        cnt = np.bincount(a * ng + b, weights=0.5 * (wa + wb),
                          minlength=ng * ng).reshape(ng, ng)
        cnt = cnt + cnt.T
        total = cnt.sum()
        if total <= 0:
            return None
        p = cnt / total
        cas.append(float((p * (i_idx - j_idx) ** 2).sum()))
        mu_i = float((i_idx * p).sum()); mu_j = float((j_idx * p).sum())
        cps.append(float((p * (i_idx + j_idx - mu_i - mu_j) ** 4).sum()))
    return float(np.mean(cas)), float(np.mean(cps))


def _step_sweep(cfg: dict) -> dict:
    """诊断 1：步距扫描（115 片标注集，原生分辨率）。"""
    rows = []
    t0 = time.time()
    for p in sorted(PHOTO_DIR.glob("*.png")):
        m = re.match(r"(\d+)_系统([\d.]+)_人工([\d.]+)", p.stem)
        if not m:
            continue
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        try:
            res = score_sheets(np.asarray(img, float), cfg)
        except ValueError:
            continue
        warped = np.asarray(res.sheet_results[0].warped_image, float)
        wmap = metrics._position_weight_map(warped.shape)
        rec: dict = {"man": float(m.group(3))}
        for d in STEPS:
            rec[str(d)] = _glcm_at_step(warped, wmap, d)
        rows.append(rec)
    print(f"  步距扫描：{len(rows)} 片 × {len(STEPS)} 个步距（{time.time() - t0:.0f}s）")

    table = []
    for d in STEPS:
        ok = [r for r in rows if r.get(str(d))]
        if len(ok) < 50:
            continue
        ca = np.array([r[str(d)][0] for r in ok])
        cpa = np.array([r[str(d)][1] for r in ok])
        man = np.array([r["man"] for r in ok])
        b1 = (ca / CA_SUP) ** 0.5
        b2 = (cpa / CP_SUP) ** 0.25
        w = 0.5 * (b1 + b2)
        cov = np.cov(np.vstack([0.5 * b1, 0.5 * b2]))
        table.append({
            "d_mm": d, "n": len(ok),
            "ca_median": round(float(np.median(ca)), 4),
            "ca_frac_of_bound": round(float(np.median(ca) / CA_SUP), 5),
            "branch1_median": round(float(np.median(b1)), 4),
            "branch2_median": round(float(np.median(b2)), 4),
            "ratio_b2_over_b1": round(float(np.median(b2) / np.median(b1)), 1),
            "rho_ca_only": round(_spearman(b1, man), 3),
            "rho_cpa_only": round(_spearman(b2, man), 3),
            "rho_combined": round(_spearman(w, man), 3),
            "branch1_var_share": round(float(cov[0, 0] / w.var()), 4),
        })
    best = max(table, key=lambda r: abs(r["rho_combined"]))
    return {
        "table": table,
        "current_d_mm": 1,
        "best_d_mm": best["d_mm"],
        "rho_at_current": next(r["rho_combined"] for r in table if r["d_mm"] == 1),
        "rho_at_best": best["rho_combined"],
        "note": "步距 d 以毫米计（mm/px=1 故 px≡mm）。判别力在 d≈20~50 mm 出峰，"
                "与应力斑的几十毫米特征尺度一致；但该优势（约 0.02）远小于饱和裁剪的"
                "影响量级，且未控制饱和，故不足以支撑改动已定的读法A（d=1 mm）",
    }


def _stratified_by_saturation() -> dict:
    """诊断 2：按饱和占比三分位分组，组内 W 与人工分的秩相关（可信区间）。"""
    L = json.loads(DOC_ASSET.read_text(encoding="utf-8"))
    recs = L["all"]
    man = np.array([r["man"] for r in recs])
    sat = np.array([r["sat_frac"] for r in recs])
    w = np.array([r["texture_w"] for r in recs])
    qs = np.quantile(sat, [0, 1 / 3, 2 / 3, 1.0])
    groups = []
    for i in range(3):
        lo, hi = qs[i], qs[i + 1]
        sel = (sat >= lo) & (sat <= hi) if i == 2 else (sat >= lo) & (sat < hi)
        groups.append({
            "sat_lo": round(float(lo), 5), "sat_hi": round(float(hi), 5),
            "n": int(sel.sum()),
            "rho_w_vs_man": round(_spearman(w[sel], man[sel]), 3),
            "rho_sat_vs_man": round(_spearman(sat[sel], man[sel]), 3),
            "man_lo": float(man[sel].min()), "man_hi": float(man[sel].max()),
        })
    return {
        "groups": groups,
        "note": "组内饱和度已拉平，此时 W 仍有的相关性是它自己挣的。"
                "低饱和组（全为好片、人工分跨度窄）相关性塌缩，说明指标在"
                "'好片之间挑更好的'这一段基本失效；中高饱和组仍强",
    }


def _branch_resolvability() -> dict:
    """诊断 3：支路1 在最终分数中是否可分辨（幅度/重排/报数精度/判级翻转）。"""
    F = json.loads(FIELD_ASSET.read_text(encoding="utf-8"))
    ca = np.array([r["ca_w"] for r in F["all"]])
    cpa = np.array([r["cpa_w"] for r in F["all"]])
    b1 = (ca / CA_SUP) ** 0.5
    b2 = (cpa / CP_SUP) ** 0.25
    w = 0.5 * (b1 + b2)
    w_frozen = 0.5 * (b2 + np.median(b1))   # 冻结支路1 → 只剩支路2 在动
    n = len(w)
    contrib = 0.5 * b1
    gaps = np.diff(np.sort(w))

    r_full = np.argsort(np.argsort(w))
    r_froz = np.argsort(np.argsort(w_frozen))
    # 逆序对占比（两套排序的 Kendall tau 距离）
    sign = np.sign(np.subtract.outer(w, w)) * np.sign(np.subtract.outer(w_frozen, w_frozen))
    iu = np.triu_indices(n, 1)
    inversions = int((sign[iu] < 0).sum())

    precision = {str(nd): int((np.round(w, nd) != np.round(w_frozen, nd)).sum())
                 for nd in (1, 2, 3, 4)}
    grade_flip = {}
    for name, qs in (("tertile", [1 / 3, 2 / 3]), ("p20_p80", [0.2, 0.8])):
        lo, hi = np.quantile(w, qs)
        flip = int((np.digitize(w, [lo, hi]) != np.digitize(w_frozen, [lo, hi])).sum())
        grade_flip[name] = {"lines": [round(float(lo), 4), round(float(hi), 4)],
                            "n_flipped": flip, "frac": round(flip / n, 4)}
    return {
        "n": n,
        "branch1_contrib_median": round(float(np.median(contrib)), 5),
        "branch1_contrib_std": round(float(contrib.std()), 5),
        "branch2_contrib_median": round(float(np.median(0.5 * b2)), 5),
        "branch2_contrib_std": round(float((0.5 * b2).std()), 5),
        "neighbor_gap_median": round(float(np.median(gaps)), 6),
        "std_over_gap": round(float(contrib.std() / np.median(gaps)), 1),
        "n_rank_changed": int((r_full != r_froz).sum()),
        "max_rank_move": int(np.abs(r_full - r_froz).max()),
        "inversion_frac": round(inversions / (n * (n - 1) / 2), 5),
        "reported_differs_by_decimals": precision,
        "min_decimals_recommended": 3,
        "grade_flip": grade_flip,
        "note": "支路1 的方差占比虽小（<1%），但其 1σ 幅度约为相邻名次间距的 11 倍，"
                "故在分数中可分辨。前提：报数精度至少 3 位小数——保留 2 位时 71% 的片"
                "看不出支路1 的作用",
    }


def _selection_bias() -> dict:
    """诊断 3 附：形式选择的乐观偏差（对半选择/评估 1000 次）。"""
    L = json.loads(DOC_ASSET.read_text(encoding="utf-8"))
    recs = L["all"]
    man = np.array([r["man"] for r in recs])
    ca = np.array([r["ca_w"] for r in recs]); cpa = np.array([r["cpa_w"] for r in recs])
    forms = {
        "cpa_only": (cpa / CP_SUP) ** 0.25,
        "ca_only": (ca / CA_SUP) ** 0.5,
        "dual_bound": 0.5 * ((ca / CA_SUP) ** 0.5 + (cpa / CP_SUP) ** 0.25),
        "geometric": ((ca / CA_SUP) ** 0.5 * (cpa / CP_SUP) ** 0.25) ** 0.5,
    }
    rng = np.random.default_rng(0)
    n = len(man)
    ins, held, picks = [], [], {}
    for _ in range(1000):
        idx = rng.permutation(n)
        A, B = idx[: n // 2], idx[n // 2:]
        best = max(forms, key=lambda k: abs(_spearman(forms[k][A], man[A])))
        picks[best] = picks.get(best, 0) + 1
        ins.append(abs(_spearman(forms[best][A], man[A])))
        held.append(abs(_spearman(forms[best][B], man[B])))
    return {
        "n_splits": 1000,
        "in_sample_mean_abs_rho": round(float(np.mean(ins)), 4),
        "held_out_mean_abs_rho": round(float(np.mean(held)), 4),
        "optimism_bias": round(float(np.mean(ins) - np.mean(held)), 4),
        "form_picked_counts": dict(sorted(picks.items(), key=lambda kv: -kv[1])),
        "note": "形式评比与最终报数用了同一批 115 片，故存在乐观偏差；实测约 +0.01，"
                "可接受。但纯数据驱动的选择更常选中其它形式——现行形式的选定"
                "还权衡了病态行为，不是纯相关性最优",
    }


def main() -> int:
    """跑三项诊断 → 落盘 values.json 并打印摘要。"""
    if not PHOTO_DIR.exists() or not any(PHOTO_DIR.glob("*.png")):
        print(f"[跳过] 未找到标注照片：{PHOTO_DIR}")
        return 0
    if not (DOC_ASSET.exists() and FIELD_ASSET.exists()):
        print("[跳过] 需先跑 make_texture_w_assets.py 与 make_texture_w_field_assets.py")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

    values = {
        "step_sweep": _step_sweep(cfg),
        "stratified_by_saturation": _stratified_by_saturation(),
        "branch_resolvability": _branch_resolvability(),
        "selection_bias": _selection_bias(),
        "_口径说明": "全部基于原生分辨率（mm/px=1，px≡mm）；幅度为 8-bit 强度域",
    }
    (OUT / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")

    ss = values["step_sweep"]
    print("\n① 步距扫描（判别力 |ρ| vs 人工分）")
    print(f"   {'d(mm)':>6s}{'Ca中位':>9s}{'占上界':>8s}{'量级比':>7s}"
          f"{'仅Ca':>8s}{'仅CPa':>8s}{'组合':>8s}{'支路1方差':>10s}")
    for r in ss["table"]:
        print(f"   {r['d_mm']:>6d}{r['ca_median']:>9.4f}{r['ca_frac_of_bound']:>8.2%}"
              f"{r['ratio_b2_over_b1']:>7.1f}{r['rho_ca_only']:>8.3f}"
              f"{r['rho_cpa_only']:>8.3f}{r['rho_combined']:>8.3f}"
              f"{r['branch1_var_share']:>10.1%}")
    print(f"   → 现行 d={ss['current_d_mm']}mm ρ={ss['rho_at_current']}；"
          f"峰值在 d={ss['best_d_mm']}mm ρ={ss['rho_at_best']}")

    st = values["stratified_by_saturation"]
    print("\n② 饱和分层（组内相关性 = 扣掉伪影后指标自己挣的）")
    for g in st["groups"]:
        print(f"   饱和 {g['sat_lo']:.2%}~{g['sat_hi']:.2%}（n={g['n']:3d}，"
              f"人工 {g['man_lo']:.0f}~{g['man_hi']:.0f}）："
              f"W~人工 {g['rho_w_vs_man']:+.3f}   饱和~人工 {g['rho_sat_vs_man']:+.3f}")

    br = values["branch_resolvability"]
    print("\n③ 支路1 可分辨性")
    print(f"   贡献 1σ={br['branch1_contrib_std']} = 相邻名次间距的 {br['std_over_gap']} 倍")
    print(f"   名次变化 {br['n_rank_changed']}/{br['n']}，最大移动 {br['max_rank_move']} 位，"
          f"逆序对 {br['inversion_frac']:.2%}")
    print(f"   报数精度：" + "；".join(
        f"{k} 位→{v}/{br['n']} 片可见" for k, v in br["reported_differs_by_decimals"].items()))
    print(f"   判级翻转：三分位线 {br['grade_flip']['tertile']['frac']:.1%}、"
          f"20/80 线 {br['grade_flip']['p20_p80']['frac']:.1%}")
    sb = values["selection_bias"]
    print(f"\n④ 形式选择乐观偏差 {sb['optimism_bias']:+.4f}"
          f"（选中分布 {sb['form_picked_counts']}）")
    print(f"\n资产 → {OUT / 'values.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

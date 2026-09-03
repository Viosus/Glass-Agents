"""W_w 与现行 CCP 关系章节的数值资产生成器（开发壳，不随核心包交付）。

为技术说明 V1.5 扩写的「与现行 CCP 的关系」一章提供全部实测数字（图文同源纪律：
正文数值一律来自资产 JSON，不写字面量）。**零图像依赖**：全部量从已入库的
data/derived/texture_w_doc/values.json 逐片特征（ca_unw / cpa_unw，native 口径的
未加权 GLCM 对比度与聚类突出——正是现行 CCP 的两个分量）确定性复算。

三组实验：
① 现行 CCP（批内锚）判别力：Cmax/CPmax 取全批最差（参考最差样品的批内代理），
   CCP = 0.5(√(Ca/Cmax) + (CP/CPmax)^0.25)，对人工分/系统分的并列平均秩 Spearman；
② 上界分母（未加权）判别力：同式但分母换理论上界 49 / 38416/12——把「换分母」
   与「加位置权重」两步改动拆开各自计效（后者与 form_comparison 现值衔接）；
③ 批内锚不稳定性（两实验室模拟）：B 次种子固定的随机对半划分，两半各自标定
   Cmax/CPmax（各自的"最差样品"），量三件事：锚值漂移 |A/B−1|、同一片在两套锚下
   的读数差、两套锚下全批排序的秩相关。

产出：data/derived/texture_w_ccp/values.json
用法：venv python fringe_scoring/make_texture_w_ccp_assets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "derived" / "texture_w_doc" / "values.json"
OUT = ROOT / "data" / "derived" / "texture_w_ccp"
DATE = "2026-08-05"
SEED, B_SPLITS = 20260805, 2000
CA_SUP = 49.0                    # (Ng−1)²，Ng=8
CP_SUP = float((2 * 7) ** 4) / 12.0  # (2(Ng−1))⁴/12 = 38416/12 ≈ 3201.33

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _avg_ranks(v) -> np.ndarray:
    """并列取平均秩（与 make_texture_w_assets._avg_ranks 同式；同仓复制避免脚本互耦）。"""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    pos = np.empty(v.size, dtype=float)
    pos[order] = np.arange(v.size, dtype=float)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=pos) / cnt)[inv]


def _spearman(x, y) -> float:
    """Spearman 秩相关（并列平均秩口径，与技术说明全篇一致）。"""
    return float(np.corrcoef(_avg_ranks(x), _avg_ranks(y))[0, 1])


def _ccp(ca: np.ndarray, cpa: np.ndarray, cmax: float, cpmax: float) -> np.ndarray:
    """现行 CCP 公式：0.5(√(Ca/Cmax) + (CP/CPmax)^0.25)，逐算子同草案同序。"""
    return 0.5 * (np.sqrt(ca / cmax) + (cpa / cpmax) ** 0.25)


def main() -> int:
    """读 115 片逐片特征 → 三组实验 → 落盘 values.json。"""
    L = json.loads(SRC.read_text(encoding="utf-8"))
    rows = L["all"]
    assert L["n_degenerate"] == 0 and not L["failures"], "源资产含退化/失败片，须先人工核定口径"
    ca = np.array([r["ca_unw"] for r in rows], dtype=float)
    cpa = np.array([r["cpa_unw"] for r in rows], dtype=float)
    man = np.array([r["man"] for r in rows], dtype=float)
    sysv = np.array([r["sys"] for r in rows], dtype=float)
    n = len(rows)
    assert (ca > 0).all() and (cpa > 0).all(), "存在零特征片，批内锚/开方口径须复核"

    # ① 现行 CCP（批内锚 = 全批最差样品代理）
    cmax_full, cpmax_full = float(ca.max()), float(cpa.max())
    ccp_full = _ccp(ca, cpa, cmax_full, cpmax_full)
    ccp_batch = {
        "cmax": round(cmax_full, 4), "cpmax": round(cpmax_full, 1),
        "anchor_sheet_cmax": int(rows[int(ca.argmax())]["no"]),
        "anchor_sheet_cpmax": int(rows[int(cpa.argmax())]["no"]),
        "vs_man": round(_spearman(ccp_full, man), 4),
        "vs_sys": round(_spearman(ccp_full, sysv), 4),
        "w_min": round(float(ccp_full.min()), 4), "w_max": round(float(ccp_full.max()), 4),
    }

    # ② 上界分母（未加权）——「换分母」单独一步的判别力
    dbu = _ccp(ca, cpa, CA_SUP, CP_SUP)
    dual_bound_unweighted = {
        "vs_man": round(_spearman(dbu, man), 4),
        "vs_sys": round(_spearman(dbu, sysv), 4),
        "note": "分母换理论上界、仍无位置权重——与①之差=换分母的排序效应；"
                "与 form_comparison.dual_bound_combined 之差=位置加权的贡献",
    }

    # ③ 批内锚不稳定性：随机对半 = 两个实验室各自标定（种子固定可复现）
    rng = np.random.default_rng(SEED)
    d_cmax, d_cpmax, reading_shift, rank_rho = [], [], [], []
    half = n // 2
    for _ in range(B_SPLITS):
        perm = rng.permutation(n)
        ia, ib = perm[:half], perm[half:]
        ca_a, cp_a = float(ca[ia].max()), float(cpa[ia].max())
        ca_b, cp_b = float(ca[ib].max()), float(cpa[ib].max())
        d_cmax.append(abs(ca_a / ca_b - 1.0))
        d_cpmax.append(abs(cp_a / cp_b - 1.0))
        w_a = _ccp(ca, cpa, ca_a, cp_a)   # 同一批片、两套锚各自全批读数
        w_b = _ccp(ca, cpa, ca_b, cp_b)
        reading_shift.append(float(np.median(np.abs(w_a - w_b))))
        rank_rho.append(_spearman(w_a, w_b))
    d_cmax, d_cpmax = np.array(d_cmax), np.array(d_cpmax)
    reading_shift, rank_rho = np.array(reading_shift), np.array(rank_rho)
    anchor_instability = {
        "B": B_SPLITS, "seed": SEED, "split": f"{half}/{n - half}",
        "cmax_drift_median": round(float(np.median(d_cmax)), 4),
        "cmax_drift_p90": round(float(np.quantile(d_cmax, 0.9)), 4),
        "cpmax_drift_median": round(float(np.median(d_cpmax)), 4),
        "cpmax_drift_p90": round(float(np.quantile(d_cpmax, 0.9)), 4),
        "reading_shift_median": round(float(np.median(reading_shift)), 4),
        "reading_shift_p90": round(float(np.quantile(reading_shift, 0.9)), 4),
        "ccp_median_full_anchor": round(float(np.median(ccp_full)), 4),
        "rank_rho_median": round(float(np.median(rank_rho)), 4),
        "rank_rho_min": round(float(rank_rho.min()), 4),
        "note": "对半划分模拟两实验室各按本方样片标定 Cmax/CPmax；"
                "reading_shift = 同一片在两套锚下读数差的批中位（CCP 值域 (0,1] 内的绝对差）",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    values = {
        "meta": {"date": DATE, "script": "fringe_scoring/make_texture_w_ccp_assets.py",
                 "source": "data/derived/texture_w_doc/values.json（all[].ca_unw/cpa_unw，"
                           "native 1px/mm，n=115）",
                 "rank_method": "average(ties)",
                 "ca_sup": CA_SUP, "cp_sup": round(CP_SUP, 2)},
        "n": n,
        "ccp_batch_anchor": ccp_batch,
        "dual_bound_unweighted": dual_bound_unweighted,
        "anchor_instability": anchor_instability,
    }
    (OUT / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"n={n}  批内锚 Cmax={ccp_batch['cmax']}（{ccp_batch['anchor_sheet_cmax']}号）"
          f" CPmax={ccp_batch['cpmax']}（{ccp_batch['anchor_sheet_cpmax']}号）")
    print(f"① CCP(批内锚)  ρ_man={ccp_batch['vs_man']:+.4f} ρ_sys={ccp_batch['vs_sys']:+.4f}"
          f"  读数域 [{ccp_batch['w_min']}, {ccp_batch['w_max']}]")
    print(f"② 上界分母(未加权) ρ_man={dual_bound_unweighted['vs_man']:+.4f}"
          f" ρ_sys={dual_bound_unweighted['vs_sys']:+.4f}")
    ai = anchor_instability
    print(f"③ 对半锚漂移：Cmax 中位 {ai['cmax_drift_median']:.1%}（p90 {ai['cmax_drift_p90']:.1%}）"
          f"  CPmax 中位 {ai['cpmax_drift_median']:.1%}（p90 {ai['cpmax_drift_p90']:.1%}）")
    print(f"   同片读数差中位 {ai['reading_shift_median']}（全批 CCP 中位 "
          f"{ai['ccp_median_full_anchor']}）  跨锚秩相关中位 {ai['rank_rho_median']}"
          f"（最小 {ai['rank_rho_min']}）")
    print(f"资产 → {OUT / 'values.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

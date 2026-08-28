"""26 片批六指标报告 v4：四项子分新刻度（产品已落地版）对照报告（开发壳）。

背景（2026-08-05 任务 1 落地）：四项深浅类子分刻度按已批规则重定标并已同步进
GlassApp 产品三端——
- 机制：gray_variance/gradient_variance 对数域（产线 p99/p5≈59/79 倍）；
  x095（物理上界 255）与 gradient_mean（p99/p5≈6.9，判据不足一个数量级）保持线性；
- refs：worst = max(产线 480 片 p99 上界化整, 26 片批 max 上界化整)，best 沿用——
  x095 20/255、gray_variance 50/6000、gradient_mean 5/60、gradient_variance 30/10000。
本报告在 26 片批上出新旧对照证据：四项 0 分数对照、总分 Spearman、名次变动。
与 v2/v3（方差两项 8000/12000 试算版）的关系：v4 为产品落地终版刻度，取代前两版数值。

数据源：values.json（原始值与旧子分）+ subscore_field/values.json（定标依据引用）。
产出：data/derived/sample26_thickness/明之言检测结果_六指标报告_v4新刻度落地.pdf
用法：venv python fringe_scoring/make_sample26_rescale_v4_pdf.py [--preview 目录]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import make_sample26_batch_pdf as v1  # 同目录版式库：页面/表格/调色板复用

OUT_PDF = v1.DERIVED / "明之言检测结果_六指标报告_v4新刻度落地.pdf"
FIELD = v1.ROOT / "data" / "derived" / "subscore_field" / "values.json"

# 产品落地终值（与 GlassApp app_config.yaml / 两份移动端快照同步；机制按键分域）
NEW_REFS = {"x095": (20.0, 255.0, "lin"), "gray_variance": (50.0, 6000.0, "log"),
            "gradient_mean": (5.0, 60.0, "lin"), "gradient_variance": (30.0, 10000.0, "log")}
FOUR = tuple(NEW_REFS)

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _new_sub(key: str, value: float) -> float:
    """终版刻度子分（与 GlassApp indicators._sub_score 同式：线性/对数按键分域）。"""
    best, worst, mech = NEW_REFS[key]
    if mech == "log":
        if value <= 0.0:
            return 100.0
        s = 100.0 * (math.log(worst) - math.log(value)) / (math.log(worst) - math.log(best))
    else:
        s = 100.0 * (worst - value) / (worst - best)
    return float(np.clip(s, 0.0, 100.0))


def _rescale(values: dict) -> list[dict]:
    """values.json → 逐张新旧对照记录（新四子分/新总分/新旧名次）。"""
    recs = []
    for r in values["photos"]:
        ind = r["indicators"]
        if ind is None:
            continue
        old_subs = {k: float(ind["sub_scores"][k]) for k in ind["sub_scores"]}
        assert abs(sum(old_subs.values()) / 6.0 - ind["total_score"]) <= 0.06, \
            f"{r['file']}: 旧子分均值与总分不符，平权前提被破坏"
        new_subs = dict(old_subs)
        for k in FOUR:
            new_subs[k] = _new_sub(k, float(ind[k]))
        recs.append({"file": r["file"], "sample_id": r["sample_id"], "type": r["type"],
                     "category": r["category"],
                     "raw": {k: float(ind[k]) for k in FOUR},
                     "old_subs": old_subs, "new_subs": new_subs,
                     "old_total": float(ind["total_score"]),
                     "new_total": float(sum(new_subs.values()) / 6.0)})
    for key, rank_key in (("old_total", "old_rank"), ("new_total", "new_rank")):
        for r in recs:
            r[rank_key] = 1 + sum(1 for o in recs if o[key] > r[key])
    return recs


def _avg_ranks(v) -> np.ndarray:
    """并列取平均秩（同 make_sample26_assets._avg_ranks，同仓复制避免脚本互耦）。"""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    pos = np.empty(v.size, dtype=float)
    pos[order] = np.arange(v.size, dtype=float)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=pos) / cnt)[inv]


def page_cover(values, recs, field, sp) -> "v1.plt.Figure":
    """封面：机制与终值 / 定标依据 / 0 分数对照 / 影响统计。"""
    zeros_old = {k: sum(1 for r in recs if r["old_subs"][k] == 0.0) for k in FOUR}
    zeros_new = {k: sum(1 for r in recs if r["new_subs"][k] == 0.0) for k in FOUR}
    # 只列并列最大 |Δ名次| 的片（避免在同 |Δ| 并列组里任取造成误导）
    max_d = max(abs(r["old_rank"] - r["new_rank"]) for r in recs)
    moved = [r for r in recs if abs(r["old_rank"] - r["new_rank"]) == max_d]
    q = field["quantiles"]

    fig = v1._new_page(v1.PORTRAIT)
    fig.text(0.08, 0.925, "明之言检测结果 · 六指标报告 v4", fontsize=19,
             color=v1.INK, fontweight="bold")
    fig.text(0.08, 0.900, "四项深浅类子分新刻度（产品已落地）· 26 片批新旧对照",
             fontsize=11.5, color=v1.INK2)
    fig.lines.append(v1.plt.Line2D([0.08, 0.92], [0.885, 0.885],
                                   transform=fig.transFigure, color=v1.RULE, lw=1.0))

    y = v1._text_block(fig, 0.08, 0.856, ["【新刻度（GlassApp 三端已同步）】"], size=12,
                       weight="bold")
    y = v1._text_block(fig, 0.08, y - 0.004, [
        "映射机制按键分域：灰度方差/梯度方差 = 对数域 s=100×(ln worst−ln v)/(ln worst−ln best)；",
        "x095 与梯度均值保持线性。refs 终值（best 沿用 / worst = max(产线 p99 上整, 26 片 max 上整)）：",
        "　x095 20/255（物理上界）｜灰度方差 50/6000（对数）｜梯度均值 5/60｜梯度方差 30/10000（对数）",
    ])
    y = v1._text_block(fig, 0.08, y - 0.016, ["【定标依据（产线 480 片普查，subscore_field 资产）】"],
                       size=12, weight="bold")
    y = v1._text_block(fig, 0.08, y - 0.004, [
        f"x095 p99={q['x095']['p99']:g}（饱和顶 255）；灰度方差 p99={q['gray_variance']['p99']:g}"
        f"（26 片 max 5615）；",
        f"梯度均值 p99={q['gradient_mean']['p99']:g}（26 片 max 56.8）；梯度方差 "
        f"p99={q['gradient_variance']['p99']:g}（26 片 max 9810）。",
        f"机制判据：p99/p5 = 灰方差 {field['p99_over_p5']['gray_variance']:g}× / "
        f"梯度方差 {field['p99_over_p5']['gradient_variance']:g}×（对数）；"
        f"梯度均值 {field['p99_over_p5']['gradient_mean']:g}× / "
        f"x095 {field['p99_over_p5']['x095']:g}×（不足一个数量级，线性）。",
    ], size=9.2, dy=0.0158)

    y = v1._text_block(fig, 0.08, y - 0.016, ["【26 片批影响】"], size=12, weight="bold")
    zline = "；".join(f"{k} {zeros_old[k]}→{zeros_new[k]}" for k in FOUR)
    y = v1._text_block(fig, 0.08, y - 0.004, [
        f"0 分片数：{zline}（张）",
        f"Spearman(旧总分, 新总分) = {sp:+.4f}（并列平均秩，n={len(recs)}）",
        f"名次变动最大（|Δ|={max(abs(r['old_rank'] - r['new_rank']) for r in recs)}）："
        + "；".join(f"{r['sample_id']}（{r['old_rank']}→{r['new_rank']}）" for r in moved)
        + "；其余 |Δ|≤1",
    ], size=9.2, dy=0.0158)

    v1._text_block(fig, 0.08, y - 0.016, [
        "【阅读注意】v2/v3 报告的方差两项 8000/12000 为试算版刻度，本版（6000/10000+四项）",
        "为产品落地终版，取代前两版数值。其余口径注意（灰度域、跨厚度不公平、不出判定、",
        "8# 零点语义等）沿用 v1 封面。三端子分对拍 38 例 1e-9 内一致（Python/Dart/JS）。",
    ], size=9.2, dy=0.0158)
    fig.text(0.08, 0.05, "生成脚本：fringe_scoring/make_sample26_rescale_v4_pdf.py"
             "（数据源 values.json + subscore_field/values.json，逐值同源）",
             fontsize=8, color=v1.INK2)
    return fig


def page_compare(recs) -> "v1.plt.Figure":
    """四项新旧子分与总分对照表（横向）。"""
    def raw(key, fmt):
        """原始值列取值器：按 fmt 格式化指标 key。"""
        return lambda r: fmt.format(r["raw"][key])

    cols = [
        ("文件", 0.025, "left", lambda r: r["file"]),
        ("样品", 0.072, "left", lambda r: r["sample_id"]),
        ("x095", 0.135, "right", raw("x095", "{:.1f}")),
        ("旧|新", 0.205, "right", lambda r: f"{r['old_subs']['x095']:.0f}|{r['new_subs']['x095']:.0f}"),
        ("灰方差", 0.285, "right", raw("gray_variance", "{:.0f}")),
        ("旧|新", 0.355, "right", lambda r: f"{r['old_subs']['gray_variance']:.0f}|{r['new_subs']['gray_variance']:.0f}"),
        ("梯度均", 0.425, "right", raw("gradient_mean", "{:.2f}")),
        ("旧|新", 0.495, "right", lambda r: f"{r['old_subs']['gradient_mean']:.0f}|{r['new_subs']['gradient_mean']:.0f}"),
        ("梯方差", 0.570, "right", raw("gradient_variance", "{:.0f}")),
        ("旧|新", 0.640, "right", lambda r: f"{r['old_subs']['gradient_variance']:.0f}|{r['new_subs']['gradient_variance']:.0f}"),
        ("旧总分", 0.720, "right", lambda r: f"{r['old_total']:.2f}"),
        ("新总分", 0.800, "right", lambda r: f"{r['new_total']:.2f}"),
        ("旧→新名次", 0.900, "right",
         lambda r: f"{r['old_rank']}→{r['new_rank']}"),
        ("Δ", 0.955, "right",
         lambda r: f"{r['old_rank'] - r['new_rank']:+d}" if r["old_rank"] != r["new_rank"] else "0"),
    ]
    foot = ("「旧|新」= 该项子分旧刻度|新刻度（整数显示）；总分 = 六项平权（texture_w/位置评分子分不变）。"
            "Δ名次 正=上升；名次按 28 张照片池计。")
    return v1._table_page("四项子分新旧对照（终版刻度）", cols, recs, foot)


def page_chart(recs) -> "v1.plt.Figure":
    """编号序新总分图（沿用 v3 版式口径）。"""
    fig = v1._new_page(v1.PORTRAIT)
    fig.text(0.08, 0.945, "新刻度六指标总分（0–100）· 按图片编号排序",
             fontsize=13, color=v1.INK, fontweight="bold")
    ax = fig.add_axes([0.15, 0.075, 0.77, 0.83])
    for i, r in enumerate(recs):
        c = r["category"]
        ax.barh(i, r["new_total"], height=0.62, color=v1.CAT_COLOR[c],
                hatch="///" if c == "control" else None,
                edgecolor="white" if c == "control" else "none", linewidth=0.0)
    ax.set_yticks(np.arange(len(recs)),
                  labels=[Path(r["file"]).stem for r in recs], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("总分（新刻度）", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[
        v1.Patch(facecolor=v1.CAT_COLOR["core"], label="普白钢化"),
        v1.Patch(facecolor=v1.CAT_COLOR["special"], label="特殊片"),
        v1.Patch(facecolor=v1.CAT_COLOR["control"], hatch="///", edgecolor="white",
                 label="未钢化对照（零点语义）"),
    ], fontsize=8, framealpha=0.9, loc="lower right")
    fig.text(0.15, 0.028, "跨厚度/跨类别比较不公平（v1 注意）。", fontsize=8, color=v1.INK2)
    return fig


def main() -> int:
    """读两份资产 → 重标定 → 三页 v4 PDF（+可选预览）。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", type=Path, default=None,
                    help="同时把每页另存为 PNG 的目录（人工核版式用）")
    args = ap.parse_args()

    values = json.loads((v1.DERIVED / "values.json").read_text(encoding="utf-8"))
    field = json.loads(FIELD.read_text(encoding="utf-8"))
    recs = _rescale(values)
    sp = float(np.corrcoef(_avg_ranks([r["old_total"] for r in recs]),
                           _avg_ranks([r["new_total"] for r in recs]))[0, 1])

    preview = args.preview
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
    idx = 1
    with v1.PdfPages(OUT_PDF) as pdf:
        idx = v1._save_page(pdf, page_cover(values, recs, field, sp), preview, idx)
        idx = v1._save_page(pdf, page_compare(recs), preview, idx)
        idx = v1._save_page(pdf, page_chart(recs), preview, idx)
        info = pdf.infodict()
        info["Title"] = "明之言检测结果 · 六指标报告 v4（新刻度落地）"
        info["Subject"] = "四项深浅类子分终版刻度（对数×2+线性×2）26 片批新旧对照"

    zeros_new = {k: sum(1 for r in recs if r["new_subs"][k] == 0.0) for k in FOUR}
    print(f"报告 → {OUT_PDF}（{idx - 1} 页）")
    print(f"Spearman(旧,新)={sp:+.4f}  新刻度 0 分片数：{zeros_new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

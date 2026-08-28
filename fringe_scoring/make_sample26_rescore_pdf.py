"""26 片批六指标报告 v2：方差两项子分对数重标定 + GlassApp 纯自动检测普查（开发壳）。

背景（2026-08-05 用户两点反馈）：
① 灰度方差/梯度方差的线性子分上下限太窄（worst=4000/1100），批内 1/16 张片被
   压成 0 分（并列 0 抹掉真实排序）——本脚本在**分析层**改用对数域映射重算，
   GlassApp 产品代码与 config **零改动**（要落产品须三端同步，另行决策）；
② GlassApp 纯自动测不出 16#——普查结论与机理页并入本报告（数据源 auto_census.json）。

评分机制（只动灰度方差、梯度方差两项，其余四项沿用 v1 子分）：
- 旧：s = 100·(worst−v)/(worst−best)，clip[0,100]，worst=4000/1100；
- 新：s = 100·(ln worst−ln v)/(ln worst−ln best)，clip[0,100]，best 沿用（50/30），
  worst 放宽为批内最大值的上界化整（8000/12000，批内初标，跨批使用须工厂复标）。
  动机：方差为尺度平方量，批内跨约 700×/250× 且右偏——线性刻度在大值区全部
  饱和为 0；对数刻度等价于按倍数等距扣分，单调变换不改单指标内部排序，
  消除并列 0、恢复两项在加权总分中的判别力。

数据源（本脚本零打分、零手填）：values.json（v1 原始值/子分/总分）+
auto_census/auto_census.json（纯自动普查）。产出：
data/derived/sample26_thickness/明之言检测结果_六指标批量报告_v2重标定.pdf
用法：venv python fringe_scoring/make_sample26_rescore_pdf.py [--preview 目录]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import make_sample26_batch_pdf as v1  # 同目录版式库：页面/表格/调色板复用

DERIVED = v1.DERIVED
CENSUS_DIR = DERIVED / "auto_census"
OUT_PDF = DERIVED / "明之言检测结果_六指标批量报告_v2重标定.pdf"

# 对数域重标定锚点：best 沿用 config 默认 refs；worst=批内最大值上界化整（初标）
LOG_REFS = {"gray_variance": {"best": 50.0, "worst": 8000.0},
            "gradient_variance": {"best": 30.0, "worst": 12000.0}}
RESCORED = tuple(LOG_REFS)

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _log_sub(value: float, best: float, worst: float) -> float:
    """对数域子分：s = 100·(ln worst − ln v)/(ln worst − ln best)，clip [0,100]。"""
    if value <= 0.0:
        return 100.0  # 方差非负；0 只在常量图出现，优于任何 best → 顶格
    s = 100.0 * (math.log(worst) - math.log(value)) / (math.log(worst) - math.log(best))
    return float(np.clip(s, 0.0, 100.0))


def _rescore(values: dict) -> list[dict]:
    """values.json → 逐张新旧对照记录（新子分/新总分/新旧名次）。

    重算前逐张断言：六项旧子分平权均值 ≈ 落盘总分（容差 0.06，1dp 舍入引起）——
    校验「默认方案=平权」前提；被破坏即崩，绝不静默给错总分。
    """
    recs = []
    for r in values["photos"]:
        ind = r["indicators"]
        if ind is None:
            continue
        old_subs = {k: float(ind["sub_scores"][k]) for k in ind["sub_scores"]}
        assert abs(sum(old_subs.values()) / 6.0 - ind["total_score"]) <= 0.06, \
            f"{r['file']}: 旧子分均值与总分不符，平权前提被破坏"
        new_subs = dict(old_subs)
        for k in RESCORED:
            raw = float(ind[k])
            assert raw < LOG_REFS[k]["worst"], \
                f"{r['file']}: {k}={raw} 超出重标定 worst，锚点须重定"
            new_subs[k] = _log_sub(raw, LOG_REFS[k]["best"], LOG_REFS[k]["worst"])
        recs.append({
            "file": r["file"], "sample_id": r["sample_id"], "type": r["type"],
            "category": r["category"],
            "raw": {k: float(ind[k]) for k in RESCORED},
            "old_subs": old_subs, "new_subs": new_subs,
            "old_total": float(ind["total_score"]),
            "new_total": float(sum(new_subs.values()) / 6.0),
        })
    for key, rank_key in (("old_total", "old_rank"), ("new_total", "new_rank")):
        for r in recs:
            r[rank_key] = 1 + sum(1 for o in recs if o[key] > r[key])
    return recs


def _census_summary(census: dict) -> dict:
    """普查 JSON → 逐张短语 + 汇总（供状态表与封面）。"""
    def short(r: dict) -> tuple[str, str]:
        """一张普查记录 → (检出短语, 结论短语)：可测 / 错检（N片） / 拒测原因。"""
        if r["auto_ok"]:
            return "1 片", "可测"
        if r["outcome"] == "detected":
            return f"{r['n_sheets']} 片", f"错检（{r['n_sheets']}片）"
        e = r["error"]
        if "未检出任何玻璃片" in e:
            return "报错", "拒测：前景为空"
        if "无法拟合为四边形" in e:
            pts = "3 点" if "3 点" in e else ("2 点" if "2 点" in e else "<4 点")
            return "报错", f"拒测：拟合退化（{pts}）"
        return "报错", "拒测：" + e[:16]
    rows = {}
    for r in census["photos"]:
        n, concl = short(r)
        rows[r["file"]] = {"n": n, "conclusion": concl, "auto_ok": r["auto_ok"],
                           "had_manual": r["had_manual_quad"]}
    return rows


def page_cover(values, census, recs, sp_old_new) -> "v1.plt.Figure":
    """v2 封面：变更机制/影响统计/普查摘要/阅读注意。"""
    meta = values["meta"]
    zeros_before = {k: sum(1 for r in recs if r["old_subs"][k] == 0.0) for k in RESCORED}
    zeros_after = {k: sum(1 for r in recs if r["new_subs"][k] == 0.0) for k in RESCORED}
    moved = sorted(recs, key=lambda r: -abs(r["old_rank"] - r["new_rank"]))[:3]

    fig = v1._new_page(v1.PORTRAIT)
    fig.text(0.08, 0.925, "明之言检测结果 · 六指标批量测量报告 v2", fontsize=19,
             color=v1.INK, fontweight="bold")
    fig.text(0.08, 0.900, "方差两项子分对数重标定 + GlassApp 纯自动检测普查（26 片批 28 张）",
             fontsize=11.5, color=v1.INK2)
    fig.lines.append(v1.plt.Line2D([0.08, 0.92], [0.885, 0.885],
                                   transform=fig.transFigure, color=v1.RULE, lw=1.0))

    y = v1._text_block(fig, 0.08, 0.856, ["【与 v1 的关系】"], size=12, weight="bold")
    y = v1._text_block(fig, 0.08, y - 0.004, [
        f"测量原始值与 v1 完全同源（{meta['date']} 打分，config SHA256/16 = "
        f"{meta['config_sha256_16']}），",
        "图像未重测，只重算评分层：灰度方差 / 梯度方差两项子分由线性映射改为对数域映射",
        "（其余四项子分与位置评分不变）。本重算在分析层完成，GlassApp 产品代码与 config",
        "零改动（落产品须三端同步，见注意 4）。",
    ])

    y = v1._text_block(fig, 0.08, y - 0.016, ["【新评分机制（仅方差两项）】"], size=12, weight="bold")
    y = v1._text_block(fig, 0.08, y - 0.004, [
        "旧：s = 100×(worst−v)/(worst−best)，clip[0,100]；worst = 4000（灰方差）/ 1100（梯方差）",
        "新：s = 100×(ln worst−ln v)/(ln worst−ln best)，clip[0,100]；best 沿用 50 / 30，",
        "      worst 放宽至 8000 / 12000（批内最大值 5615 / 9810 的上界化整，批内初标）",
        "动机：方差是尺度平方量，批内跨约 700× / 250× 且右偏；线性刻度大值区全部饱和为 0，",
        "对数刻度=按倍数等距扣分。单调变换不改单指标内部排序，只消除并列 0、恢复判别力。",
    ], size=9.2, dy=0.0158)

    y = v1._text_block(fig, 0.08, y - 0.016, ["【重算影响】"], size=12, weight="bold")
    y = v1._text_block(fig, 0.08, y - 0.004, [
        f"0 分片数：灰度方差 {zeros_before['gray_variance']}→{zeros_after['gray_variance']} 张，"
        f"梯度方差 {zeros_before['gradient_variance']}→{zeros_after['gradient_variance']} 张"
        f"（重算后无并列 0）    Spearman(旧总分, 新总分) = {sp_old_new:+.4f}",
        "名次变动最大：" + "；".join(
            f"{r['sample_id']}（{r['old_rank']}→{r['new_rank']}）" for r in moved),
    ], size=9.2, dy=0.0158)

    y = v1._text_block(fig, 0.08, y - 0.016, ["【GlassApp 纯自动检测普查（详见后两页）】"],
                       size=12, weight="bold")
    y = v1._text_block(fig, 0.08, y - 0.004, [
        f"桌面端同款纯自动路径：可测 {census['n_auto_ok']}/{census['n_photos']} 张；"
        f"拒测报错 {census['n_error']} 张（8#-1/8#-2 未钢化前景为空、",
        f"16-1/16-2 拟合退化）；错检片数 {census['n_miscount']} 张（1-2 检成 3 片、13 检成 2 片）。",
        "16# 测不出的机理与参数实验（close_frac 0.005→0.01 可救回，IoU 0.97）见第 3 页。",
        "本报告 16#/8#/1-2/13 的指标值来自人工定角口径（v1 同源），表中以＊标注。",
    ], size=9.2, dy=0.0158)

    y = v1._text_block(fig, 0.08, y - 0.016, ["【阅读注意】"], size=12, weight="bold")
    v1._text_block(fig, 0.08, y - 0.004, [
        "1. 沿用 v1 全部口径注意（X0.95 为灰度域、跨厚度/跨类别比较不公平、不出合格判定、",
        "   8# 高分为零点语义、W_w 评估域=整片矫正图）。",
        "2. 新 worst 锚点（8000/12000）为批内初标——非好/差样人工标定，跨批使用前须工厂复标。",
        "3. 对数映射只动两项方差子分；单指标内部排序不变，总分排序变化全部来自去饱和后的间距重排。",
        "4. 若采纳进 GlassApp 产品：改 indicators 评分层属算法变更，须 Python 权威版 + Dart +",
        "   小程序 JS 三处同步，并更新两份移动端 local_config 快照——本报告不改动任何一处。",
    ], size=9.2, dy=0.0158)

    fig.text(0.08, 0.045, "生成脚本：fringe_scoring/make_sample26_rescore_pdf.py"
             "（数据源 values.json + auto_census.json，逐值同源可追溯）",
             fontsize=8, color=v1.INK2)
    return fig


def page_census_table(values, census_rows) -> "v1.plt.Figure":
    """② 纯自动检测状态表（28 行）。"""
    cols = [
        ("文件", 0.030, "left", lambda r: r["file"]),
        ("样品", 0.095, "left", lambda r: r["sample_id"]),
        ("类型", 0.150, "left", lambda r: r["type"][:18]),
        ("纯自动检出", 0.470, "right", lambda r: census_rows[r["file"]]["n"]),
        ("纯自动结论", 0.640, "right", lambda r: census_rows[r["file"]]["conclusion"]),
        ("v1 曾人工定角", 0.800, "right",
         lambda r: "是" if census_rows[r["file"]]["had_manual"] else "—"),
        ("本报告指标来源", 0.960, "right",
         lambda r: "人工定角＊" if census_rows[r["file"]]["had_manual"] else "自动"),
    ]
    foot = ("口径 = GlassApp 桌面端真实调用路径（score_sheets 纯自动检测段），ValueError 弹窗拒绝打分。"
            "「错检」= 检出片数≠1（本批每张恰一片实物）。＊值仍为 v1 人工定角口径测量。")
    return v1._table_page("GlassApp 纯自动检测普查（28 张照片）", cols, values["photos"], foot)


def page_16_mechanism() -> "v1.plt.Figure":
    """③ 16# 测不出机理页：两张四联诊断图 + 机理与参数实验结论。"""
    import matplotlib.image as mpimg

    fig = v1._new_page(v1.PORTRAIT)
    fig.text(0.06, 0.955, "16#（6Low-E+12A+6 钢化中空）为什么测不出", fontsize=13,
             color=v1.INK, fontweight="bold")
    for i, name in enumerate(("diag_16-1.png", "diag_16-2.png")):
        ax = fig.add_axes([0.04, 0.590 - i * 0.210, 0.92, 0.185])
        ax.imshow(mpimg.imread(str(CENSUS_DIR / name)))
        ax.set_axis_off()
    v1._text_block(fig, 0.07, 0.920, [
        "机理（四联图：原图 → 阈值前景 → 闭运算 → 角点对比，红=自动 / 绿虚=人工）：",
        "① Low-E 镀膜 + 中空双片把片内应力斑发光压到前景阈值以下（16-1：背景灰度 6.0、",
        "    阈值 45.0，前景占比仅 6.4%）——阈值由边缘亮圈的高亮参考项主导，内部整体落选；",
        "② 前景只剩边缘亮圈碎段，默认闭运算（3px）连不成闭环 → 面积/边长过滤后仅剩 2 条",
        "    弯折弧段（非实心片域）；",
        "③ 弧段凸包近似退化，四边形拟合收敛到 3 点（16-1）/ 2 点（16-2）< 4 → 抛 ValueError，",
        "    App 弹窗拒绝打分（worker.py 失败契约：不静默、不给猜的分）。",
    ], size=9.0, dy=0.0155)
    v1._text_block(fig, 0.07, 0.345, [
        "参数敏感性实验（16# 网格 + 全批回归，详值见 auto_census 与会话记录）：",
        "· close_frac 0.005→0.01（其余默认）：16-1/16-2 均正确检出（与人工角点 IoU 0.97），",
        "  8#（未钢化，物理无信号）保持正确拒测；1-2/13 错检维持现状；",
        "· close_frac ≥0.02 可再修复 1-2/13，但 8#-1 会被检成假片（IoU 0.02）——不可取；",
        "· fg_min_rel 降至 0.20 以下同样可救 16#，救援窗口宽（IoU 0.93–0.97 全网格稳定）。",
        "落地前提：close_frac 同时作用于整床多片切分（片间桥接风险），须整床照回归；",
        "该参数在移动端 local_config 快照内，改 config 须两移动端同步。",
        "另注意：即便几何救回，Low-E 中空的指标幅度被镀膜/双片系统性衰减（v1 总分 16-1",
        "57.6 vs 16-2 48.4 双拍差 9.2 分；本报告新口径 56.2 vs 45.8 差 10.4），只作定性",
        "参考，不与单片同池排名（v1 注意 6）。",
    ], size=9.0, dy=0.0155)
    return fig


def page_compare_table(recs) -> "v1.plt.Figure":
    """④ 新旧对照表：方差两项 raw/旧子分/新子分 + 新旧总分与名次。"""
    cols = [
        ("文件", 0.030, "left", lambda r: r["file"]),
        ("样品", 0.082, "left", lambda r: r["sample_id"]),
        ("灰方差", 0.190, "right", lambda r: f"{r['raw']['gray_variance']:.1f}"),
        ("旧分", 0.245, "right", lambda r: f"{r['old_subs']['gray_variance']:.1f}"),
        ("新分", 0.300, "right", lambda r: f"{r['new_subs']['gray_variance']:.1f}"),
        ("梯方差", 0.400, "right", lambda r: f"{r['raw']['gradient_variance']:.1f}"),
        ("旧分", 0.455, "right", lambda r: f"{r['old_subs']['gradient_variance']:.1f}"),
        ("新分", 0.510, "right", lambda r: f"{r['new_subs']['gradient_variance']:.1f}"),
        ("旧总分", 0.610, "right", lambda r: f"{r['old_total']:.2f}"),
        ("新总分", 0.685, "right", lambda r: f"{r['new_total']:.2f}"),
        ("Δ总分", 0.760, "right", lambda r: f"{r['new_total'] - r['old_total']:+.2f}"),
        ("旧名次", 0.830, "right", lambda r: str(r["old_rank"])),
        ("新名次", 0.895, "right", lambda r: str(r["new_rank"])),
        ("Δ名次", 0.960, "right",
         lambda r: f"{r['old_rank'] - r['new_rank']:+d}" if r["old_rank"] != r["new_rank"]
         else "0"),
    ]
    foot = ("新子分 = 对数域映射（封面公式；worst 放宽至 8000/12000）；其余四项子分不变；"
            "总分 = 六项平权均值。Δ名次 正=上升。名次按 28 张照片池计（含特殊片与对照，仅供机制对比）。")
    return v1._table_page("方差两项重标定：新旧子分 / 总分 / 名次对照", cols, recs, foot)


def page_new_chart(recs, census_rows) -> "v1.plt.Figure":
    """⑤ 新总分排序图（＊=纯自动测不出/错检，值为人工定角口径）。"""
    rs = sorted(recs, key=lambda r: -r["new_total"])
    fig = v1._new_page(v1.PORTRAIT)
    fig.text(0.08, 0.945, "重标定后六指标总分排序（0–100，越高越好）",
             fontsize=13, color=v1.INK, fontweight="bold")
    ax = fig.add_axes([0.15, 0.075, 0.77, 0.83])
    for i, r in enumerate(rs):
        c = r["category"]
        ax.barh(i, r["new_total"], height=0.62, color=v1.CAT_COLOR[c],
                hatch="///" if c == "control" else None,
                edgecolor="white" if c == "control" else "none", linewidth=0.0)
    labels = [Path(r["file"]).stem + ("＊" if not census_rows[r["file"]]["auto_ok"] else "")
              for r in rs]
    ax.set_yticks(np.arange(len(rs)), labels=labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("总分（方差两项对数重标定后）", fontsize=9)
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
    fig.text(0.15, 0.028, "＊= GlassApp 纯自动测不出/错检（值为人工定角口径）。"
             "跨厚度/跨类别比较不公平（v1 注意）。", fontsize=8, color=v1.INK2)
    return fig


def page_new_subs_table(recs, census_rows) -> "v1.plt.Figure":
    """⑥ 新口径六子分与总分总表。"""
    def sub(key):
        """子分列取值器（新口径）。"""
        return lambda r: f"{r['new_subs'][key]:.1f}"

    cols = [
        ("文件", 0.030, "left", lambda r: r["file"]),
        ("样品", 0.085, "left", lambda r: r["sample_id"]),
        ("类型", 0.135, "left", lambda r: r["type"][:18]),
        ("X0.95", 0.420, "right", sub("x095")),
        ("灰度方差†", 0.500, "right", sub("gray_variance")),
        ("梯度均值", 0.575, "right", sub("gradient_mean")),
        ("梯度方差†", 0.655, "right", sub("gradient_variance")),
        ("W_w", 0.715, "right", sub("texture_w")),
        ("位置评分", 0.785, "right", sub("position_score")),
        ("新总分", 0.850, "right", lambda r: f"{r['new_total']:.2f}"),
        ("名次", 0.900, "right", lambda r: str(r["new_rank"])),
        ("纯自动", 0.960, "right",
         lambda r: "可测" if census_rows[r["file"]]["auto_ok"] else "＊"),
    ]
    foot = ("† = 本次对数重标定的两项；其余子分与 v1 相同。总分 = 六项平权均值。"
            "＊= 纯自动测不出/错检（值为人工定角口径）。")
    return v1._table_page("重标定后六子分与总分（0–100）", cols, recs, foot)


def main() -> int:
    """读 values.json + auto_census.json → 重算 → 六页 v2 PDF（+可选 PNG 预览）。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", type=Path, default=None,
                    help="同时把每页另存为 PNG 的目录（人工核版式用）")
    args = ap.parse_args()

    values = json.loads((DERIVED / "values.json").read_text(encoding="utf-8"))
    census = json.loads((CENSUS_DIR / "auto_census.json").read_text(encoding="utf-8"))
    recs = _rescore(values)
    census_rows = _census_summary(census)
    assert len(recs) == values["n_scored"] and len(census_rows) == values["n_photos"]

    # 新旧总分秩相关（并列平均秩；照片池 n=28）
    def avg_ranks(v):
        """并列取平均秩（同 make_sample26_assets._avg_ranks，同仓复制避免脚本互耦）。"""
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="mergesort")
        pos = np.empty(v.size, dtype=float)
        pos[order] = np.arange(v.size, dtype=float)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        return (np.bincount(inv, weights=pos) / cnt)[inv]

    sp = float(np.corrcoef(avg_ranks([r["old_total"] for r in recs]),
                           avg_ranks([r["new_total"] for r in recs]))[0, 1])

    preview = args.preview
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
    idx = 1
    with v1.PdfPages(OUT_PDF) as pdf:
        idx = v1._save_page(pdf, page_cover(values, census, recs, sp), preview, idx)
        idx = v1._save_page(pdf, page_census_table(values, census_rows), preview, idx)
        idx = v1._save_page(pdf, page_16_mechanism(), preview, idx)
        idx = v1._save_page(pdf, page_compare_table(recs), preview, idx)
        idx = v1._save_page(pdf, page_new_chart(recs, census_rows), preview, idx)
        idx = v1._save_page(pdf, page_new_subs_table(recs, census_rows), preview, idx)
        info = pdf.infodict()
        info["Title"] = "明之言检测结果 · 六指标批量测量报告 v2（方差重标定+检测普查）"
        info["Subject"] = "对数域重标定灰度方差/梯度方差子分；GlassApp 纯自动检测普查"

    moved = sorted(recs, key=lambda r: -abs(r["old_rank"] - r["new_rank"]))[:5]
    print(f"报告 → {OUT_PDF}（{idx - 1} 页）")
    print(f"Spearman(旧总分, 新总分) = {sp:+.4f}（n={len(recs)}）")
    print("名次变动 Top5：" + "；".join(
        f"{r['sample_id']} {r['old_rank']}→{r['new_rank']}" for r in moved))
    return 0


if __name__ == "__main__":
    sys.exit(main())

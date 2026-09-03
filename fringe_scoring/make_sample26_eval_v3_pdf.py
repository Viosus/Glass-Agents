"""26 片批六指标评测报告 v3：新机制（方差两项对数域）正式版，按图片编号顺序（开发壳）。

定位（2026-08-05 用户要求）：v2 是机制对照/诊断报告，本版是**新口径的正式评测**——
排版全部按图片编号顺序（manifest 序：1-1, 1-2, 2, …, 28），含逐片详情缩略图页。
评分 = v2 同一重算层（make_sample26_rescore_pdf._rescore，392 格已对拍验证）：
灰度方差/梯度方差子分对数域映射（best 50/30，worst 8000/12000 批内初标），
其余四项子分沿用 v1；总分 = 六项平权均值。

数据源（零打分、零手填）：values.json + auto_census/auto_census.json。
产出：data/derived/sample26_thickness/明之言检测结果_六指标评测报告_v3新机制.pdf
用法：venv python fringe_scoring/make_sample26_eval_v3_pdf.py [--preview 目录]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

import make_sample26_batch_pdf as v1     # 同目录版式库：页面/表格/调色板/缩略图
import make_sample26_rescore_pdf as v2   # 同目录重算层：_rescore/_census_summary/锚点

OUT_PDF = v1.DERIVED / "明之言检测结果_六指标评测报告_v3新机制.pdf"

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def page_cover(values: dict, census: dict, recs: list[dict]) -> "v1.plt.Figure":
    """封面：新机制口径 + 批次概览 + 阅读注意（详细机制推导见 v2 报告）。"""
    meta = values["meta"]
    totals = [r["new_total"] for r in recs]
    n_cat = {c: sum(1 for r in recs if r["category"] == c) for c in v1.CAT_ZH}

    fig = v1._new_page(v1.PORTRAIT)
    fig.text(0.08, 0.925, "明之言检测结果 · 六指标评测报告（新机制）", fontsize=19,
             color=v1.INK, fontweight="bold")
    fig.text(0.08, 0.900, "26 片厚度批 28 张照片 · 方差两项对数域子分 · 按图片编号排序",
             fontsize=11.5, color=v1.INK2)
    fig.lines.append(v1.plt.Line2D([0.08, 0.92], [0.885, 0.885],
                                   transform=fig.transFigure, color=v1.RULE, lw=1.0))

    y = v1._text_block(fig, 0.08, 0.855, ["【出处与口径】"], size=12, weight="bold")
    y = v1._text_block(fig, 0.08, y - 0.004, [
        f"测量日期：{meta['date']}    数据源：data/images/明之言检测结果（28 张 PNG）",
        "算法：GlassApp 权威 Python 实现 fringe_scoring v1.12（texture_w 版六指标）",
        f"配置：app_config.yaml 默认方案（六项平权）    config SHA256/16 = {meta['config_sha256_16']}",
        "评分机制：灰度方差/梯度方差子分 = 对数域映射 s=100×(ln worst−ln v)/(ln worst−ln best)，",
        "      best 50 / 30，worst 8000 / 12000（批内初标）；其余四项子分为 v1 线性口径。",
        f"评估域：{meta['评估域']}",
        f"尺度口径：{meta['口径']}",
    ])

    y = v1._text_block(fig, 0.08, y - 0.018, ["【批次概览】"], size=12, weight="bold")
    y = v1._text_block(fig, 0.08, y - 0.004, [
        f"照片 28 张全部出值。总分范围 {min(totals):.2f} ~ {max(totals):.2f}，"
        f"中位 {float(np.median(totals)):.2f}。",
        f"类别：普白钢化 {n_cat['core']} 张 · 特殊片 {n_cat['special']} 张 · 未钢化对照 "
        f"{n_cat['control']} 张。",
        f"GlassApp 纯自动可测 {census['n_auto_ok']}/28：8#-1/8#-2（未钢化）与 16-1/16-2"
        "（Low-E 中空）拒测、",
        "1-2/13 错检——这六张的值来自人工定角口径，全文以＊标注（机理见 v2 报告第 3 页）。",
    ])

    y = v1._text_block(fig, 0.08, y - 0.018, ["【阅读注意】"], size=12, weight="bold")
    v1._text_block(fig, 0.08, y - 0.004, [
        "1. X0.95 为灰度域 95% 分位（gray→nm 未标定），只在本批内比较。",
        "2. 总分口径 = 六项平权 + 全厚度默认参考值；方差两项 worst 锚点为批内初标，跨批使用前",
        "   须工厂复标。厚玻璃应力斑天生更重，跨厚度直接比总分不公平，同厚度内比较才有意义。",
        "3. 判定线 fail_line 未启用——只报分，不出「合格/不合格」判定。",
        "4. 8#-1/8#-2 为未钢化零点对照：高分是「无应力斑信号」的零点语义，不代表质量好。",
        "5. 特殊片（中空/夹层/压花/Low-E/均质/高应力防火）光学条件各异，分数只作定性参考，",
        "   不与普白钢化同池解读；Low-E 中空（16#）幅度被镀膜系统性衰减，尤须谨慎。",
        "6. 评估域：W_w 取整片矫正图；其余五项取扣免罚边框带内部。位置评分为双支路取小。",
        "7. 新旧机制逐片对照与名次变化见 v2 报告（明之言检测结果_六指标批量报告_v2重标定.pdf）。",
    ], size=9.2, dy=0.0158)

    fig.text(0.08, 0.05, "生成脚本：fringe_scoring/make_sample26_eval_v3_pdf.py"
             "（数据源 values.json + auto_census.json，逐值同源可追溯）",
             fontsize=8, color=v1.INK2)
    return fig


def page_chart(recs: list[dict], census_rows: dict) -> "v1.plt.Figure":
    """总分横条图：**按图片编号顺序**（不按分数排序），类别着色+对照纹理。"""
    fig = v1._new_page(v1.PORTRAIT)
    fig.text(0.08, 0.945, "六指标总分（新机制，0–100，越高越好）· 按图片编号排序",
             fontsize=13, color=v1.INK, fontweight="bold")
    ax = fig.add_axes([0.15, 0.075, 0.77, 0.83])
    for i, r in enumerate(recs):
        c = r["category"]
        ax.barh(i, r["new_total"], height=0.62, color=v1.CAT_COLOR[c],
                hatch="///" if c == "control" else None,
                edgecolor="white" if c == "control" else "none", linewidth=0.0)
    labels = [Path(r["file"]).stem + ("＊" if not census_rows[r["file"]]["auto_ok"] else "")
              for r in recs]
    ax.set_yticks(np.arange(len(recs)), labels=labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("总分（新机制）", fontsize=9)
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
             "跨厚度/跨类别比较不公平（封面注意 2/5）。", fontsize=8, color=v1.INK2)
    return fig


def pages_details(values: dict, recs: list[dict], census_rows: dict,
                  photo_dir: Path):
    """逐片详情页（2×2：缩略图 + 原始值与新子分），按图片编号顺序逐页产出。"""
    by_file = {r["file"]: r for r in recs}
    vrecs = values["photos"]
    branch_zh = {"coverage": "覆盖度", "concentration": "集中度"}
    per_page = 4
    for p0 in range(0, len(vrecs), per_page):
        fig = v1._new_page(v1.PORTRAIT)
        fig.text(0.06, 0.955,
                 f"逐片详情（{p0 + 1}–{min(p0 + per_page, len(vrecs))} / {len(vrecs)}）"
                 "    子分见括号；† = 对数重标定项；＊ = 纯自动测不出/错检",
                 fontsize=10, color=v1.INK2)
        for k, vr in enumerate(vrecs[p0:p0 + per_page]):
            col, row = k % 2, k // 2
            x0 = 0.06 + col * 0.47
            ytop = 0.935 - row * 0.45
            ax = fig.add_axes([x0, ytop - 0.21, 0.41, 0.20])
            ax.imshow(v1._load_gray_thumb(photo_dir / vr["file"]), cmap="gray",
                      vmin=0, vmax=255, aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(v1.RULE)
            cat = vr["category"]
            cr = census_rows[vr["file"]]
            star = "＊" if not cr["auto_ok"] else ""
            fig.text(x0, ytop + 0.008, f"{vr['file']} · 样品 {vr['sample_id']}{star}",
                     fontsize=9.5, color=v1.INK, fontweight="bold")
            fig.text(x0 + 0.41, ytop + 0.008, v1.CAT_ZH[cat], fontsize=8.5,
                     color=v1.CAT_COLOR[cat], ha="right")
            r = by_file[vr["file"]]
            ind = vr["indicators"]
            ns = r["new_subs"]
            spec = vr["spec_mm"]
            lines = [
                f"类型 {vr['type']} ｜ 厚度 {v1._fmt_th(vr['thickness_mm'])} mm ｜ "
                f"规格 {spec[0]:g}×{spec[1]:g} mm",
                f"纯自动 {cr['conclusion']} ｜ 本报告角点 {vr['detection']['provenance']} ｜ "
                f"QA {vr['qa']['level']}",
                f"X0.95 {ind['x095']:.1f} 灰度（{ns['x095']:.1f}）｜ "
                f"灰度方差 {ind['gray_variance']:.1f}（{ns['gray_variance']:.1f}†）",
                f"梯度均值 {ind['gradient_mean']:.2f}（{ns['gradient_mean']:.1f}）｜ "
                f"梯度方差 {ind['gradient_variance']:.1f}（{ns['gradient_variance']:.1f}†）",
                f"W_w {ind['texture_w']:.4f}（{ns['texture_w']:.1f}）｜ 位置评分 "
                f"{ind['position_score']:.1f}（支路 {branch_zh[ind['binding_branch']]}）",
            ]
            y = v1._text_block(fig, x0, ytop - 0.228, lines, size=7.6, dy=0.0135,
                               color=v1.INK)
            fig.text(x0, y - 0.002,
                     f"总分 {r['new_total']:.2f} ｜ 名次 {r['new_rank']}/{len(recs)} ｜ "
                     f"复核门 {v1._verif_zh(ind['verification'])}",
                     fontsize=8.6, color=v1.INK, fontweight="bold")
        yield fig


def main() -> int:
    """读 values.json + auto_census.json → 重算层 → 编号序评测 PDF（+可选预览）。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", type=Path, default=None,
                    help="同时把每页另存为 PNG 的目录（人工核版式用）")
    args = ap.parse_args()

    values = json.loads((v1.DERIVED / "values.json").read_text(encoding="utf-8"))
    census = json.loads((v2.CENSUS_DIR / "auto_census.json").read_text(encoding="utf-8"))
    man = yaml.safe_load((v1.DERIVED / "manifest.yaml").read_text(encoding="utf-8"))
    photo_dir = v1.ROOT / man["photo_dir"]
    recs = v2._rescore(values)           # v2 同一重算层（已对拍验证）
    census_rows = v2._census_summary(census)
    assert [r["file"] for r in recs] == [p["file"] for p in values["photos"]], \
        "重算记录顺序须与 values.json（图片编号序）一致"

    preview = args.preview
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
    idx = 1
    with v1.PdfPages(OUT_PDF) as pdf:
        idx = v1._save_page(pdf, page_cover(values, census, recs), preview, idx)
        idx = v1._save_page(pdf, page_chart(recs, census_rows), preview, idx)
        idx = v1._save_page(pdf, v1.page_raw_table(values), preview, idx)
        idx = v1._save_page(pdf, v2.page_new_subs_table(recs, census_rows), preview, idx)
        for fig in pages_details(values, recs, census_rows, photo_dir):
            idx = v1._save_page(pdf, fig, preview, idx)
        info = pdf.infodict()
        info["Title"] = "明之言检测结果 · 六指标评测报告 v3（新机制，编号序）"
        info["Subject"] = ("方差两项对数域子分；GlassApp v1.12 六指标；"
                           "数据源 values.json + auto_census.json")

    print(f"报告 → {OUT_PDF}（{idx - 1} 页）"
          + (f"；预览 PNG → {preview}" if preview is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

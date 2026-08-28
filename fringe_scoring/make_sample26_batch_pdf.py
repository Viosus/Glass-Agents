"""明之言检测结果 26 片批 · GlassApp 六指标批量测量报告 PDF 生成器（开发壳，不随核心包交付）。

数据源 = data/derived/sample26_thickness/values.json（make_sample26_assets.py 产物：
GlassApp 权威 v1.12 六指标对 data/images/明之言检测结果 全量打分）+ 同目录 manifest.yaml
（只取 photo_dir）。本脚本**零打分、零手填测量值**——所有数字一律取自 values.json；
values.json 缺什么本报告就缺什么，绝不补数。

产出：data/derived/sample26_thickness/明之言检测结果_六指标批量报告.pdf
页面：① 封面（口径与阅读注意）② 总分排序图 ③ 原始值汇总表 ④ 子分汇总表 ⑤ 逐片详情（2×2 缩略图）。
用法：venv python fringe_scoring/make_sample26_batch_pdf.py [--preview 目录]
（--preview 同时把每页另存 PNG，供人工核版式。）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无窗口环境直接落盘
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived" / "sample26_thickness"
OUT_PDF = DERIVED / "明之言检测结果_六指标批量报告.pdf"

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # Windows 中文字体
plt.rcParams["axes.unicode_minus"] = False

PORTRAIT, LANDSCAPE = (8.27, 11.69), (11.69, 8.27)  # A4（英寸）
INK, INK2, RULE = "#222222", "#5a5a5a", "#c8c8c8"
# 类别调色板（dataviz 校验器全项通过；绿↔橙 CVD ΔE=7.5 在 6–8 地板带 →
# 未钢化对照条另加斜纹理做第二编码；橙对白底 2.65:1 的 WARN 由③④表页兜底）
CAT_COLOR = {"core": "#3b6fb6", "special": "#d98c21", "control": "#12a06b"}
CAT_ZH = {"core": "普白钢化", "special": "特殊片", "control": "未钢化对照"}


def _load_gray_thumb(path: Path, max_side: int = 640) -> np.ndarray:
    """照片 → 灰度缩略图（imdecode 走 np.fromfile，中文路径安全；INTER_AREA 降采样）。"""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读图失败：{path}")
    h, w = img.shape
    k = max(h, w) / float(max_side)
    if k > 1.0:
        img = cv2.resize(img, (int(round(w / k)), int(round(h / k))),
                         interpolation=cv2.INTER_AREA)
    return img


def _fmt_th(t) -> str:
    """厚度显示：None（中空/夹层无单一公称厚度）→ 「—」。"""
    return "—" if t is None else f"{t:g}"


def _verif_zh(v: dict) -> str:
    """复核门结论 → 短语（passed=False 时带原因，绝不吞）。"""
    if v["passed"]:
        return "通过（触发后查证）" if v["triggered"] else "通过"
    return "未过：" + "；".join(v["reasons"])


def _new_page(figsize) -> "plt.Figure":
    """新建一页 A4 画布（白底，figsize 区分纵/横向）。"""
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    return fig


def _save_page(pdf: PdfPages, fig, preview_dir: Path | None, idx: int) -> int:
    """把一页落进 PDF（--preview 时同步另存 PNG），关闭画布并返回下一页码。"""
    pdf.savefig(fig)
    if preview_dir is not None:
        fig.savefig(preview_dir / f"page{idx:02d}.png", dpi=150)
    plt.close(fig)
    return idx + 1


def _text_block(fig, x: float, y: float, lines, size=9.5, dy=0.0165,
                color=INK, weight="normal") -> float:
    """从 y 向下逐行落笔，返回末行下方的 y。"""
    for ln in lines:
        fig.text(x, y, ln, fontsize=size, color=color, fontweight=weight)
        y -= dy
    return y


def page_cover(values: dict) -> "plt.Figure":
    """封面：出处/口径/批次概览/阅读注意。全部事实字段取自 values.json meta。"""
    meta = values["meta"]
    recs = values["photos"]
    scored = [r for r in recs if r["indicators"] is not None]
    totals = [r["indicators"]["total_score"] for r in scored]
    n_cat = {c: sum(1 for r in recs if r["category"] == c) for c in CAT_ZH}
    # 样品数与双拍名单从记录推导（8#-1/8#-2 是两个不同样品，不得手写名单）
    by_sid: dict[str, int] = {}
    for r in recs:
        by_sid[r["sample_id"]] = by_sid.get(r["sample_id"], 0) + 1
    dbl = [s for s, n in by_sid.items() if n >= 2]
    units = {r["indicators"]["x095_unit"] for r in scored}
    assert units == {"gray"}, f"X0.95 单位混杂：{units}（版式按灰度域写死，先人工核对）"
    sp = values.get("spearman_w_vs_thickness")

    fig = _new_page(PORTRAIT)
    fig.text(0.08, 0.925, "明之言检测结果 · 六指标批量测量报告", fontsize=19,
             color=INK, fontweight="bold")
    fig.text(0.08, 0.900, "26 片厚度批样品（28 张照片）· GlassApp v1.12 应力斑六指标",
             fontsize=11.5, color=INK2)
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.885, 0.885], transform=fig.transFigure,
                                color=RULE, lw=1.0))

    y = _text_block(fig, 0.08, 0.855, ["【出处与口径】"], size=12, weight="bold")
    y = _text_block(fig, 0.08, y - 0.004, [
        f"测量日期：{meta['date']}    数据源：data/images/明之言检测结果（28 张 PNG）",
        "算法：GlassApp 权威 Python 实现 fringe_scoring v1.12（texture_w 版六指标）",
        f"配置：app_config.yaml 默认方案（六项平权）    config SHA256/16 = {meta['config_sha256_16']}",
        f"打分脚本：{meta['script']}（本 PDF 仅排版，不复算）",
        f"评估域：{meta['评估域']}",
        f"尺度口径：{meta['口径']}",
    ])

    y = _text_block(fig, 0.08, y - 0.018, ["【批次概览】"], size=12, weight="bold")
    y = _text_block(fig, 0.08, y - 0.004, [
        f"照片 {values['n_photos']} 张 / 出值 {values['n_scored']} 张 / 无指标 "
        f"{values['n_no_indicators']} 张（QA fail {sum(1 for r in recs if r['qa']['level'] == 'fail')}）",
        f"类别：普白钢化 {n_cat['core']} 张 · 特殊片 {n_cat['special']} 张 · 未钢化对照 "
        f"{n_cat['control']} 张（样品 {len(by_sid)} 个，{'/'.join(dbl)} 有双拍）",
        f"总分范围 {min(totals):.2f} ~ {max(totals):.2f}，中位 {float(np.median(totals)):.2f}"
        + (f"    Spearman(W_w, 厚度) = {sp:+.4f}（普白钢化入围 {values['eligible_n']} 片）"
           if sp is not None else ""),
    ])

    y = _text_block(fig, 0.08, y - 0.018, ["【阅读注意（先读再看分）】"], size=12, weight="bold")
    notes = [
        "1. X0.95 为灰度域 95% 分位（gray→nm 标定未到位，单位=灰度）——与 nm 域国标限值",
        "   不可直接对照，只在本批内比较。",
        "2. 总分 = 默认方案口径：六项平权 + 全厚度默认参考值（refs_by_thickness 未启用）。",
        "   厚玻璃应力斑天生更重（见上 Spearman），跨厚度直接比总分不公平；同厚度内比较才有意义。",
        "3. 判定线 fail_line 未启用——本报告只报分，不出「合格/不合格」判定。",
        "4. 8#-1/8#-2 为未钢化零点对照：暗场下几乎不发亮 → 高分是「无应力斑信号」的零点语义，",
        "   不代表质量好（未钢化片不属于钢化品评价范畴）。",
        "5. 评估域差异：W_w 取整片矫正图（含边框带，量化刻度承重设计）；其余五项取扣除",
        "   免罚边框带后的内部。位置评分为集中度/覆盖度双支路取小。",
        "6. 特殊片（中空/夹层/压花/Low-E/均质/高应力防火）光学条件各异，其分数只作定性参考，",
        "   不与普白钢化同池排名解读。",
    ]
    y = _text_block(fig, 0.08, y - 0.004, notes, size=9.2, dy=0.0158)

    fig.text(0.08, 0.06, "生成脚本：fringe_scoring/make_sample26_batch_pdf.py（数据源 values.json，"
             "逐值同源可追溯）", fontsize=8, color=INK2)
    return fig


def page_bar_chart(values: dict) -> "plt.Figure":
    """总分排序横条图：单序列（总分），类别以颜色+纹理区分；精确值见表页。"""
    recs = [r for r in values["photos"] if r["indicators"] is not None]
    recs = sorted(recs, key=lambda r: r["indicators"]["total_score"], reverse=True)
    labels = [Path(r["file"]).stem for r in recs]
    totals = [r["indicators"]["total_score"] for r in recs]
    cats = [r["category"] for r in recs]

    fig = _new_page(PORTRAIT)
    fig.text(0.08, 0.945, "六指标加权总分排序（默认方案，0–100，越高越好）",
             fontsize=13, color=INK, fontweight="bold")
    ax = fig.add_axes([0.14, 0.075, 0.78, 0.83])
    ypos = np.arange(len(recs))
    for i, (t, c) in enumerate(zip(totals, cats)):
        ax.barh(i, t, height=0.62, color=CAT_COLOR[c],
                hatch="///" if c == "control" else None,
                edgecolor="white" if c == "control" else "none", linewidth=0.0)
    ax.set_yticks(ypos, labels=labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("总分（默认方案）", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=CAT_COLOR["core"], label="普白钢化"),
        Patch(facecolor=CAT_COLOR["special"], label="特殊片"),
        Patch(facecolor=CAT_COLOR["control"], hatch="///", edgecolor="white",
              label="未钢化对照（零点语义）"),
    ], fontsize=8, framealpha=0.9, loc="lower right")
    fig.text(0.14, 0.028, "跨厚度/跨类别比较不公平（封面注意 2/6）；精确值见后两页表。",
             fontsize=8, color=INK2)
    return fig


# 表页列定义：(表头, x 位置, 对齐, 取值函数)
def _table_page(title: str, cols, recs, foot: str) -> "plt.Figure":
    """通用汇总表页（A4 横向）：表头加粗 + 零星横规线 + 隔行浅底。"""
    fig = _new_page(LANDSCAPE)
    fig.text(0.03, 0.945, title, fontsize=12.5, color=INK, fontweight="bold")
    y0, dy = 0.885, 0.0272
    for h, x, ha, _ in cols:
        fig.text(x, y0, h, fontsize=8.2, color=INK, fontweight="bold", ha=ha)
    fig.lines.append(plt.Line2D([0.03, 0.97], [y0 - 0.008, y0 - 0.008],
                                transform=fig.transFigure, color=INK2, lw=0.8))
    y = y0 - dy
    for i, r in enumerate(recs):
        if i % 2 == 1:
            fig.patches.append(Rectangle((0.03, y - 0.006), 0.94, dy,
                                         transform=fig.transFigure,
                                         facecolor="#f2f2f2", zorder=0))
        for _, x, ha, get in cols:
            fig.text(x, y, get(r), fontsize=7.8, color=INK, ha=ha)
        y -= dy
    fig.text(0.03, max(y - 0.006, 0.02), foot, fontsize=7.8, color=INK2)
    return fig


def page_raw_table(values: dict) -> "plt.Figure":
    """③ 六指标原始值汇总表（含厚度/角点来源/复核门）。"""
    def ind(key, fmt):
        """原始值列取值器：按 fmt 格式化指标 key，无指标记录显示「—」。"""
        return lambda r: (fmt.format(r["indicators"][key]) if r["indicators"] else "—")

    cols = [
        ("文件", 0.030, "left", lambda r: r["file"]),
        ("样品", 0.085, "left", lambda r: r["sample_id"]),
        ("类型", 0.135, "left", lambda r: r["type"][:18]),
        ("厚mm", 0.360, "right", lambda r: _fmt_th(r["thickness_mm"])),
        ("角点", 0.415, "right", lambda r: r["detection"]["provenance"]),
        ("X0.95(灰度)", 0.500, "right", ind("x095", "{:.1f}")),
        ("灰度方差", 0.575, "right", ind("gray_variance", "{:.1f}")),
        ("梯度均值", 0.645, "right", ind("gradient_mean", "{:.2f}")),
        ("梯度方差", 0.715, "right", ind("gradient_variance", "{:.1f}")),
        ("W_w", 0.775, "right", ind("texture_w", "{:.4f}")),
        ("位置评分", 0.840, "right", ind("position_score", "{:.1f}")),
        ("总分", 0.895, "right", ind("total_score", "{:.2f}")),
        ("复核", 0.970, "right",
         lambda r: ("—" if not r["indicators"]
                    else ("通过" if r["indicators"]["verification"]["passed"] else "未过"))),
    ]
    foot = ("六项原始值方向：前五项越小越好，位置评分越大越好；W_w 报 4 位小数（报数精度口径）。"
            "角点 auto=自动检测 / manual=人工定角。")
    return _table_page("六指标原始值汇总（28 张照片，按样品号序）", cols,
                       values["photos"], foot)


def page_sub_table(values: dict) -> "plt.Figure":
    """④ 六指标 0–100 子分与总分汇总表（含位置评分生效支路）。"""
    def sub(key):
        """子分列取值器：取 sub_scores[key] 一位小数，无指标记录显示「—」。"""
        return lambda r: (f"{r['indicators']['sub_scores'][key]:.1f}"
                          if r["indicators"] else "—")

    branch_zh = {"coverage": "覆盖度", "concentration": "集中度", None: "—"}
    cols = [
        ("文件", 0.030, "left", lambda r: r["file"]),
        ("样品", 0.085, "left", lambda r: r["sample_id"]),
        ("类型", 0.135, "left", lambda r: r["type"][:18]),
        ("X0.95", 0.400, "right", sub("x095")),
        ("灰度方差", 0.475, "right", sub("gray_variance")),
        ("梯度均值", 0.550, "right", sub("gradient_mean")),
        ("梯度方差", 0.625, "right", sub("gradient_variance")),
        ("W_w", 0.685, "right", sub("texture_w")),
        ("位置评分", 0.755, "right", sub("position_score")),
        ("总分", 0.815, "right",
         lambda r: (f"{r['indicators']['total_score']:.2f}" if r["indicators"] else "—")),
        ("位置支路", 0.895, "right",
         lambda r: branch_zh.get((r["indicators"] or {}).get("binding_branch"), "—")),
        ("QA", 0.955, "right", lambda r: r["qa"]["level"]),
    ]
    foot = ("子分 = 100×(worst−v)/(worst−best)（默认全厚度参考值），越高越好；总分 = 六项平权均值。"
            "位置支路 = 集中度/覆盖度双支路取小时的生效侧。")
    return _table_page("六指标子分与加权总分（0–100）", cols, values["photos"], foot)


def pages_details(values: dict, photo_dir: Path):
    """⑤ 逐片详情页（2×2：缩略图 + 指标文本块），生成器逐页产出。"""
    recs = values["photos"]
    per_page = 4
    for p0 in range(0, len(recs), per_page):
        fig = _new_page(PORTRAIT)
        fig.text(0.06, 0.955, f"逐片详情（{p0 + 1}–{min(p0 + per_page, len(recs))} / {len(recs)}）",
                 fontsize=11, color=INK2)
        for k, r in enumerate(recs[p0:p0 + per_page]):
            col, row = k % 2, k // 2
            x0 = 0.06 + col * 0.47
            ytop = 0.935 - row * 0.45
            ax = fig.add_axes([x0, ytop - 0.21, 0.41, 0.20])
            ax.imshow(_load_gray_thumb(photo_dir / r["file"]), cmap="gray",
                      vmin=0, vmax=255, aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(RULE)
            cat = r["category"]
            fig.text(x0, ytop + 0.008, f"{r['file']} · 样品 {r['sample_id']}",
                     fontsize=9.5, color=INK, fontweight="bold")
            fig.text(x0 + 0.41, ytop + 0.008, CAT_ZH[cat], fontsize=8.5,
                     color=CAT_COLOR[cat], ha="right")
            spec = r["spec_mm"]
            qa = r["qa"]
            det = r["detection"]
            lines = [
                f"类型 {r['type']} ｜ 厚度 {_fmt_th(r['thickness_mm'])} mm ｜ "
                f"规格 {spec[0]:g}×{spec[1]:g} mm",
                f"角点 {det['provenance']} ｜ 检出 {det['n_sheets']} 片 ｜ QA {qa['level']}"
                + ("（" + "；".join(qa["reasons"]) + "）" if qa["reasons"] else ""),
            ]
            cc = r["crop_check"]
            if cc is not None:
                lines.append(f"裁切覆盖度 长边 {cc['coverage_long']:.1%} / 短边 "
                             f"{cc['coverage_short']:.1%} ｜ 长宽比偏差 {cc['aniso']:.1%}")
            ind = r["indicators"]
            if ind is None:
                lines.append("无指标值（QA fail，见原因）")
            else:
                ss = ind["sub_scores"]
                branch_zh = {"coverage": "覆盖度", "concentration": "集中度"}
                lines += [
                    f"X0.95 {ind['x095']:.1f} 灰度（子分 {ss['x095']:.1f}）｜ "
                    f"灰度方差 {ind['gray_variance']:.1f}（{ss['gray_variance']:.1f}）",
                    f"梯度均值 {ind['gradient_mean']:.2f}（{ss['gradient_mean']:.1f}）｜ "
                    f"梯度方差 {ind['gradient_variance']:.1f}（{ss['gradient_variance']:.1f}）",
                    f"W_w {ind['texture_w']:.4f}（{ss['texture_w']:.1f}）｜ 位置评分 "
                    f"{ind['position_score']:.1f}（支路 {branch_zh[ind['binding_branch']]}）",
                ]
            y = _text_block(fig, x0, ytop - 0.228, lines, size=7.6, dy=0.0135, color=INK)
            if ind is not None:
                fig.text(x0, y - 0.002,
                         f"总分 {ind['total_score']:.2f} ｜ 复核门 {_verif_zh(ind['verification'])}",
                         fontsize=8.6, color=INK, fontweight="bold")
        yield fig


def main() -> int:
    """读 values.json + manifest → 封面/排序图/两张汇总表/逐片详情 → 落 PDF。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", type=Path, default=None,
                    help="同时把每页另存为 PNG 的目录（人工核版式用）")
    args = ap.parse_args()

    values = json.loads((DERIVED / "values.json").read_text(encoding="utf-8"))
    man = yaml.safe_load((DERIVED / "manifest.yaml").read_text(encoding="utf-8"))
    photo_dir = ROOT / man["photo_dir"]
    assert values["n_photos"] == len(values["photos"]), "values.json 计数与记录数不符"

    preview = args.preview
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)

    idx = 1
    with PdfPages(OUT_PDF) as pdf:
        idx = _save_page(pdf, page_cover(values), preview, idx)
        idx = _save_page(pdf, page_bar_chart(values), preview, idx)
        idx = _save_page(pdf, page_raw_table(values), preview, idx)
        idx = _save_page(pdf, page_sub_table(values), preview, idx)
        for fig in pages_details(values, photo_dir):
            idx = _save_page(pdf, fig, preview, idx)
        info = pdf.infodict()
        info["Title"] = "明之言检测结果 · 六指标批量测量报告"
        info["Subject"] = ("GlassApp v1.12 fringe_scoring 六指标；数据源 "
                           "data/derived/sample26_thickness/values.json")

    print(f"报告 → {OUT_PDF}（{idx - 1} 页）"
          + (f"；预览 PNG → {preview}" if preview is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""《应力斑位置评分_测试集检测报告》PDF 生成器（开发壳，不随核心包交付）。

全部带人工评分的测试图的逐片检测/打分记录：批次概要（可判数、检测异常
说明、分布统计与人工分一致性）+ 逐片明细表（人工分/系统参考分/位置评分/
ρu/加权覆盖度/主导支路/X0.95/W_w/备注）。数值取自
data/derived/uniformity_doc/values.json（与技术说明图文同源，同一次实测）。
用法：venv python fringe_scoring/make_batch_report_doc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 脚本直跑时可 import 包

from matplotlib.backends.backend_pdf import PdfPages

from fringe_scoring.make_uniformity_gb_doc import _vals, render_pages

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-07-22"
ALGO_VER = "GlassApp v1.10.0（位置评分双支路口径）"
_ROWS_PER_PAGE = 40

# 检测异常三例的处置记录（2026-07-22 修复/核实，详见仓库 commit 12bc2837）
_NOTES = {
    17: "检测修复：中带亮条曾致整图拒判",
    25: "画幅右缘含真实邻片残段（检出2片，取主片）",
    88: "检测修复：片内亮团曾误当第二片",
}


def _row(r: dict) -> str:
    """明细表一行：编号/人工/系统参考/评分/ρu/A_w/支路/X0.95/W_w/备注。"""
    branch = "覆盖" if r["branch"] == "coverage" else "集中"
    note = _NOTES.get(r["no"], "")
    return (f"{r['no']:03d}　　{r['man']:>3.0f}　　{r['sys']:>5.1f}　　{r['U']:>5.1f}　　"
            f"{r['rho']:.3f}　　{r['cov']:.3f}　　{branch}　　{r['x095']:>6.1f}　　"
            f"{r['texture_w']:.4f}{'　← ' + note if note else ''}")


_HEADER = ("编号　　人工　　系统参考　　评分　　 ρu　　　A_w　　支路　　X0.95　　 W_w")


def report_pages() -> list[list[tuple[str, str]]]:
    """全部页面（概要一页 + 明细表分页）。"""
    V = _vals()
    rows = sorted(V["all"], key=lambda r: r["no"])

    def _band_row(b: dict) -> str:
        """分档统计一行。"""
        return (f"{b['man']:>4.0f}　　　{b['n']:>3d}　　　{b['u_mean']:>5.1f}　　"
                f"[{b['u_min']:>5.1f} , {b['u_max']:>5.1f}]　　{b['rho_mean']:.3f}　　　{b['cov_mean']:.3f}")

    pages = [[
        ("space", "0.05"),
        ("h1", "应力斑位置评分 · 测试集检测报告"),
        ("space", "0.008"),
        ("cnote", f"{DATE}　|　算法 {ALGO_VER}　|　数值与技术说明第五部分同源"),
        ("space", "0.020"),
        ("h2", "批次概要"),
        ("body",
         f"测试集共 {V['n_scored']} 张带人工评分的单片应力斑照片（另含少量整床参考照，不在\n"
         f"本报告统计内），处理分辨率长边 2000 像素。全部 {V['n_scored']} 张检出为单片并成功\n"
         "打分，无拒判。检测环节三例特殊情况及处置：\n"
         "· 017——片内中带亮条曾被误当独立连通域且拟合不出四角、导致整图拒判，\n"
         "　已由\"包含剔除\"规则修复（玻璃片不可能整片落在另一片凸包之内）；\n"
         "· 088——片内亮团曾被误检为第二片且抢占阅读序首位（曾算错对象），同上修复；\n"
         "· 025——画幅右缘存在真实邻片残段（自带边框），检出两片属正确行为，\n"
         "　统计取阅读序主片。"),
        ("space", "0.008"),
        ("body",
         f"打分一致性：位置评分与人工评分秩相关（Spearman，并列取平均秩）= {V['spearman']}；\n"
         f"评分范围 [{V['u_min']}, {V['u_max']}]，低于 60 分 {V['n_below_60']} 张。分档均值总体随人工分\n"
         "上升，但**并非逐档严格单调**——人工 90 分档均值低于 85 分档（两档分别\n"
         "13 片与 17 片，最高两档样本量偏小），如实列于下表。"),
        ("space", "0.008"),
        ("tbl",
         "人工分　　片数　　评分均值　　[ 最低 ,  最高 ]　　　ρu均值　　A_w均值\n"
         + "\n".join(_band_row(b) for b in V["bands"])),
        ("space", "0.010"),
        ("note",
         "列说明：人工=专家分档打分（50~95，5 分一档）；系统参考=图片文件名携带的外部\n"
         "系统参考分，\n"
         "仅作对照非本算法输出；评分=位置评分（双支路取小，0~100 越高越好）；ρu=中心\n"
         "集中度（越大越差）；A_w=位置加权覆盖度；支路=最终评分由哪条支路给出。\n"
         "X0.95 为灰度域代理值（未作灰度—光程差标定），同批相对可比；\n"
         "W_w = 位置加权组合纹理指数（越大越差，归一分母为数学上界、零标定）。"),
    ]]

    for i in range(0, len(rows), _ROWS_PER_PAGE):
        chunk = rows[i: i + _ROWS_PER_PAGE]
        pages.append([
            ("space", "0.04"),
            ("h3", f"逐片明细（{chunk[0]['no']:03d}–{chunk[-1]['no']:03d}）"),
            ("tbl", _HEADER + "\n" + "\n".join(_row(r) for r in chunk)),
        ])
    return pages


def main() -> int:
    """生成《应力斑位置评分_测试集检测报告》PDF 到 docs/。"""
    out = ROOT / "docs" / "应力斑位置评分_测试集检测报告.pdf"
    with PdfPages(out) as pdf:
        render_pages(pdf, report_pages())
    print(f"已生成 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""草案 §4.3 替代条文建议文本生成器（开发壳，不随核心包交付）。

任务（2026-08-05 用户）：以《钢化玻璃应力斑分级及检测方法》草案相同体例，把
§4.3（基于各向同性值 IsoT 的质量等级）替换为基于位置加权组合纹理指数 W_w 与
应力斑位置评分 U 的分级条文。定位 = **征求意见的建议文本**（非标准正文）；
用户拍板：分级限值填批内初标示例值 + 待复标脚注（「草案不必过于严谨」口径）。

体例仿草案 §4.4 样板：引导句 + 全角编号步骤 a)～f) + 居中公式行尾点线引至
（N）+「式中：」逐符号「——」说明 + 厚度档三线表 + 表下脚注。

限值由脚本确定性推导（图文同源，不手抄）：
- W_w 厚度档 A/B 切点 = refs_by_thickness_draft.yaml 各档 [best, worst] 三等分
  内插（A_max = best + 1/3 区间、B_max = best + 2/3 区间）；
- U 切点 = uniformity_doc/values.json 人工分档 u_mean 的相邻档中点（A 切点取
  85|80 档中点、B 切点取 70|65 档中点），就近取 5 的倍数。

产出：docs/应力斑国标文档/应力斑分级_4.3替代条文建议_Ww与位置评分.pdf + .docx
用法：venv python fringe_scoring/make_gb43_draft_doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无窗口环境直接落盘
import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 以脚本方式直跑时保证 fringe_scoring 包可导入
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "docs" / "应力斑国标文档"
OUT_STEM = "应力斑分级_4.3替代条文建议_Ww与位置评分"
DRAFT_YAML = ROOT / "data" / "derived" / "sample26_thickness" / "refs_by_thickness_draft.yaml"
U_VALUES = ROOT / "data" / "derived" / "uniformity_doc" / "values.json"
DATE = "2026-08-05"
VERSION = "V1.0"

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PAGE = (8.27, 11.69)  # A4 纵向（英寸）
# 版式：块类型 → (字号, 行步进)；gbtbl 单独处理（三线表）
_STYLES = {"h1": (15.5, 0.046), "sub": (10.0, 0.028), "h2": (12.0, 0.034),
           "h3": (10.5, 0.030), "body": (9.8, 0.0215), "formula": (11.0, 0.040),
           "vars": (9.0, 0.0195), "note": (8.5, 0.0185), "tblrow": (9.0, 0.0250)}
_INK, _GRAY, _RULE = "black", "dimgray", "#404040"


def _round5(x: float) -> int:
    """就近取 5 的倍数（U 切点化整口径）。"""
    return int(round(x / 5.0) * 5)


def derive_limits() -> dict:
    """从两份已入库资产确定性推导限值：W_w 厚度档三等分内插 + U 相邻档中点。"""
    draft = yaml.safe_load(DRAFT_YAML.read_text(encoding="utf-8"))
    ww_rows = []
    for band in draft["refs_by_thickness"]:
        b = float(band["refs"]["texture_w"]["best"])
        w = float(band["refs"]["texture_w"]["worst"])
        span = w - b
        ww_rows.append({"t": band["max_thickness_mm"], "n_note": None,
                        "a_max": round(b + span / 3.0, 4),
                        "b_max": round(b + 2.0 * span / 3.0, 4)})
    uv = json.loads(U_VALUES.read_text(encoding="utf-8"))
    bands = {int(x["man"]): x for x in uv["bands"]}
    u_a = _round5((bands[85]["u_mean"] + bands[80]["u_mean"]) / 2.0)
    u_b = _round5((bands[70]["u_mean"] + bands[65]["u_mean"]) / 2.0)
    assert 0 < u_b < u_a < 100, "U 切点推导越界，须人工核对 bands 数据"
    return {"ww_rows": ww_rows, "u_a_min": u_a, "u_b_min": u_b,
            "u_bands_n": {k: bands[k]["n"] for k in (65, 70, 80, 85)}}


# ---------- 页面渲染（本脚本局部实现，含 gbtbl 三线表；不动共享引擎） ----------

def _draw_gbtbl(fig, y: float, text: str) -> float:
    """三线表：首行=表题（居中黑体），次行=表头，余行=数据；顶/栏头/底三横线。

    单元格以 2 个及以上全角空格分隔；列 x 位置按各列最大字符宽估算分配。
    返回表格底部之下的 y。
    """
    import re

    lines = text.split("\n")
    title, rows = lines[0], [re.split(r"\u3000{2,}", ln.strip("\u3000"))
                             for ln in lines[1:] if ln.strip()]
    ncols = max(len(r) for r in rows)
    widths = [max((len(r[c]) if c < len(r) else 0) for r in rows) for c in range(ncols)]
    total = sum(widths) or 1
    x0, x1 = 0.10, 0.90
    # 各列中心 x（按字符宽占比分配列宽）
    centers, cur = [], x0
    for c in range(ncols):
        wfrac = (x1 - x0) * widths[c] / total
        centers.append(cur + wfrac / 2.0)
        cur += wfrac
    size, step = _STYLES["tblrow"]
    fig.text(0.5, y, title, fontsize=10.0, ha="center", fontweight="bold")
    y -= 0.030
    top = y + 0.012
    for ri, row in enumerate(rows):
        for c, cell in enumerate(row):
            fig.text(centers[c], y, cell, fontsize=size, ha="center",
                     fontweight="bold" if ri == 0 else "normal")
        y -= step
        if ri == 0:  # 栏头线
            fig.lines.append(plt.Line2D([x0, x1], [y + step - 0.008] * 2,
                                        transform=fig.transFigure, color=_RULE, lw=0.8))
    # 顶线与底线（粗）
    for yy in (top, y + step - 0.010):
        fig.lines.append(plt.Line2D([x0, x1], [yy] * 2, transform=fig.transFigure,
                                    color=_RULE, lw=1.4))
    return y - 0.008


def render_page_gb(pdf: PdfPages, blocks: list[tuple[str, str]]) -> None:
    """一页条文的局部渲染器（h1/sub/h2/h3/body/formula/vars/note/gbtbl/space）。"""
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    y = 0.94
    for kind, text in blocks:
        if kind == "space":
            y -= float(text)
            continue
        if kind == "gbtbl":
            y = _draw_gbtbl(fig, y, text)
            continue
        size, step = _STYLES[kind]
        for line in text.split("\n"):
            if kind in ("h1", "sub"):
                fig.text(0.5, y, line, fontsize=size, ha="center",
                         fontweight="bold" if kind == "h1" else "normal",
                         color=_INK if kind == "h1" else _GRAY)
            elif kind == "formula":
                fig.text(0.5, y, line, fontsize=size, ha="center")
            else:
                fig.text(0.09, y, line, fontsize=size,
                         fontweight="bold" if kind in ("h2", "h3") else "normal",
                         color=_GRAY if kind == "note" else _INK)
            y -= step
    pdf.savefig(fig)
    plt.close(fig)


# ---------- 条文内容（全部限值来自 derive_limits，零手抄） ----------

def doc_pages(lim: dict) -> list[list[tuple[str, str]]]:
    """条文页面：前置说明 → 术语补条 → 4.3 正文（4.3.1/4.3.2）→ 表 2a/2b → 脚注。"""
    ww = lim["ww_rows"]
    ua, ub = lim["u_a_min"], lim["u_b_min"]
    ww_tbl_rows = "\n".join(
        f"{r['t']:g}　　≤{r['a_max']}　　＞{r['a_max']} 且≤{r['b_max']}　　＞{r['b_max']}"
        for r in ww)
    return [
        # ── P1 封面/前置说明 + 术语补条 ──
        [
            ("space", "0.02"),
            ("h1", "《钢化玻璃应力斑分级及检测方法》草案 §4.3 替代条文（建议文本）"),
            ("sub", f"基于位置加权组合纹理指数 W_w 与应力斑位置评分 U 的质量等级　"
                    f"{VERSION}　{DATE}"),
            ("space", "0.012"),
            ("note",
             "【文件性质】本文本为对草案（2026.5.18 征求意见稿）第 4.3 条的替代条文建议，"
             "供标准工作组讨论，\n非标准正文。条文体例（步骤编号、公式引出、三线表、脚注）"
             "仿草案 §4.4 样板。表中分级限值为批内初标\n示例值（脚注注明来源与样本量），"
             "正式发布前必须由同厚度好/差样片人工标定复核。两项指标的完整定义、\n"
             "数学性质与实测验证见《应力斑位置加权组合纹理指数W_w_技术说明》（V1.5）与"
             "《应力斑位置评分指标_技术说明》（V2.1）。"),
            ("space", "0.016"),
            ("h2", "一、术语补条（拟并入草案第 3 章）"),
            ("h3", "3.x　位置加权组合纹理指数　position-weighted combined texture index (W_w)"),
            ("body",
             "　　在评估区域 M 内，对光程差图像按位置权重 w(r)=1−r 计票构建灰度共生矩阵，"
             "由对比度与聚类突出\n　　两个分量经理论上界归一化后组合而成的无量纲纹理指数，"
             "取值 [0,1)，数值越大表示应力斑纹理越重、\n　　越靠近板面中心。"),
            ("space", "0.006"),
            ("h3", "3.y　应力斑位置评分　fringe position score (U)"),
            ("body",
             "　　在评估区域 M 内，对判定为应力斑的像素按其深浅与位置加权得到的 0～100 评分，"
             "由中心集中度支路\n　　与加权覆盖度支路取小而得；数值越大表示斑纹越轻、"
             "越远离板面中心。"),
            ("space", "0.014"),
            ("h2", "二、替代条文正文"),
            ("h3", "4.3　基于位置加权组合纹理指数与位置评分的质量等级"),
            ("body",
             "　　应按 4.3.1 计算评估区域 M 内的位置加权组合纹理指数 W_w，按 4.3.2 计算"
             "应力斑位置评分 U，\n　　并按表 2a 与表 2b 的规定分别确定等级，"
             "取二者中较严（较低）者为该片玻璃的最终质量等级。"),
        ],
        # ── P2 4.3.1 W_w 计算 ──
        [
            ("space", "0.02"),
            ("h3", "4.3.1　位置加权组合纹理指数 W_w 的计算"),
            ("body",
             "　　W_w 的计算应符合以下步骤：\n"
             "　　a） 按第 6 章确定评估区域 M，取 M 内的光程差图像；\n"
             "　　b） 图像标准化：重采样至 1 px/mm（同 4.4 b) 口径）；\n"
             "　　c） 按公式(1)、公式(2)计算各像素的归一化坐标与位置权重；\n"
             "　　d） 将图像按该片图像的 [最小值, 最大值] 线性量化至 Ng=8 个灰度级"
             "（同 4.4 a) 口径）；\n"
             "　　e） 在 0°、45°、90°、135° 四个方向构建位置加权灰度共生矩阵：每个共生"
             "像素对按公式(3)的\n"
             "　　　　端点权重算术平均计票，四向平均后归一化；\n"
             "　　f） 按公式(4)、公式(5)提取位置加权对比度与聚类突出，按公式(6)计算 W_w。"),
            ("space", "0.006"),
            ("formula", r"$u = 2(x_c+0.5)/N_c - 1,\ \ v = 2(x_r+0.5)/N_r - 1,"
                        r"\ \ r = \max(|u|,|v|)$　……………………（1）"),
            ("formula", r"$w(r) = 1 - r$　………………………………………………………………（2）"),
            ("formula", r"$w_{pair} = \frac{1}{2}(w_p + w_{p'})$　"
                        r"…………………………………………………（3）"),
            ("formula", r"$C_{a,w} = \sum_{i,j} P_w(i,j)\,(i-j)^2$　"
                        r"……………………………………………（4）"),
            ("formula", r"$CP_{a,w} = \sum_{i,j} P_w(i,j)\,(i+j-\mu_i-\mu_j)^4$　"
                        r"…………………………（5）"),
            ("formula", r"$W_w = \frac{1}{2}\left[\sqrt{C_{a,w}/(N_g-1)^2} + "
                        r"\sqrt[4]{CP_{a,w}/\left((2(N_g-1))^4/12\right)}\right]$　"
                        r"……………（6）"),
            ("vars",
             "式中：\n"
             "W_w　　——位置加权组合纹理指数，无量纲，越大越差；\n"
             "x_c、x_r——像素的列号与行号；N_c、N_r——评估区图像的列数与行数；\n"
             "r　　　——像素到板面中心的切比雪夫归一化距离；w(r)——位置权重（中心 1，"
             "向边缘线性降至 0，无下限）；\n"
             "w_p、w_p'——共生像素对两端点的位置权重；P_w(i,j)——位置加权归一化灰度共生矩阵；\n"
             "μ_i、μ_j——P_w 的行/列边缘均值；N_g——灰度级数（=8）；\n"
             "(N_g−1)²、(2(N_g−1))⁴/12——两分量的纯数学理论上界（=49 与 3201.33，"
             "非标定值，任何实验室同数）。"),
            ("space", "0.006"),
            ("note",
             "注 1：W_w 对逐片仿射灰度变换不变，故光程差未经 nm 标定（灰度代理）时公式与"
             "数值均不受影响。\n"
             "注 2：量化刻度取整片评估区图像的极值——评估域改变即数值改变，跨实验室比对时"
             "评估域必须一致。\n"
             "注 3：本条位置权重 w(r)=1−r 为零常数几何定义，与 4.3.2 的 w′(r) 不同，"
             "二者不得混用。"),
        ],
        # ── P3 4.3.2 位置评分计算 ──
        [
            ("space", "0.02"),
            ("h3", "4.3.2　应力斑位置评分 U 的计算"),
            ("body",
             "　　U 的计算应符合以下步骤：\n"
             "　　a） 在评估区域 M 内按稳健标准化（中位数/MAD）判定应力斑像素域 Ω 及其"
             "深浅强度 s（判斑\n"
             "　　　　口径与预处理见《应力斑位置评分指标_技术说明》，本条不重复）；\n"
             "　　b） 按公式(7)计算带下限的位置权重 w′(r)（r 同公式(1)）；\n"
             "　　c） 按公式(8)、公式(9)计算中心集中度 ρ_u 与集中度评分 U_ρ；按公式(10)、"
             "公式(11)计算位置\n"
             "　　　　加权覆盖度 A_w 与覆盖度评分 U_cov；\n"
             "　　d） 按公式(12)取两支路较小者为 U。"),
            ("space", "0.006"),
            ("formula", r"$w'(r) = \max(1-r,\ 0.3)$　"
                        r"………………………………………………………（7）"),
            ("formula", r"$\rho_u = \sum_{p\in\Omega} s(p)\,w'(r_p)\ /\ "
                        r"\sum_{p\in M} w'(r_p)$　………………………………（8）"),
            ("formula", r"$U_\rho = \mathrm{clip}\left(100\times(1-\rho_u/\rho_0),"
                        r"\ 0,\ 100\right),\ \ \rho_0=0.33$　………………（9）"),
            ("formula", r"$A_w = \sum_{p\in M:\ dev(p)\geq G_0} w'(r_p)\ /\ "
                        r"\sum_{p\in M} w'(r_p),\ \ G_0=30$　………………（10）"),
            ("formula", r"$U_{cov} = \mathrm{clip}\left(100\times"
                        r"\left(1-\frac{A_w-C_0}{C_1-C_0}\right),\ 0,\ 100\right),"
                        r"\ \ C_0=0.10,\ C_1=0.65$　……（11）"),
            ("formula", r"$U = \min(U_\rho,\ U_{cov})$　"
                        r"…………………………………………………………（12）"),
            ("vars",
             "式中：\n"
             "Ω　　——判定为应力斑的像素集合；s(p)——像素 p 的斑深浅强度（0～1）；\n"
             "w′(r)——带下限 0.3 的位置权重（边缘重斑不免罚）；\n"
             "ρ_u　——中心集中度（0～1，越大越差）；ρ_0——集中度零分线；\n"
             "dev(p)——像素 p 的绝对灰度偏差；G_0——覆盖判斑灰度阈；\n"
             "A_w　——位置加权覆盖度；C_0、C_1——覆盖度不扣分下限与零分上限；\n"
             "U　　——应力斑位置评分（0～100，越大越好）。"),
            ("space", "0.006"),
            ("note",
             "注 4：ρ_0、G_0、C_0、C_1 为本条规定值（121 片人工标注集初标口径）；"
             "供需双方可在合同中另行约定。"),
        ],
        # ── P4 分级表 + 脚注 ──
        [
            ("space", "0.02"),
            ("h3", "4.3.3　质量等级判定"),
            ("body",
             "　　应按表 2a 的规定根据 W_w 确定纹理等级，按表 2b 的规定根据 U 确定位置"
             "等级；该片玻璃的最终\n　　质量等级取二者中较严（较低）者。"),
            ("space", "0.010"),
            ("gbtbl",
             "表 2a　基于位置加权组合纹理指数 W_w 的质量等级（注 a）\n"
             "玻璃公称厚度（mm）　　A级　　B级　　C级\n"
             + ww_tbl_rows + "\n"
             "≥10（注 b）　　b　　b　　b"),
            ("space", "0.012"),
            ("gbtbl",
             "表 2b　基于应力斑位置评分 U 的质量等级（注 c）\n"
             "适用范围　　A级　　B级　　C级\n"
             f"全部公称厚度　　≥{ua}　　＜{ua} 且≥{ub}　　＜{ub}"),
            ("space", "0.012"),
            ("note",
             "a：W_w 限值为 26 片带厚度标注样片批的批内初标示例值（每档 n=2～5，"
             "档内 [最好, 最差] 三等分内插），\n"
             "　  正式发布前须按同厚度好/差样片人工标定复核；限值保留 4 位小数"
             "（W_w 报数精度要求，见技术说明第七部分）。\n"
             "　  表中 5 mm 档限值低于 4 mm 档、8 mm 档低于 6 mm 档，系小样本波动"
             "（档间差在单片重复性 ±0.02～0.04 量级内），\n"
             "　  复标时应期待限值随厚度单调上升。\n"
             "b：对于公称厚度大于等于 10 mm 的钢化玻璃，限值由供需双方商定"
             "（初标批 10/12/15 mm 档均为单片，不足以出档）。\n"
             "c：U 限值为 115 片人工标注集初标（A 切点取人工分 85|80 档均值中点、"
             "B 切点取 70|65 档中点，就近取 5 的\n"
             "　  倍数），全厚度统一；正式发布前须扩样复核。"),
            ("space", "0.014"),
            ("h2", "三、与草案其他条文的衔接说明（资料性）"),
            ("body",
             "　　1. 本条替代现行 4.3（基于各向同性值 IsoT）；4.1 总则的「多法判定不一致时"
             "取最严」与「供需双方\n　　　  可约定特定方法」两款对本条继续适用。\n"
             "　　2. W_w 保留了 4.4（CCP）的处理骨架与两个纹理分量，但两个归一化分母为"
             "纯数学理论上界、共生对\n　　　  按位置权重计票——数值与 CCP 不可互换算，"
             "亦不得沿用表 3 的限值。\n"
             "　　3. 附录 A（三种评估方法的特点比较）若采纳本条，宜将 IsoT 列替换为 "
             "W_w 与 U 两列。"),
            ("space", "0.010"),
            ("note",
             "【诚实披露】W_w 的两批实测验证（115 片人工标注集、480 片产线批）均在整片"
             "矫正图口径上完成，与本条\n定义层的 M 域（扣除边缘排除区 E 与孔洞排除区 H）"
             "不一致；正式采标前须按 M 域几何复核限值（W_w 的量化\n刻度取输入域极值，"
             "换评估域即换数值）。位置评分不受此影响（其口径即扣边框带内部）。\n"
             f"文档 {VERSION}（{DATE}）。生成脚本 fringe_scoring/make_gb43_draft_doc.py；"
             "限值由 refs_by_thickness_draft.yaml\n与 uniformity_doc/values.json "
             "确定性推导（图文同源）。"),
        ],
    ]


def main() -> int:
    """推导限值 → 渲染 PDF → build_docx 出 Word（gbtbl 映射回 tbl）。"""
    lim = derive_limits()
    pages = doc_pages(lim)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_out = OUT_DIR / f"{OUT_STEM}.pdf"
    with PdfPages(pdf_out) as pdf:
        for blocks in pages:
            render_page_gb(pdf, blocks)
    print(f"已生成 → {pdf_out}")

    from fringe_scoring.make_docx_versions import _assert_no_tex_residue, build_docx

    docx_pages = [[("tbl", t) if k == "gbtbl" else (k, t) for k, t in blocks]
                  for blocks in pages]
    docx_out = OUT_DIR / f"{OUT_STEM}.docx"
    build_docx(docx_pages, docx_out)
    _assert_no_tex_residue(docx_out)
    print(f"已生成 → {docx_out}（行内数学残留校验通过）")

    print(f"W_w 档切点：{[(r['t'], r['a_max'], r['b_max']) for r in lim['ww_rows']]}")
    print(f"U 切点：A≥{lim['u_a_min']}、B≥{lim['u_b_min']}"
          f"（依据档 n={lim['u_bands_n']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

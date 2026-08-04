"""《26 块带厚度标注样片检测报告》生成器（开发壳，不随核心包交付）。

只读 data/derived/sample26_thickness/values.json（make_sample26_assets.py 产物），
渲染管线复用 make_uniformity_gb_doc.render_pages（PDF）与 make_docx_versions（Word）。
章节：方法口径 / 名册与 mm/px 标定 / 逐片明细 / 厚度分档与趋势 / 特种片对照 /
未钢化零点与 ε / 检测异常处置 / refs 初标草案 / 局限清单。
用法：venv python fringe_scoring/make_sample26_report_doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fringe_scoring.make_uniformity_gb_doc import render_pages  # noqa: E402

DATE = "2026-08-04"
VERSION = "V1.0"
VAL = ROOT / "data" / "derived" / "sample26_thickness" / "values.json"

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _vals() -> dict:
    """读资产 json（图文同源）。"""
    with open(VAL, "r", encoding="utf-8") as f:
        return json.load(f)


def _detail_row(r: dict) -> str:
    """逐片明细一行。"""
    i = r["indicators"]
    m = r["mm_per_px"]
    prov = {"auto": "自动", "manual": "人工", "fullframe": "整图"}.get(
        r["detection"]["provenance"], r["detection"]["provenance"])
    qa = {"pass": "", "warn": " (warn)", "fail": " (FAIL)"}[r["qa"]["level"]]
    return (f"{r['sample_id']:6s}　{r['thickness_mm'] if r['thickness_mm'] else '—':>4}　"
            f"{m['mean']:.4f}　{i['x095']:>5.0f}　{i['gray_variance']:>7.1f}　"
            f"{i['gradient_mean']:>5.2f}　{i['texture_w']:.4f}　"
            f"{i['position_score']:>5.1f}　{i['total_score']:>6.2f}　{prov}{qa}")


def report_pages() -> list[list[tuple[str, str]]]:
    """全部页面（手工分页）。"""
    V = _vals()
    photos = V["photos"]
    core = [r for r in photos if r["category"] == "core"]
    special = [r for r in photos if r["category"] == "special"]
    control = [r for r in photos if r["category"] == "control"]
    eps = V["epsilon"]
    rep = V["repeatability"]
    mppx_all = [r["mm_per_px"]["mean"] for r in photos if r["mm_per_px"]]
    aniso_all = [r["mm_per_px"]["aniso"] for r in photos if r["mm_per_px"]]

    def band_row(b):
        """厚度档统计表一行：厚度/n/成员/W_w 均值与极差/总分均值。"""
        return (f"{b['thickness_mm']:>4}mm　n={b['n']}　{'/'.join(b['members']):24s}　"
                f"{b['texture_w']['mean']:.4f}　[{b['texture_w']['min']:.4f}, "
                f"{b['texture_w']['max']:.4f}]　{b['total_score']['mean']:>6.2f}")

    return [
        # ── P1 封面 + 方法口径 ──
        [
            ("space", "0.06"),
            ("h1", "26 块带厚度标注样片 · 检测报告"),
            ("cnote", f"版本 {VERSION}　|　{DATE}　|　六指标（v1.12 texture_w 版）"),
            ("space", "0.02"),
            ("h2", "一、样批与方法"),
            ("body",
             f"样批：{V['n_photos']} 张照片、26 块样片（1#/16# 中空各两张；规格表原件即缺\n"
             "5#/9#/17# 编号）。厚度 3.2~15mm，尺寸 610×610 或 510×360mm，含两块\n"
             "未钢化对照（8#-1/8#-2）与中空/夹层/超白/Low-E/压花特种片。\n"
             "\n"
             "方法：GlassApp v1.12 六指标（X0.95/灰度方差/梯度均值/梯度方差/纹理指数\n"
             "W_w/位置评分），逐张由已知物理尺寸反解 mm/px 并注入 1px/mm 重采样\n"
             "（仅影响 W_w）；评估域=整片矫正图（W_w）/扣免罚边框带内部（其余五项）。\n"
             "检出失败或错检的照片以人工角点（剖面法定界+目检交叉验证）补测，\n"
             "来源在明细表如实标注；判定层默认停用，本报告只出分数不下结论。"),
            ("space", "0.008"),
            ("note",
             "三条前置披露：① 本批照片为 ~650×630 导出缩图（非相机原始分辨率），\n"
             "mm/px≈1.0 是导出缩放的产物，原始光学分辨率已丢失；② 全批曝光一致性未知，\n"
             "绝对灰度类指标（X0.95/灰度方差/梯度两项）的跨片可比性以此为前提；\n"
             "③ 本批无人工质量评分——所有统计只反映指标行为，不构成质量判级。"),
            ("space", "0.010"),
            ("h2", "二、mm/px 标定（已知尺寸反解）"),
            ("body",
             f"可反解的 {len(mppx_all)} 张：mm/px ∈ [{min(mppx_all):.4f}, {max(mppx_all):.4f}]，"
             f"中位 {sorted(mppx_all)[len(mppx_all)//2]:.4f}；\n"
             f"两轴各向异性（长宽比偏差）最大 {max(aniso_all):.1%}（QA 容差：>5% warn、\n"
             ">10% fail）。同一样片两张照片的标定重复性：\n"
             + "\n".join(
                 f"　{sid}：Δmm/px = {d['d_mm_per_px']}，ΔW_w = {d['d_texture_w']}，"
                 f"Δ总分 = {d['d_total']}" for sid, d in rep.items())
             + "\n——单拍 W_w 的重复性差 ≈0.02~0.04，与相邻厚度档的档均值差同量级，\n"
             "解读分档结论时须带上这个不确定度。"),
        ],
        # ── P2 逐片明细 ──
        [
            ("space", "0.04"),
            ("h2", "三、逐片明细（编号序）"),
            ("tbl",
             "编号　　厚度mm　mm/px　X0.95　灰度方差　梯度均　 W_w　　位置分　 总分　角点来源\n"
             + "\n".join(_detail_row(r) for r in photos)),
            ("space", "0.006"),
            ("note",
             "X0.95 为灰度域代理值（gray→nm 未标定）；W_w 保 4 位小数（报数精度要求）；\n"
             "角点来源：自动=检测器、人工=剖面法定界（13#/1-2/16-1/16-2/8#-1/8#-2 共 6 张，\n"
             "见第八章）。总分为默认平权方案，仅作批内排序参考。"),
        ],
        # ── P3 厚度分档 ──
        [
            ("space", "0.04"),
            ("h2", "四、厚度分档与趋势（纯普白钢化）"),
            ("body",
             f"入围 {V['eligible_n']} 片（category=core 且 QA 通过；排除 "
             f"{len(V['excluded_from_stats'])} 张：特种/对照/双照片重复），按公称厚度分档：\n"),
            ("tbl",
             "厚度　　 n　 成员　　　　　　　　　　　 W_w均值　W_w [min, max]　　　总分均值\n"
             + "\n".join(band_row(b) for b in V["bands"])),
            ("space", "0.008"),
            ("body",
             f"跨档趋势：Spearman(W_w, 厚度) = **{V['spearman_w_vs_thickness']:+.4f}**"
             f"（并列平均秩，n={V['eligible_n']}）——\n"
             "**W_w 随厚度显著上升**，「厚片天生斑重」首次获得受控样本支撑。\n"
             "两点注意：① 6mm 档均值（0.2499）高于 8mm 档（0.2380），在单拍重复性\n"
             "±0.02~0.04 的量级内属可容忍倒挂，不宜过度解读；② 15mm 仅 1 片\n"
             "（W_w=0.3047，全批钢化片最高），单点不入档、只作趋势参考。"),
            ("space", "0.006"),
            ("img", f"{V['scatter']}::0.30"),
            ("cnote", "图 4-1　W_w 与公称厚度：实心=普白钢化（分档），方框=均质/高应力 n=1，"
                      "三角=特种片，菱形=未钢化对照"),
        ],
        # ── P4 特种片对照 ──
        [
            ("space", "0.04"),
            ("h2", "五、特种片单列（不入分档统计）"),
            ("tbl",
             "编号　　类型　　　　　　　　　　　　　　W_w　　位置分　总分\n"
             + "\n".join(
                 f"{r['sample_id']:6s}　{(r['type'] or '')[:22]:22s}　"
                 f"{r['indicators']['texture_w']:.4f}　{r['indicators']['position_score']:>5.1f}"
                 f"　{r['indicators']['total_score']:>6.2f}"
                 for r in special)),
            ("space", "0.008"),
            ("body",
             "定性读法（与 6mm 普白档均值 W_w=0.2499 对照）：\n"
             "· 3# 超白 0.2103、19# Low-E 0.1873 落在普白 6mm 分布下沿附近；18# Low-E\n"
             "　 0.2902 偏高——Low-E 两片同厚同膜差 0.10，膜面方向/批次差异可疑，n=2 不下结论；\n"
             "· 14# 高应力防火 0.3494 为全批最高，与「应力刻意偏高」的工艺预期一致；\n"
             "· 中空（1#/16#）与夹层（2#）为双片光路叠加，数值不可与单片直接比较；\n"
             "· 13# 压花 0.1267：**落在未钢化对照（0.13）水平**——规格表未注明该片\n"
             "　 是否钢化，且压花几何光路复杂（周期平滑纹理在 min/max 量化下低对比），\n"
             "　 两种解释无法区分，仅作记录不下结论。"),
        ],
        # ── P5 零点与 ε ──
        [
            ("space", "0.04"),
            ("h2", "六、未钢化对照：指标零点验证"),
            ("tbl",
             "编号　　 W_w　　位置分　X0.95　总分　　复核门\n"
             + "\n".join(
                 f"{r['sample_id']:6s}　{r['indicators']['texture_w']:.4f}　"
                 f"{r['indicators']['position_score']:>5.1f}　{r['indicators']['x095']:>5.0f}　"
                 f"{r['indicators']['total_score']:>6.2f}　"
                 f"{'触发后放行' if r['indicators']['verification']['triggered'] else '未触发'}"
                 for r in control)),
            ("space", "0.008"),
            ("body",
             "三条结论：\n"
             "① **自动检测检不出未钢化片**（暗场下无应力斑=不发亮）——这本身就是\n"
             "　 最强的零点证据：体系的「信号」确实来自钢化应力，本报告以人工角点补测；\n"
             "② 补测后两片 W_w=0.1315/0.1367，为全批最低（低于一切钢化片），\n"
             "　 位置评分 92.7/92.0、总分 97.5/95.0——体系把未钢化片判为「近满分好片」，\n"
             "　 复核门触发亦放行。**这是体系语义而非缺陷**：六指标测应力斑轻重、\n"
             "　 **不测钢化身份**——现场不得把高分当作「已钢化且合格」的凭据；\n"
             "③ ε 条款（最小动态范围）候选：官方判据（整片 max−min）实测**无区分力**\n"
             f"　 ——对照片 DR={eps['controls_dr'][0]:.0f}/{eps['controls_dr'][1]:.0f} 与钢化片下界 "
             f"{eps['tempered_dr_min']:.0f} 同域（片上标签纸与边缘辉光把 max 拉满）；\n"
             f"　 内缩 5% 口径才分离：对照 {eps['controls_dr_inset5'][0]:.1f}/"
             f"{eps['controls_dr_inset5'][1]:.1f} vs 钢化下界 {eps['tempered_dr_inset5_min']:.1f}"
             "（隔离带 2.4 倍）。\n"
             "　 ε 落地需 (a) 无标签样本重测或 (b) 判据改内缩域（工程决策），\n"
             "　 且须产线原生分辨率复核——**定值仍 TODO(plant)**。"),
        ],
        # ── P6 L2 跨仪器对照 ──
        [
            ("space", "0.04"),
            ("h2", "七、第三方对照：Softsolution L2 线扫（北玻送测）"),
            ("body",
             "同批样片另经行业标准各向异性扫描仪 Softsolution LineScanner（L2）检测\n"
             "（2026-06-30，报告见 data/images/北玻 2026. 8.4/），给出 nm 域光程差分位。\n"
             "对应表区分「同一块玻璃」（直接可比）与「同一炉仅供参考」（批级代理）："),
            ("space", "0.006"),
            ("tbl",
             "编号　　匹配级别　　L2 95%(nm)　L2 98%(nm)　本报告 X0.95(灰度)　W_w\n"
             + "\n".join(
                 f"{r['sample_id']:6s}　{'同一块' if r['match'] == 'same_sheet' else '同炉参考'}"
                 f"　　{r['q95_nm']:>6}　　　{r['q98_nm']:>6}　　　{r['x095_gray']:>8.1f}"
                 f"　　　{r['texture_w']:.4f}"
                 for r in V["l2_cross"]["rows"])),
            ("space", "0.008"),
            ("body",
             f"秩一致性（并列平均秩，n={len(V['l2_cross']['rows'])}）：\n"
             f"　X0.95（灰度域） vs L2 95% 分位（nm 域）：**ρ = "
             f"{V['l2_cross']['spearman_x095_vs_q95']:+.4f}**\n"
             f"　W_w vs L2 95% 分位：ρ = {V['l2_cross']['spearman_texture_w_vs_q95']:+.4f}\n"
             "——两套系统相隔一月、不同成像、不同评估域（L2 扣角 75mm/扣边 25mm），\n"
             "秩一致性仍达 0.91+：灰度代理口径的排序有效性获得跨仪器支撑。\n"
             "三条彼此独立的印证：\n"
             "① **13# 压花：L2 亦「测量失败」**——与本报告的检测失败互为佐证，压花片\n"
             "　 超出两套系统的适用域；\n"
             "② 8#-1 未钢化 L2=35nm 为全表最低，钢化片 45~116nm——零点排序在 nm 域成立\n"
             "　 （35nm 亦提示未钢化基线非零，与本报告「近满分而非满分」一致）；\n"
             "③ 18# Low-E 的 98% 分位=307nm（95%=84nm）——重尾异常，与本报告 18#/19#\n"
             "　 同规格差 0.10 的可疑观察方向一致，该片建议复测。\n"
             "如实声明：以上为**秩一致性对照，不构成 gray→nm 标定**（B8 仍缺）——\n"
             "标定需同域同期成像与线性段核查。"),
        ],
        # ── P7 异常处置 + 草案 ──
        [
            ("space", "0.04"),
            ("h2", "八、检测异常与处置（6 张人工角点）"),
            ("body",
             "· 8#-1/8#-2（未钢化）：检测失败（前景为空）——机理见第六章①；剖面法\n"
             "　 （行/列亮度剖面最陡沿）定界，与目检一致，aniso 2.0%/0.8%；\n"
             "· 13#（压花）：错检成 2 片（压花亮纹被当独立连通域）——剖面法定界后\n"
             "　 aniso 1.3%；\n"
             "· 1-2/16-1/16-2（框装中空）：暗框内亮玻璃区错检/漏检——取亮区包围盒，\n"
             "　 aniso 0.7%/3.5%/2.6%。\n"
             "六张全部通过几何 QA 后入册；分档统计不受影响（均非 core 类）。\n"
             "唯一例外披露：8# 两张为 core 厚度（4mm）但 category=control，永不入统计。"),
            ("space", "0.012"),
            ("h2", "九、refs_by_thickness 初标草案"),
            ("body",
             "4/5/6/8mm 四档（n=2~5），best/worst=档内批内极值；完整可粘贴片段见\n"
             "data/derived/sample26_thickness/refs_by_thickness_draft.yaml。要点：\n"
             "· **批内极值≠好/差样片标定**：本批无人工分，best/worst 只是「本批最好/最差」，\n"
             "　 发档使用前须同厚度好/废片人工复标；\n"
             "· 粘贴位置：app.plans.<方案名>.refs_by_thickness（顶层 indicators 段会被\n"
             "　 apply_active_plan 覆盖，别粘那里）；\n"
             "· 选档口径：max_thickness_mm ≥ 实测厚度中取最小档；\n"
             "　 **厚度超出全部档（如 10/12/15mm）静默回退默认 refs，不是回退最大档**；\n"
             "· n=1 厚度（10/12/15mm）不出档：单点无区间，TODO(plant) 待补样。"),
        ],
        # ── P7 局限 ──
        [
            ("space", "0.04"),
            ("h2", "十、局限清单（如实列示）"),
            ("body",
             "① 无人工质量评分：一切统计只反映指标行为，非质量判级依据；\n"
             "② 样本量：核心档 n=2~5、三个厚度 n=1，档间差异与单拍重复性（ΔW_w\n"
             "　 0.02~0.04）同量级——分档 refs 是初值不是刻度；\n"
             "③ 导出缩图：非原始分辨率，1px/mm 重采样近似恒等但光学细节已丢，\n"
             "　 W_w 的 d=1≡1mm 口径此处指「导出后像素」；\n"
             "④ 曝光一致性未知：绝对灰度类指标跨片可比性存疑（W_w 仿射不变不受此限，\n"
             "　 但 8-bit 饱和裁剪属非仿射污染，本批未逐片核查饱和占比）；\n"
             "⑤ 人工角点 6 张：定界误差直接进 mm/px 与评估域（aniso ≤3.5% 可控）；\n"
             "⑥ 特种片结论全部定性：中空/夹层双片光路、压花几何纹理均使 W_w 语义\n"
             "　 偏离单片钢化应力斑，不得横向比较；\n"
             "⑦ ε 候选窗口依赖内缩口径与本批成像条件，定值须产线原生数据。"),
            ("space", "0.010"),
            ("note",
             f"报告版本 {VERSION}（{DATE}）。数值资产 data/derived/sample26_thickness/\n"
             "values.json（make_sample26_assets.py 复算落盘，图文同源）；名册与人工角点\n"
             "manifest.yaml；照片按仓库规则不入库。"),
        ],
    ]


def main() -> int:
    """出 PDF + Word。"""
    pages = report_pages()
    pdf_out = ROOT / "docs" / "样片26_厚度分档三合一检测报告.pdf"
    with PdfPages(pdf_out) as pdf:
        render_pages(pdf, pages)
    print(f"已生成 → {pdf_out}")
    from fringe_scoring.make_docx_versions import _assert_no_tex_residue, build_docx
    docx_out = pdf_out.with_suffix(".docx")
    build_docx(pages, docx_out)
    _assert_no_tex_residue(docx_out)
    print(f"已生成 → {docx_out}")
    print("行内数学残留校验：通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""公式说明与工厂使用手册 PDF 生成器（开发壳，不随核心包交付）。

产出两份文档到 docs/（不含现场照片，可入版本库）：
- 应力斑测量_公式说明.pdf —— 模块用到的全部公式（面向技术对接方）；
- 应力斑测量_使用手册.pdf —— 工厂操作口径（工业语言，零 AI 术语）。
内容与代码同源维护：公式/键名改了，改本文件重跑即出新版。
用法：venv python fringe_scoring/make_docs_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无窗口环境直接落盘
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # Windows 中文字体
plt.rcParams["axes.unicode_minus"] = False

PAGE = (8.27, 11.69)  # A4 纵向（英寸）
DATE = "2026-07-03"

# 版式块类型 → (字号, 行距/占位)。Block = (kind, text)
_STYLES = {
    "h1": (17, 0.050),
    "h2": (13, 0.036),
    "h3": (11, 0.030),
    "body": (9.5, 0.0215),
    "formula": (12, 0.040),
    "note": (8.5, 0.019),
}


def render_pages(pdf: PdfPages, pages: list[list[tuple[str, str]]]) -> None:
    """把分页好的内容块渲染进 PDF：每页从上往下排版（内容手工分页，不自动断页）。"""
    for blocks in pages:
        fig = plt.figure(figsize=PAGE)
        y = 0.94
        for kind, text in blocks:
            if kind == "space":
                y -= float(text)
                continue
            size, step = _STYLES[kind]
            if kind == "h1":
                fig.text(0.5, y, text, fontsize=size, ha="center", fontweight="bold")
                y -= step
            elif kind in ("h2", "h3"):
                fig.text(0.07, y, text, fontsize=size, fontweight="bold")
                y -= step
            elif kind == "formula":  # mathtext 公式：居中，公式内不放中文
                fig.text(0.5, y, text, fontsize=size, ha="center")
                y -= step
            else:  # body / note：逐行
                for line in text.split("\n"):
                    fig.text(0.07, y, line, fontsize=size,
                             color="dimgray" if kind == "note" else "black")
                    y -= step
        pdf.savefig(fig)
        plt.close(fig)


def formula_doc_pages() -> list[list[tuple[str, str]]]:
    """《应力斑测量_公式说明》全部页面内容。"""
    return [
        # ── P1 封面 + 符号约定 ──
        [
            ("space", "0.06"),
            ("h1", "应力斑测量 · 公式说明"),
            ("body", f"版本日期：{DATE}　|　适用：fringe_scoring 模块（嵌入式测量算法）"),
            ("space", "0.02"),
            ("h2", "0. 符号与约定"),
            ("body",
             "I(x, y)：矫正后单片灰度图像，x=列、y=行；8 位图像取值 0–255。\n"
             "M：统计域掩膜（矫正片内部，扣除四边细圈边框带），|M| 为其像素数。\n"
             "median(·) / Q_q(·)：中位数 / q 分位数；var(·)：方差；clip(v,a,b)：钳制到 [a,b]。\n"
             "MAD：中位绝对偏差；稳健尺度 $\\hat{\\sigma} = 1.4826 \\times MAD$（正态一致性系数）。\n"
             "所有可调参数在配置（config）中，右侧括号里给出键名；程序每次调用时读取。"),
            ("space", "0.015"),
            ("h2", "1. 整床多片切分"),
            ("body", "一张照片含多块玻璃。裁掉照片最外圈条带（edge_trim_frac）后："),
            ("formula",
             r"$T_{fg} = B + \max(\,z_{min}\cdot\hat{\sigma}_{dark},"
             r"\ r_{min}\cdot(P_{bright} - B)\,)$"),
            ("body",
             "其中 B = 全图低分位灰度（bg_quantile，背景水平）；$\\hat{\\sigma}_{dark}$ = 暗侧稳健尺度；\n"
             "P_bright = 亮参考分位（bright_ref_quantile）；z_min=fg_min_z，r_min=fg_min_rel。\n"
             "前景 = I > T_fg → 闭运算连补断点（close_frac）→ 外轮廓连通域 →\n"
             "面积 ≥ min_area_frac 且最小边长 ≥ min_side_frac 的连通域 → 凸包拟合 4 角。"),
            ("space", "0.015"),
            ("h2", "2. 透视矫正"),
            ("body",
             "每片 4 角点（左上→右上→右下→左下）与目标矩形做透视变换（单应矩阵），\n"
             "目标宽/高 = 对边平均长度（保像素尺度）。矫正后按矩形域计算全部指标。"),
        ],
        # ── P2 背景估计 + 斑分割 ──
        [
            ("space", "0.04"),
            ("h2", "3. 背景照度估计（暗地板锚）"),
            ("body",
             "把矫正片内部分块（块边长 ≈ 内部短边 × background_block_frac），每块取低分位："),
            ("formula", r"$z_k = Q_{q_b}(\mathrm{block}_k), \quad q_b = \mathrm{background\_block\_quantile}$"),
            ("body",
             "以块中心为样本点，用 Tukey 双平方权重的迭代重加权最小二乘（IRLS）\n"
             "拟合低次多项式曲面（背景次数 background_poly_degree，默认平面）："),
            ("formula",
             r"$B(x,y) = \sum_{i+j \leq d} c_{ij}\, u^i v^j, \quad"
             r" w_k = (1 - t_k^2)^2 \cdot \mathbf{1}[|t_k|<1], \quad"
             r" t_k = \frac{r_k}{4.685\,\hat{\sigma}}$"),
            ("body",
             "残差 R(x,y) = I(x,y) − B(x,y)。低分位取块统计的意义：应力斑只会比背景亮\n"
             "（暗场偏光成像），块内必有暗隙，背景因此锚定在暗地板上——即使整片布满亮斑，\n"
             "背景也不会被抬高（否则会漏罚）。"),
            ("space", "0.015"),
            ("h2", "4. 应力斑分割（二值掩膜）"),
            ("body", "灰度域偏离（只算偏亮，polarity=bright）："),
            ("formula", r"$D(x,y) = \max(\,R(x,y) - Q_{q_b}(R|_M),\ 0\,)$"),
            ("body", "斑掩膜（gray_threshold 法，面积不设上限）："),
            ("formula",
             r"$\mathrm{mask} = \{\,(x,y) \in M:\ D(x,y) \geq d_{min}\,\},"
             r" \quad d_{min} = \mathrm{min\_dev\_gray}$"),
            ("body",
             "掩膜即软件输出的二值化图（白=判为应力斑）。深浅强度（绝对灰度锚）："),
            ("formula",
             r"$s(x,y) = \mathrm{clip}\left(\frac{D(x,y)}{s_{sat}},\ 0,\ 1\right),"
             r" \quad s_{sat} = \mathrm{s\_saturation\_gray}$"),
            ("space", "0.015"),
            ("h2", "5. 边框带处理"),
            ("body",
             "四边各自扫描 |预残差| 剖面自适应确定边框重应力带宽度；玻璃边框带物理上必然\n"
             "存在，其中\"细细一圈\"（≤ normal_band_frac × 边长）免罚；超出允许宽度的带段\n"
             "视为质量缺陷，按普通斑计罚。背景/残差估计始终使用实测全带宽。"),
        ],
        # ── P3 位置权重 + 均匀度 ──
        [
            ("space", "0.04"),
            ("h2", "6. 位置权重与罚分"),
            ("body", "像素坐标归一到 [−1,1]²，中心距离取棋盘距离（同心四边形等罚线）："),
            ("formula", r"$r = \max(|u|, |v|), \quad w(r) = \max(1 - r,\ w_{floor})$"),
            ("body",
             "w_floor = weight.floor：边缘权重下限（边缘重斑不免罚，中心>边缘次序不变）。\n"
             "罚分与罚分比："),
            ("formula", r"$\rho = \frac{\sum_{mask} s(x,y)\, w(r)}{\sum_{M} w(r)}$"),
            ("body", "分布评分（0–100，越高越好；Z = penalty_ratio_at_zero 为人工标定刻度锚）："),
            ("formula",
             r"$\mathrm{score} = \mathrm{clip}\left(100 \times"
             r" \left(1 - \frac{\rho}{Z}\right),\ 0,\ 100\right)$"),
            ("space", "0.015"),
            ("h2", "7. 均匀度指标（六指标之一，与深浅解耦）"),
            ("body",
             "均匀度只衡量斑纹的空间分布，不受整体深浅影响：同一图案整体加深，均匀度不变；\n"
             "全白/无纹理玻璃 = 100。实现：把第 4 步的深浅强度换成逐片相对口径——\n"
             "偏离除以本片自身稳健尺度（z 分数）再进第 6 步罚分（专属刻度锚\n"
             "uniformity_ratio_at_zero）。该口径对 I → a·I + b 的整体明暗变化严格不变。"),
            ("formula",
             r"$z(x,y) = \frac{R - \mathrm{median}(R|_M)}{1.4826 \cdot \mathrm{MAD}(R|_M)},"
             r" \quad s_u = \mathrm{clip}(|z| / z_{sat},\ 0,\ 1)$"),
        ],
        # ── P4 六指标 ──
        [
            ("space", "0.04"),
            ("h2", "8. 深浅类五指标（统计域均为 M）"),
            ("h3", "8.1 95% 分位光程差 X0.95"),
            ("formula", r"$X_{0.95} = Q_{0.95}(I|_M) \times k_{nm}$"),
            ("body",
             "k_nm = gray_to_nm（灰度→光程差 nm 标定系数）。未标定时 k_nm=1，输出为灰度\n"
             "代理值（单位标注 gray）：同一台设备上可比，跨设备不可比。"),
            ("h3", "8.2 灰度方差"),
            ("formula", r"$V = \mathrm{var}(I|_M)$"),
            ("h3", "8.3 / 8.4 梯度均值与梯度方差"),
            ("formula",
             r"$G = \sqrt{G_x^2 + G_y^2}, \quad G_x = \mathrm{Sobel}_x(I),"
             r"\ G_y = \mathrm{Sobel}_y(I) \ (3\times3)$"),
            ("formula", r"$\mu_G = \mathrm{mean}(G|_M), \quad V_G = \mathrm{var}(G|_M)$"),
            ("h3", "8.5 组合纹理特征 CCP"),
            ("body",
             "在内部矩形上计算灰度共生矩阵 P(i,j)：灰度线性量化为 Ng=8 级，步距 1 像素，\n"
             "0°/45°/90°/135° 四方向、对称、归一。mm_per_px 标定后先重采样到 1 px/mm\n"
             "（跨尺寸/设备可比；未标定按原始像素，仅同设备可比）。"),
            ("formula", r"$C_a = \frac{1}{4}\sum_{\theta}\sum_{i,j} P_\theta(i,j)\,(i-j)^2$"),
            ("formula", r"$CP_a = \frac{1}{4}\sum_{\theta}\sum_{i,j} P_\theta(i,j)\,(i+j-\mu_x-\mu_y)^4$"),
            ("formula",
             r"$CCP = \frac{1}{2}\left(\sqrt{\frac{C_a}{C_{max}}} +"
             r" \left(\frac{CP_a}{CP_{max}}\right)^{1/4}\right)$"),
            ("note",
             "Cmax=ccp_c_max、CPmax=ccp_cp_max 应取自参考最差样品的现场标定；当前配置值为\n"
             "开发期批内实测初值（ccp_reference_is_plant_calibrated=false），标定后替换。"),
        ],
        # ── P5 子分映射 + 加权总分 + 参数表 ──
        [
            ("space", "0.04"),
            ("h2", "9. 子分映射与加权总分"),
            ("body",
             "深浅类五指标（越小越好）经好/差参考值线性映射为 0–100 子分（越高越好）：\n"
             "best/worst 在 config 的 indicators.refs 中逐指标给出，工厂可按自家口径调整。"),
            ("formula",
             r"$s_k = \mathrm{clip}\left(100 \times"
             r" \frac{worst_k - v_k}{worst_k - best_k},\ 0,\ 100\right)$"),
            ("body", "均匀度已是 0–100（越高越好），直接作为子分。加权总分："),
            ("formula",
             r"$\mathrm{Total} = \frac{\sum_k w_k\, s_k}{\sum_k w_k},"
             r" \quad k \in \{X_{0.95},\ V,\ \mu_G,\ V_G,\ CCP,\ U\}$"),
            ("body", "权重 w_k = indicators.weights（非负、总和为正）；默认全 1（平权）。"),
            ("space", "0.015"),
            ("h2", "10. 公式符号与配置键对照"),
            ("body",
             "T_fg 阈值项…………… sheets.fg_min_z / fg_min_rel / bg_quantile / bright_ref_quantile\n"
             "背景块分位 q_b ……… segment.background_block_quantile（暗地板锚，默认 0.15）\n"
             "斑判定门槛 d_min …… segment.min_dev_gray；深浅饱和 s_sat = segment.s_saturation_gray\n"
             "边框免罚宽度 ………… border.normal_band_frac；权重下限 w_floor = weight.floor\n"
             "刻度锚 Z ……………… scoring.penalty_ratio_at_zero；均匀度锚 indicators.uniformity_ratio_at_zero\n"
             "灰度→nm 标定 k_nm … indicators.calibration.gray_to_nm；空间标定 mm_per_px 同段\n"
             "CCP 参考 ……………… indicators.calibration.ccp_c_max / ccp_cp_max\n"
             "子分参考 ……………… indicators.refs.<指标>.best / .worst；权重 indicators.weights.<指标>"),
            ("note",
             "本文档与代码同源维护；任何公式或键名变更以最新版代码为准（fringe_scoring 模块，\n"
             f"生成日期 {DATE}）。"),
        ],
    ]


def manual_doc_pages() -> list[list[tuple[str, str]]]:
    """《应力斑测量_使用手册》全部页面内容（工业语言，不用 AI 术语）。"""
    return [
        # ── P1 封面 + 概述 ──
        [
            ("space", "0.06"),
            ("h1", "应力斑测量软件 · 使用手册"),
            ("body", f"版本日期：{DATE}　|　适用对象：钢化线质检 / 工艺人员"),
            ("space", "0.02"),
            ("h2", "这套软件是做什么的"),
            ("body",
             "对着检测台拍一张整床玻璃的照片，软件自动完成三件事：\n"
             "  1) 把照片里的每一块玻璃找出来、摆正；\n"
             "  2) 对每块玻璃量出 6 项应力斑指标，并合成一个 0–100 的总分（越高越好）；\n"
             "  3) 给出每块玻璃的\"应力斑分布图\"（黑白对照图，白色=判为应力斑的部位）。\n"
             "所有判定门槛、指标权重都写在一份配置文件里，工厂可以自己调，改完立即生效，\n"
             "不需要改程序。"),
            ("space", "0.02"),
            ("h2", "拍照要求（不满足会直接影响结果）"),
            ("body",
             "· 暗场偏光成像：没有玻璃的地方是暗背景，应力越强的部位越亮；\n"
             "· 一床照片里玻璃总面积别超过画面 3/4（背景至少留 1/4）；\n"
             "· 同一批对比的照片必须用同样的曝光/光圈/增益（本批为 400US/F4）；\n"
             "· 玻璃之间留缝，别叠放；玻璃为凸四边形（可斜放，软件会自动摆正）。"),
        ],
        # ── P2 六个指标 ──
        [
            ("space", "0.04"),
            ("h2", "六个指标各是什么"),
            ("body",
             "1) 95% 分位光程差（X0.95）——把这块玻璃所有点的亮度从低到高排队，取第 95%\n"
             "   位置的值。代表\"这块玻璃比较亮的部位有多亮\"，即应力斑深浅的整体水平。\n"
             "   数值越小越好。\n"
             "\n"
             "2) 灰度方差——亮度波动的整体幅度。斑越深、明暗反差越大，方差越大。越小越好。\n"
             "\n"
             "3) 梯度均值——相邻部位亮度变化的平均剧烈程度。斑纹越密越锐利，数值越大。\n"
             "   越小越好。\n"
             "\n"
             "4) 梯度方差——亮度变化剧烈程度的不均衡度。局部有突兀的亮线/亮点时会升高。\n"
             "   越小越好。\n"
             "\n"
             "5) CCP（组合纹理特征）——按国标草案口径衡量斑纹的纹理粗糙程度。越小越好。\n"
             "\n"
             "6) 均匀度——只看斑纹摆放得匀不匀，不管深浅：斑纹铺得均匀、不扎堆在\n"
             "   中央 → 分高；集中在中央一团 → 分低。一块全白但均匀的玻璃，均匀度是 100\n"
             "   （它的问题会体现在前五项上）。越大越好。"),
            ("note",
             "为什么要把\"深浅\"和\"均匀\"分开：一块整体都深的玻璃摆放上是均匀的——它的\n"
             "毛病由前五项扣分；一块只有中央一条斑的玻璃深浅不重——它的毛病由均匀度扣分。\n"
             "两类问题分开量，工厂可以按自家侧重加权。"),
        ],
        # ── P3 总分与调权 ──
        [
            ("space", "0.04"),
            ("h2", "总分怎么算"),
            ("body",
             "每项指标先换算成 0–100 的\"子分\"（都是越高越好）：配置里给每项设了一个\n"
             "\"好参考值\"（best，得 100 分）和一个\"差参考值\"（worst，得 0 分），实测值\n"
             "落在两者之间按比例给分。然后按权重加权平均得到总分：\n"
             "\n"
             "    总分 = (w1×子分1 + w2×子分2 + … + w6×子分6) ÷ (w1+…+w6)\n"
             "\n"
             "当前六项权重都是 1（平权）。"),
            ("space", "0.02"),
            ("h2", "怎么按自家口径调（改配置文件，立即生效）"),
            ("body",
             "① 改权重——配置文件 indicators.weights 段。比如更看重均匀度：\n"
             "       uniformity: 1.0   改成   uniformity: 2.0\n"
             "   某项不想参与总分就把它设为 0（六项不能全为 0）。\n"
             "\n"
             "② 改好/差参考值——indicators.refs 段。比如觉得灰度方差超过 3000 就该 0 分：\n"
             "       gray_variance: {best: 50.0, worst: 4000.0}\n"
             "       改成 gray_variance: {best: 50.0, worst: 3000.0}\n"
             "   当前参考值来自 2026-07 首批 25 块玻璃实测（好片/废片的典型值），\n"
             "   仅是初始口径，建议积累自家数据后再调。\n"
             "\n"
             "③ 改判废门槛——总分多少算不及格由工厂定；首批人工比对的判废线是 50 分。"),
        ],
        # ── P4 分布图 + 分数怎么读 ──
        [
            ("space", "0.04"),
            ("h2", "应力斑分布图（黑白对照图）怎么看"),
            ("body",
             "每块玻璃软件会输出一张与玻璃等大的黑白图：\n"
             "   · 白色部位 = 软件判为\"应力斑\"的区域（亮度明显高出背景的部位）；\n"
             "   · 黑色部位 = 正常区域；\n"
             "   · 玻璃四边\"细细一圈\"的边缘应力带是工艺上必然存在的，只要不超过允许宽度\n"
             "     （默认为边长的 4%）就不参与扣分；超宽的部分按缺陷计。\n"
             "用途：核对软件判的斑和人眼看的是否一致。如果白色区域明显跟人眼对不上，\n"
             "先检查拍照条件是否符合第 1 页要求，再考虑调整判定门槛（min_dev_gray）。"),
            ("space", "0.02"),
            ("h2", "分数怎么读"),
            ("body",
             "· 总分是同一套配置下的横向排序工具：分数排序 = 玻璃优劣排序；\n"
             "· 改了配置（权重/参考值/门槛）后分数不能和改之前的直接比——请在报告上注明\n"
             "  配置版本；\n"
             "· 首批实测参考：人工判废的玻璃总分 15–43，正常玻璃 69–92，最好的 88–92。\n"
             "· 软件算不了时会明确报错（比如照片里找不到玻璃、参数填错），\n"
             "  宁可不给分也不会给一个猜的分。"),
        ],
        # ── P5 标定清单 ──
        [
            ("space", "0.04"),
            ("h2", "标定清单（哪些情况要重新标定）"),
            ("body",
             "以下数值和拍照条件绑定。换相机、换曝光、换光源、换检测台后必须重标，\n"
             "否则分数与之前不可比：\n"
             "\n"
             "① 深浅饱和值（s_saturation_gray，当前 200）与斑判定门槛（min_dev_gray，当前 10）\n"
             "   ——按新条件下最深斑/纯背景的亮度重新确定；\n"
             "② 六项指标的好/差参考值（indicators.refs）——用新条件下的好片/废片重测；\n"
             "③ CCP 参考值（ccp_c_max / ccp_cp_max）——正式做法是用\"参考最差样品\"标定，\n"
             "   当前为首批玻璃的临时值（配置里有明确标记）。\n"
             "\n"
             "另有两项一次性标定，完成后填入配置即自动切换（未标定前软件用替代口径并标注）：\n"
             "④ 灰度→光程差换算系数（gray_to_nm）：标定后 X0.95 直接以 nm 输出；\n"
             "   未标定时 X0.95 是亮度代理值（同一台设备上可比，跨设备不可比）；\n"
             "⑤ 图像比例尺（mm_per_px，每像素多少毫米）：标定后 CCP 按国标口径（1 像素=1 毫米\n"
             "   重采样）计算，跨设备可比；\n"
             "⑥ 玻璃厚度（thickness_mm）：当前未知；将来做国标判级（A/B/C）时必填。"),
        ],
        # ── P6 常见问题 ──
        [
            ("space", "0.04"),
            ("h2", "常见问题"),
            ("body",
             "问：软件报错\"未检出任何玻璃片\"？\n"
             "答：多半是拍照条件问题——背景不够暗、玻璃太满、或曝光变了。对照第 1 页检查。\n"
             "\n"
             "问：一床里有块玻璃没被框出来？\n"
             "答：检查它是否与相邻玻璃贴得太近（留缝）、是否只露出一部分在画面里。\n"
             "\n"
             "问：分数突然整体变了？\n"
             "答：先查配置文件是否被改过（权重/参考值/门槛任何一项变了分数都会变），\n"
             "    再查拍照条件是否变了（曝光/光源老化）。\n"
             "\n"
             "问：黑白分布图里玻璃边上一圈全是白的？\n"
             "答：边缘应力带是正常的。只有这一圈超过允许宽度（默认边长 4%）时才计入扣分；\n"
             "    如果人眼认为边太宽也没问题，可在配置里调大 normal_band_frac。\n"
             "\n"
             "问：两台检测台的分数能直接比吗？\n"
             "答：完成第 5 页的 ④⑤ 标定之前不能比；标定后 X0.95（nm）与 CCP 可跨台比。\n"
             "\n"
             "问：想要\"只按深浅\"或\"只按均匀\"的分？\n"
             "答：把其它指标权重设 0 即可（见第 3 页）。子分本身也逐项输出，可直接取用。"),
            ("space", "0.02"),
            ("note", f"手册与软件版本配套（{DATE}）。配置键的完整对照表见《应力斑测量_公式说明》第 10 节。"),
        ],
    ]


def main() -> int:
    """生成两份 PDF 到 docs/。"""
    out_dir = ROOT / "docs"
    formula_path = out_dir / "应力斑测量_公式说明.pdf"
    manual_path = out_dir / "应力斑测量_使用手册.pdf"
    with PdfPages(formula_path) as pdf:
        render_pages(pdf, formula_doc_pages())
    with PdfPages(manual_path) as pdf:
        render_pages(pdf, manual_doc_pages())
    print(f"已生成 → {formula_path}")
    print(f"已生成 → {manual_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""A4 中文文档的排版引擎：估宽 / 折行 / 避头尾 / 行内加粗 / 自动分页。

从 make_shifu_questions_pdf.py 拆出——那份文件长到 `tools/review.py` 的模型通道
（本地 3B，n_ctx=8192）装不下、连着四次跳过审查。拆开后两份都能审。

内容与版式分离：本模块只管「怎么摆」，一个字的内容都不含；具体文档各自维护自己的
DATE/COVER/题库来源，用 Doc 往下顺排即可。

踩过的坑都固化在这里，改之前先看清楚：
  - 估宽用**字体实测前进宽度**，不是「ASCII 一律 0.5」（微软雅黑数字 0.586、`%` 0.89）
  - 加粗另收窄 6%：粗体实际比字号宽，不留余量会顶出右边界
  - PdfPages 必须在第一次写之前打开：分页发生在写的过程中，不能等最后才开
  - 中文避头尾：收尾标点不许落在行首
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无窗口环境直接落盘
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PAGE = (8.27, 11.69)          # A4 纵向（英寸）
LEFT, RIGHT = 0.075, 0.945    # figure 坐标下的左右边界
TOP, BOTTOM = 0.945, 0.058    # 正文上下边界（下边界之下留页码）

# 块类型 → (字号, 行距)。行距按 figure 高度比例，越大的字给越松的行距。
_STYLES: dict[str, tuple[float, float]] = {
    "title": (19.0, 0.052),
    "sub": (10.0, 0.026),
    "sect": (13.5, 0.040),    # 大类标题
    "intro": (9.5, 0.023),    # 大类开场白
    "q": (11.0, 0.028),       # 问题正文
    "why": (8.3, 0.020),      # 「为什么问」小字
    "opt": (9.8, 0.026),      # 勾选项
    "hint": (8.6, 0.021),     # 答案形态提示
    "body": (9.8, 0.024),
    "note": (8.6, 0.021),
}

GRAY = "dimgray"
RULE = "#b0b0b0"

# 避头尾：这些字符不许出现在一行的开头（中文排版基本规矩）
_NO_LINE_START = set("、。，．：；？！）〕］｝』」》〉〙〗”’%‰℃…·-—")


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _load_advances() -> dict[int, float]:
    """从字体文件读逐字符前进宽度（单位 em）；读不到就返回空表、退回粗估。

    为什么不能用「ASCII 一律 0.5」：微软雅黑实测数字 **0.586**、大写字母均值 0.667、
    `%` 0.89、`w` 0.937 —— 一行里十来个数字就够把行顶出右边界
    （2026-08-27 实测：含 "2026-08-27" 的一条注解越界 4.4pt）。CJK 恒为 1.0，那部分本来就准。
    """
    try:
        from fontTools.ttLib import TTFont
        from matplotlib import font_manager

        path = font_manager.findfont(
            font_manager.FontProperties(family=plt.rcParams["font.sans-serif"][0]))
        tt = TTFont(path, fontNumber=0)
        upm = tt["head"].unitsPerEm
        cmap: dict[int, str] = {}
        for table in tt["cmap"].tables:
            cmap.update(table.cmap)
        metrics = tt["hmtx"].metrics
        return {cp: metrics[g][0] / upm for cp, g in cmap.items() if g in metrics}
    except Exception:      # 字体缺失/格式意外都不该让生成失败，降级即可
        return {}


_ADVANCES = _load_advances()


def _disp_width(s: str) -> float:
    """显示宽度（em）：优先字体实测前进宽度，取不到才退回 CJK 1.0 / ASCII 0.5 粗估。"""
    w = 0.0
    for ch in s:
        adv = _ADVANCES.get(ord(ch))
        w += adv if adv is not None else (1.0 if ord(ch) > 0x2E7F else 0.5)
    return w


def _mark_bold(text: str) -> list[tuple[str, bool]]:
    """`**…**` → 逐字 (字符, 是否加粗)；标记本身不进结果（不能印到纸上）。

    逐字而非逐段，是为了让折行能在加粗段中间断开——按段折行会让一整个加粗
    短语挤不下时整段掉到下一行，留一截空白。
    """
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in _BOLD_RE.finditer(text):
        out += [(c, False) for c in text[pos:m.start()]]
        out += [(c, True) for c in m.group(1)]
        pos = m.end()
    out += [(c, False) for c in text[pos:]]
    return out


def _wrap(text: str, fontsize: float, indent: float = 0.0,
          right_pad: float = 0.0, bold: bool = False) -> list[list[tuple[str, bool]]]:
    """按可用宽度折行，返回逐行的 (字符, 是否加粗) 序列（CJK 友好，不依赖空格）。

    可用宽度换算：一个 CJK 字宽 ≈ fontsize/72 英寸；页面可用英寸 = 图宽 × (右-左-缩进-右留白)。
    `right_pad` 给右上角优先级标签让位，`bold` 收窄 6%（粗体实际比字号宽，不留余量会顶出边界）。
    显式 \\n 优先断行。
    """
    usable_in = PAGE[0] * (RIGHT - LEFT - indent - right_pad)
    per_char_in = fontsize / 72.0
    limit = max(8.0, usable_in / per_char_in * (0.94 if bold else 0.985))

    out: list[list[tuple[str, bool]]] = []
    for para in text.split("\n"):
        if not para:
            out.append([])
            continue
        cur: list[tuple[str, bool]] = []
        w = 0.0
        for ch, bd in _mark_bold(para):
            cw = _disp_width(ch)
            if w + cw > limit and cur:
                # 避头尾：新行不能以收尾标点开头（「…平常设多少 / 、最赶的时候…」很难看）。
                # 把上一行末字挪下来陪它——除非上一行只剩一个字，那样挪了会空行。
                if ch in _NO_LINE_START and len(cur) > 1:
                    out.append(cur[:-1])
                    cur = [cur[-1], (ch, bd)]
                    w = _disp_width(cur[0][0]) + cw
                else:
                    out.append(cur)
                    cur, w = [(ch, bd)], cw
            else:
                cur.append((ch, bd))
                w += cw
        if cur:
            out.append(cur)
    return out


class Doc:
    """自动分页的顺排文档（上下文管理器）。

    分页发生在写内容的过程中（一页排满就 savefig 翻页），因此 PdfPages 必须在
    第一次写之前就打开——用 with 进出，别在最后才开。
    """

    def __init__(self, path: Path, footer: str) -> None:
        """建文档对象；footer 是每页左下角的小字（版本/用途标识）。"""
        self.path = path
        self.pdf: PdfPages | None = None
        self.fig = None
        self.y = TOP
        self.page_no = 0
        self.footer = footer

    def __enter__(self) -> Doc:
        """开 PdfPages 并建父目录——必须早于任何写入（分页时就要 savefig）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf = PdfPages(self.path)
        return self

    def __exit__(self, *exc: object) -> None:
        """收尾最后一页并关闭 PdfPages（异常时也保证句柄释放）。"""
        self._close_page()
        if self.pdf is not None:
            self.pdf.close()

    # ---------------- 分页 ----------------
    def _new_page(self) -> None:
        self._close_page()
        self.fig = plt.figure(figsize=PAGE)
        self.page_no += 1
        self.y = TOP

    def _close_page(self) -> None:
        """给当前页补页脚与页码后落盘；无当前页则空操作。"""
        if self.fig is None:
            return
        self.fig.text(LEFT, 0.030, self.footer, fontsize=7.5, color=GRAY)
        self.fig.text(RIGHT, 0.030, f"第 {self.page_no} 页", fontsize=7.5,
                      color=GRAY, ha="right")
        assert self.pdf is not None
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def _need(self, height: float) -> None:
        """确保剩余空间够放 height；不够就翻页（避免题目被腰斩）。"""
        if self.fig is None or self.y - height < BOTTOM:
            self._new_page()

    # ---------------- 画元素 ----------------
    def _lines(self, text: str, kind: str, *, indent: float = 0.0,
               color: str = "black", bold: bool = False, right_pad: float = 0.0) -> None:
        size, step = _STYLES[kind]
        lines = _wrap(text, size, indent, right_pad, bold)
        self._need(step * len(lines))
        for ln in lines:
            assert self.fig is not None
            self._draw_rich(LEFT + indent, ln, size, color, bold)
            self.y -= step

    def _draw_rich(self, x: float, line: list[tuple[str, bool]],
                   size: float, color: str, base_bold: bool) -> None:
        """画一行，行内 `**…**` 段加粗。

        matplotlib 一次 text() 只能一个字重，故按字重分段落笔：每段画完量它的
        实际像素宽，换算成 figure 坐标作为下一段起点（估算宽度会累积漂移）。
        """
        assert self.fig is not None
        if not line:
            return
        segs: list[tuple[str, bool]] = []          # 合并相邻同字重的字符
        for ch, bd in line:
            if segs and segs[-1][1] == bd:
                segs[-1] = (segs[-1][0] + ch, bd)
            else:
                segs.append((ch, bd))
        if len(segs) == 1:                          # 常见情况：整行同字重，一次画完
            self.fig.text(x, self.y, segs[0][0], fontsize=size, color=color,
                          fontweight="bold" if (base_bold or segs[0][1]) else "normal")
            return
        renderer = self.fig.canvas.get_renderer()
        fig_px = self.fig.get_figwidth() * self.fig.dpi
        cx = x
        for seg, bd in segs:
            t = self.fig.text(cx, self.y, seg, fontsize=size, color=color,
                              fontweight="bold" if (base_bold or bd) else "normal")
            cx += t.get_window_extent(renderer).width / fig_px

    def gap(self, h: float) -> None:
        """纵向留白（页首不留，避免翻页后顶部空一截）。"""
        if self.fig is not None and self.y < TOP - 1e-9:
            self.y -= h

    def title_page(self, title: str, subtitle: str, blocks: list[tuple[str, str]]) -> None:
        """封面：大标题 + 副标题 + 若干 (小标题, 正文) 段。"""
        self._new_page()
        self.y = 0.86
        assert self.fig is not None
        self.fig.text(0.5, self.y, title, fontsize=_STYLES["title"][0],
                      ha="center", fontweight="bold")
        self.y -= _STYLES["title"][1]
        self.fig.text(0.5, self.y, subtitle, fontsize=_STYLES["sub"][0],
                      ha="center", color=GRAY)
        self.y -= _STYLES["sub"][1] * 2.0
        for head, body in blocks:
            self._lines(head, "sect", bold=True)
            self._lines(body, "body")
            self.gap(0.014)

    def section(self, title: str, intro: str, first_item: dict | None = None) -> None:
        """大类标题：连同开场白与**首题实际高度**一起估高。

        传 first_item 才准：早先用固定 0.10 估首题，遇到选项多的长题（如第七节第 1 题
        有 9 个勾选项）会让标题孤零零留在页尾、下面空掉半页。
        """
        size, step = _STYLES["sect"]
        need = step + _STYLES["intro"][1] * len(_wrap(intro, _STYLES["intro"][0]))
        need += self._q_height(first_item) if first_item else 0.10
        need += 0.024          # 本方法自己要调的两处 gap（0.016 前 + 0.008 后），漏算就差这点把首题挤走
        self._need(need)
        self.gap(0.016)
        assert self.fig is not None
        self.fig.text(LEFT, self.y, title, fontsize=size, fontweight="bold")
        self.y -= step * 0.55
        self.fig.add_artist(plt.Line2D([LEFT, RIGHT], [self.y, self.y],
                                       color="black", linewidth=1.1))
        self.y -= step * 0.45
        if intro:
            self._lines(intro, "intro", color=GRAY)
        self.gap(0.008)

    def _q_height(self, item: dict, num: str = "00.") -> float:
        """一道题占的纵向高度（含题干折行、小字、勾选项、填写横线）。

        question() 与 section() 共用：前者据此保证整题不跨页，后者据此避免
        标题与首题被拆到两页。
        """
        q_size, q_step = _STYLES["q"]
        pad = 0.062 if item.get("prio") else 0.0
        n_q = len(_wrap(f"{num}　{item['q']}", q_size, 0.0, pad, bold=True))
        why, hint = item.get("why", ""), item.get("hint", "")
        n_why = len(_wrap(why, _STYLES["why"][0], indent=0.022)) if why else 0
        n_opt = sum(len(_wrap(o, _STYLES["opt"][0], indent=0.045))
                    for o in item.get("options", []))
        return (q_step * n_q + _STYLES["why"][1] * n_why
                + _STYLES["hint"][1] * (1 if hint else 0) + _STYLES["opt"][1] * n_opt
                + 0.026 * item.get("lines", 0) + 0.022)

    def question(self, num: str, prio: str, text: str, why: str,
                 hint: str, options: list[str], write_lines: int) -> None:
        """一道题：编号+优先级 / 题干 / 为什么问 / 勾选项 / 填写横线。

        整题估高后一次性 _need，保证不会跨页断开（师傅在纸上翻页找选项很烦）。
        """
        head = f"{num}　{text}"
        pad = 0.062 if prio else 0.0     # 给右上角 [优先级] 让位，避免题干撞上去
        self._need(self._q_height(
            {"q": text, "why": why, "hint": hint, "options": options,
             "lines": write_lines, "prio": prio}, num))

        assert self.fig is not None
        if prio:
            self.fig.text(RIGHT, self.y, f"[{prio}]", fontsize=8.0,
                          color=GRAY, ha="right")
        self._lines(head, "q", bold=True, right_pad=pad)
        if why:
            self._lines(f"— {why}", "why", indent=0.022, color=GRAY)
        if hint:
            self._lines(f"（{hint}）", "hint", indent=0.022, color=GRAY)
        self.gap(0.004)
        for opt in options:
            self._lines(f"□  {opt}", "opt", indent=0.045)
        for _ in range(write_lines):
            self.y -= 0.020
            self._need(0.006)
            assert self.fig is not None
            self.fig.add_artist(plt.Line2D([LEFT + 0.030, RIGHT], [self.y, self.y],
                                           color=RULE, linewidth=0.7))
            self.y -= 0.006
        self.gap(0.014)

    def table(self, headers: list[str], rows: list[str]) -> None:
        """空白填写表：首行是表头，每行左侧印行名、右侧留格子给师傅填。

        用于第 1 题这类「一档一档填数字」的题——纸上有格子，师傅才填得下去。
        """
        n_col = len(headers)
        row_h, head_h = 0.026, 0.030
        self._need(head_h + row_h * len(rows) + 0.012)
        x0, x1 = LEFT + 0.030, RIGHT
        name_w = 0.150                                   # 左侧行名列宽
        cell_w = (x1 - x0 - name_w) / max(1, n_col - 1)

        assert self.fig is not None
        for i, h in enumerate(headers):
            x = x0 + (0.0 if i == 0 else name_w + cell_w * (i - 1) + cell_w * 0.5)
            self.fig.text(x, self.y, h, fontsize=8.6, fontweight="bold",
                          ha="left" if i == 0 else "center")
        self.y -= head_h * 0.42
        self.fig.add_artist(plt.Line2D([x0, x1], [self.y, self.y],
                                       color="black", linewidth=0.9))
        self.y -= head_h * 0.58

        for r in rows:
            self.fig.text(x0, self.y, r, fontsize=9.0)
            base = self.y - 0.006
            for c in range(n_col - 1):                   # 每个数据格画一条底线
                cx0 = x0 + name_w + cell_w * c + 0.008
                self.fig.add_artist(plt.Line2D([cx0, cx0 + cell_w - 0.016],
                                               [base, base], color=RULE, linewidth=0.7))
            self.y -= row_h
        self.gap(0.010)

    def record_page(self, rows: list[str]) -> None:
        """末页：访谈记录栏（谁答的、哪台炉、什么时候——答案要能溯源）。"""
        self._new_page()
        self.section("访谈记录（回收时填）", "这一页是给记录人填的，方便日后追溯每条答案的来源。")
        for label in rows:
            self._lines(label, "body")
            self.y -= 0.014
            assert self.fig is not None
            self.fig.add_artist(plt.Line2D([LEFT + 0.030, RIGHT], [self.y, self.y],
                                           color=RULE, linewidth=0.7))
            self.y -= 0.022



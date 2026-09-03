"""《问师傅清单》PDF 生成器：可打印、带勾选框与填写栏的现场调研问卷。

用途：把当前卡住开发的工艺真值缺口（信息需求清单 A/B/D/E 组未闭环项）与三端 UI
合理性疑问，整理成老师傅能在车间当场答完的纸质清单。答案回收后按每题标注的
「去向」填进对应 config 或文档，并把信息需求清单的状态改 ✅。

题库真源 = docs/问师傅清单_题库.json，**运行时直接读**，本文件不留副本。
改题目就改那份 JSON 再重跑本脚本即可；没有中间生成步骤，也就不存在副本漂移。
版式引擎在 tools/pdf_layout.py（估宽/折行/分页），本文件只管内容。

用法：& "D:\\Glass Agents\\.venv\\Scripts\\python.exe" tools\\make_shifu_questions_pdf.py
产物：docs/问师傅清单_<日期>.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 支持直接 `python tools/make_shifu_questions_pdf.py`

from tools.pdf_layout import Doc  # noqa: E402

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


DATE = "2026-08-27"

COVER = [
    ("这份清单是干什么的",
     "钢化炉调参系统里有一批数值和判断，只有天天上炉的师傅知道。没有这些，系统只能一直\n"
     "保守地拦着不放行，或者只能靠猜——我们不猜。另一半问题是问界面好不好用：哪些字段\n"
     "其实每炉都不动、哪些数看不清、哪一步最烦，我们照着改。"),
    ("怎么答",
     "有方框的直接打勾，可以多选；带横线的写几个字或说给记录人写。**答不上来的直接跳过**，\n"
     "空着比编一个数有用得多——写错的数会被系统当成真的学进去。\n"
     "凭印象给的估计值，请在旁边标一下「大概」。"),
    ("大概要多久",
     "整份约一小时五十分。带 [阻塞] 的最急，时间不够就先答这些，[重要] 的可以留到下次。\n"
     "**第零节请务必先问完**——那是上一轮答案的追问，上次拿到的是「一般都怎么做」，\n"
     "这次要的是「你这台炉现在是多少」。只有一次时间就做第零、一、二节。"),
]

RECORD_ROWS = [
    "受访师傅（工号或姓名）",
    "工龄 / 主要负责的炉号",
    "受访日期与时长",
    "记录人",
    "现场补充（记录人填：当场看到的现象、师傅提到但清单没问的事）",
]


# ==================================================================================== #
# 内容区：题库真源 = docs/问师傅清单_题库.json，**运行时读**，本文件不留副本
# 早先这里放的是 SECTIONS/BRIEFING 字面量（由 JSON 生成）。副本会漂——2026-08-23 有一次
# 只在 .py 里修了字形，重跑一次就被覆盖回去；54 题的字面量还把文件顶过 review.py 的上下文上限。
# ==================================================================================== #
BANK = ROOT / "docs" / "问师傅清单_题库.json"


def load_bank() -> tuple[list[dict], list[str]]:
    """读题库 JSON → (sections, briefing) 两份排版用数据。

    JSON 用录题时顺手的键名（question/why/answer_form/priority），这里转成排版用的短键。
    `lines` = 填写横线数：有勾选项的题勾完就行给 1 行，开放题给 3 行。
    """
    import json

    data = json.loads(BANK.read_text(encoding="utf-8"))
    sections: list[dict] = []
    for sec in data["sections"]:
        items = []
        for it in sec["items"]:
            opts = list(it.get("options") or [])
            items.append({
                "q": it["question"],
                "why": it["why"],
                "hint": it.get("answer_form", ""),
                "options": opts,
                "prio": it["priority"],
                "ref": it.get("registry_ref", ""),
                "lines": 1 if opts else 3,
            })
        sections.append({"title": sec["title"], "intro": sec["intro"], "items": items})
    return sections, list(data["notes"])


# 给记录人的说明（师傅不用看这页）——来自定稿阶段的分流意见

# 第 1 题（厚度→加热时长）单独给一张填写表，纸上有格子师傅才填得下去
THICKNESS_TABLE = (
    ["厚度", "平常设多少秒", "最赶能压到", "最慢拖到"],
    ["4 mm", "5 mm", "6 mm", "8 mm", "10 mm", "12 mm", "15 mm", "其他：____"],
)

# 上表挂在哪一题下面：认题干里这句话，不认题号（插节/调序都不会跟丢）
_TABLE_ANCHOR = "加热时间各设多少秒"


def build(out_path: Path) -> tuple[Path, int, int]:
    """生成 PDF；返回 (路径, 题目总数, 页数)。"""
    sections, briefing = load_bank()
    n_q = 0
    with Doc(out_path, f"问师傅清单 · {DATE} · 钢化炉调参系统") as d:
        d.title_page("问 师 傅 清 单", f"钢化炉调参系统 · 现场调研 · {DATE}", COVER)

        d._new_page()
        d.section("给记录人的话（这页师傅不用看）",
                  "出发前先看一遍：哪些题不该问师傅、哪些必须站在设备旁边问、时间不够先问哪几节。")
        for note in briefing:
            d._lines(note, "note", color="black")
            d.gap(0.010)

        # 厚度—秒数表按**题干内容**锚定，不按题号：2026-08-27 在最前面插了第零节，
        # 原来的 `n_q == 1` 就把表挂到了新第 1 题（问「上次那五个答案是谁答的」）头上。
        anchored = [it for sec in sections for it in sec["items"] if _TABLE_ANCHOR in it["q"]]
        if len(anchored) != 1:
            raise SystemExit(f"厚度—秒数表的锚点命中 {len(anchored)} 题，应为 1——题干改过就同步改 _TABLE_ANCHOR")

        for sec in sections:
            # 传首题让 section 按真实高度估：不传会让选项多的长题把标题孤立在页尾
            d.section(sec["title"], sec["intro"], sec["items"][0] if sec["items"] else None)
            for item in sec["items"]:
                n_q += 1
                tag = f"{item['prio']}"
                with_table = _TABLE_ANCHOR in item["q"]
                lines = 0 if with_table else item["lines"]   # 下面接填写表就不用横线
                d.question(f"{n_q}.", tag, item["q"], item["why"],
                           item["hint"], item["options"], lines)
                if with_table:
                    d.table(*THICKNESS_TABLE)

        d.record_page(RECORD_ROWS)
        pages = d.page_no
    return out_path, n_q, pages


def main() -> int:
    """生成《问师傅清单》PDF 到 docs/。"""
    out = ROOT / "docs" / f"问师傅清单_{DATE}.pdf"
    path, n_q, pages = build(out)
    print(f"已生成：{path}")
    print(f"题目 {n_q} 道，共 {pages} 页")
    return 0


if __name__ == "__main__":
    sys.exit(main())

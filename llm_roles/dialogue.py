"""CLI 对话壳：多轮指正参数 / 呈现样片指标 / 工艺问答的会话层（LLM 只做语言壳）。

设计（规则 > AI）：
- **确定性关键词路由为主**（词表 config/dialogue_rules.yaml，数据驱动可现场补词），
  LLM 只兜底做单标签意图分类（temperature=0，非枚举输出一律按 unknown）；
- **数字提取绝不交给 LLM**：改参数的数值由正则从用户原话确定性抽取，歧义→反问不猜；
- 多轮记忆放 DialogueState（确定性状态），不放 LLM 上下文；
- 改参后强制 tools.constraints.validate(new, prev=旧参数)——单次调温幅度(C1-d)靠 prev 生效；
  越界只呈现 violations（复用 param_translator，不给"照做"建议）；
- 诊断只**呈现**数据中已带的指标与分数（含外部 fringe 分），诊断类目未拍板（TODO(plant)）
  ——只报数，不下"哪坏了"的结论。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from llm_roles._llm import chat
from llm_roles.kb_qa import DEFAULT_KB_PATH, retrieve
from llm_roles.param_translator import render_skeleton, translate_params
from schemas.archive import ArchiveSample, MetricRecord
from schemas.process_params import ProcessParams
from tools.constraints import CheckResult, validate

_ROOT = Path(__file__).resolve().parents[1]
_RULES_PATH = _ROOT / "config" / "dialogue_rules.yaml"
_FALLBACK_PROMPT_PATH = _ROOT / "config" / "prompt_dialogue_fallback.md"

Intent = Literal["param_edit", "param_check", "process_qa", "diagnose", "show_state", "help", "exit", "unknown"]
_INTENTS: tuple[Intent, ...] = (
    "param_edit", "param_check", "process_qa", "diagnose", "show_state", "help", "exit", "unknown",
)

_RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# 别名命中后，在其后多长的窗口内找数值（覆盖"改到 705"这类近邻表达）
_NUMBER_WINDOW = 20

_HELP_TEXT = """我能做的（数值全部来自确定性规则，不由大模型生成）：
- 改参数：如「上炉温改到705」→ 改后自动过安全闸门，越界会如实列出问题
- 查参数：「检查当前参数」「当前参数」
- 看样片：「诊断一下」→ 呈现已加载样片的指标与应力斑分布分（只报数；诊断类目 TODO(plant)）
- 问工艺：「为什么上炉温要高于下炉温」→ 查知识库，缺依据会拒答
- 退出：「退出」"""


@dataclass
class DialogueState:
    """会话状态 = 确定性内存（多轮记忆放这里，不放 LLM 上下文）。"""

    furnace_id: str = "unknown"
    params: ProcessParams | None = None              # 当前讨论中的参数组
    prev_params: ProcessParams | None = None         # 上一轮参数（C1-d 单次调温幅度检查的 prev）
    last_check: CheckResult | None = None
    metrics: MetricRecord | None = None              # 已加载样片的指标（外部评好的分随数据到达）
    turns: list[tuple[str, str]] = field(default_factory=list)   # (用户, 助手) 审计留痕


def load_dialogue_rules(path: Path | None = None) -> dict:
    """读路由词表（每次调用按需读取，现场补词即时生效）。"""
    p = Path(path) if path is not None else _RULES_PATH
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def route_intent(text_zh: str, rules: dict | None = None) -> Intent:
    """确定性关键词路由：按词表顺序首个命中即定；无命中 → unknown（交兜底/反问）。"""
    r = rules if rules is not None else load_dialogue_rules()
    text = text_zh.strip().lower()
    for intent, keywords in (r.get("intents") or {}).items():
        if intent not in _INTENTS:
            continue
        for kw in keywords or []:
            if str(kw).lower() in text:
                return intent  # type: ignore[return-value]
    return "unknown"


def classify_with_llm(text_zh: str, llm: Any) -> Intent:
    """LLM 兜底分类（temperature=0）：输出限定意图枚举词，非枚举一律按 unknown（不猜）。"""
    template = _FALLBACK_PROMPT_PATH.read_text(encoding="utf-8")
    out = chat(llm, template.replace("{utterance}", text_zh)).strip().lower()
    return out if out in _INTENTS else "unknown"  # type: ignore[return-value]


def parse_param_updates(text_zh: str, rules: dict | None = None) -> tuple[dict[str, float], list[str]]:
    """从用户原话确定性抽取「字段→新值」；读不到数值的字段进反问列表，绝不猜。

    例：「上炉温改到705」→ ({"temp_upper": 705.0}, [])；「把风速调大点」→ ({}, ["风速 要改到多少？…"])。
    """
    r = rules if rules is not None else load_dialogue_rules()
    aliases: dict[str, str] = r.get("field_aliases") or {}
    updates: dict[str, float] = {}
    questions: list[str] = []
    matched_fields: set[str] = set()

    for alias, field_name in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        idx = text_zh.find(alias)
        if idx < 0 or field_name in matched_fields:
            continue
        window = text_zh[idx + len(alias) : idx + len(alias) + _NUMBER_WINDOW]
        m = _RE_NUMBER.search(window)
        if m:
            updates[field_name] = float(m.group())
        else:
            questions.append(f"「{alias}」要改到多少？这句话里没读到数值，请给出目标值（含单位更好）")
        matched_fields.add(field_name)

    if not updates and not questions:
        questions.append("没识别出要改哪个参数。支持的叫法：" + "、".join(aliases))
    return updates, questions


def _render_metrics(metrics: MetricRecord, check: CheckResult | None) -> str:
    """确定性呈现样片指标与分数（只报数；诊断类目 TODO(plant) 不下结论）。"""
    def fmt(v: object) -> str:
        """缺值显示为（缺），绝不填数。"""
        return "（缺）" if v is None else str(v)

    lines = [
        f"- X0.95: {fmt(metrics.x0_95_nm)} nm（判级 {fmt(metrics.x0_95_grade)}）",
        f"- IsoT: {fmt(metrics.iso_t_pct)} %（判级 {fmt(metrics.iso_t_grade)}）",
        f"- CCP: {fmt(metrics.ccp_value)}（判级 {fmt(metrics.ccp_grade)}；已标定 {metrics.ccp_is_calibrated}）",
        f"- 应力斑分布分(0-100，外部评定): {fmt(metrics.fringe_score_0_100)}",
    ]
    if check is not None and not check.within_limits:
        lines.append("- 当前参数未过安全闸门：" + "；".join(check.violations))
    lines.append("（以上仅为指标呈现；异常/故障/失调的诊断类目未拍板 TODO(plant)，不下结论）")
    return "\n".join(lines)


def _handle_param_edit(state: DialogueState, text: str, rules: dict, thresholds: dict | None) -> str:
    """改参：确定性抽数 → model_copy 更新 → 强制过闸门（prev=改前参数）→ 复用翻译官呈现。"""
    if state.params is None:
        return "当前会话还没有参数组。请先用 --baseline 载入基准配方，或加载样片（--sample）。"
    updates, questions = parse_param_updates(text, rules)
    if questions:
        return "\n".join(questions)

    old = state.params
    new = old.model_copy(update=updates)
    check = validate(new.to_param_set(), prev=old.to_param_set(), thresholds=thresholds)
    state.prev_params = old
    state.params = new
    state.last_check = check

    changed = "；".join(f"{k} → {v}" for k, v in updates.items())
    result = translate_params(new, check, polish=False)
    head = f"已按你说的更新：{changed}\n"
    if not check.within_limits:
        head = f"按你说的试算了：{changed}\n"
    return head + result.advice_zh


def _handle_param_check(state: DialogueState, thresholds: dict | None) -> str:
    """查参：当前参数过闸门 → 确定性骨架/violations 呈现。"""
    if state.params is None:
        return "当前会话还没有参数组，无从检查。先 --baseline 载入基准配方。"
    prev = state.prev_params.to_param_set() if state.prev_params is not None else None
    check = validate(state.params.to_param_set(), prev=prev, thresholds=thresholds)
    state.last_check = check
    return translate_params(state.params, check, polish=False).advice_zh


def _handle_process_qa(text: str, llm: Any | None, kb_path: Path) -> str:
    """工艺问答：词法检索知识库；无 LLM 时直接呈现命中问答对（缺依据/缺值一律拒答）。"""
    hits = retrieve(text, kb_path)
    usable = [h for h in hits if not h.needs_plant_value]
    if not usable:
        return "无法基于现有资料回答（检索无依据或答案依赖未填的 TODO(plant) 真值），不编造。"
    if llm is None:
        top = usable[0]
        src = "、".join(top.sources) or "（未标出处）"
        return f"知识库命中（未经 LLM 综合）：\nQ: {top.question_zh}\nA: {top.answer_zh}\n出处: {src}"
    from llm_roles.kb_qa import answer_question

    ans = answer_question(text, kb_path=kb_path, llm=llm)
    src = "、".join(ans.sources) or "（无）"
    return f"{ans.answer_zh}\n出处: {src}"


def respond(
    state: DialogueState,
    user_text: str,
    *,
    llm: Any | None = None,
    rules: dict | None = None,
    thresholds: dict | None = None,
    kb_path: Path = DEFAULT_KB_PATH,
) -> tuple[DialogueState, str]:
    """一轮对话：老状态 + 用户输入 → 新状态 + 回复。llm 注入式（None=全程确定性）。"""
    r = rules if rules is not None else load_dialogue_rules()
    intent = route_intent(user_text, r)
    if intent == "unknown" and llm is not None:
        intent = classify_with_llm(user_text, llm)

    if intent == "exit":
        reply = "已结束对话。参数与检查结果都在会话记录里，需要留档请走归档流程。"
    elif intent == "help":
        reply = _HELP_TEXT
    elif intent == "show_state":
        reply = render_skeleton(state.params) if state.params is not None else "当前会话还没有参数组。"
    elif intent == "param_edit":
        reply = _handle_param_edit(state, user_text, r, thresholds)
    elif intent == "param_check":
        reply = _handle_param_check(state, thresholds)
    elif intent == "diagnose":
        if state.metrics is None:
            reply = "当前会话没有加载样片数据（--sample 载入 ArchiveSample JSON 后可看指标与分数）。"
        else:
            reply = _render_metrics(state.metrics, state.last_check)
    elif intent == "process_qa":
        reply = _handle_process_qa(user_text, llm, kb_path)
    else:
        reply = "没听懂这句话（不猜）。可以说「帮助」看我能做什么，或换个说法。"

    state.turns.append((user_text, reply))
    return state, reply


def state_from_sample(sample: ArchiveSample) -> DialogueState:
    """从归档样本初始化会话：带炉体身份、参数（作为当前）、基准与指标。"""
    return DialogueState(
        furnace_id=sample.furnace_id,
        params=sample.params,
        prev_params=sample.baseline_params,
        metrics=sample.metrics,
    )

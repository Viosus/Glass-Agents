"""Teacher 在环原型（N1 闭环骨架）：本地 Qwen 按炉况产参 → 过硬约束闸门 → 翻译官出中文。

Teacher LLM 选型已拍板（2026-07-02，信息需求清单 D1）：Qwen2.5-14B-Instruct 4bit GPU，
经 config/llm.yaml 数据驱动（load_teacher_llm）；14B 权重未落盘时按 config 回退/报错，
产出的数字在现场标定前仍**只作草稿**——产参 → 解析 → constraints.validate → 翻译。
规则 > AI：Teacher 出参**强制过 tools.constraints.validate**；within_limits=False 原样返回
violations，**不自动修、不放宽**（安全闸门在模型之外）。

Prompt 模板：config/prompt_teacher.md（占位 {context} {baseline}）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llm_roles._llm import chat, load_teacher_llm
from llm_roles.param_translator import TranslationResult, translate_params
from schemas.process_params import ProcessParams
from tools.constraints import CheckResult, validate

_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_PATH = _ROOT / "config" / "prompt_teacher.md"
_PARAM_FIELDS = tuple(ProcessParams.model_fields.keys())


@dataclass
class TeacherSuggestion:
    """Teacher 单次建议：参数 + 理由 + 闸门结果 + 原始输出（解析失败时 params/check 为 None）。"""

    params: ProcessParams | None
    rationale: str
    check: CheckResult | None
    raw: str
    error: str | None = None


def _parse_json_object(raw: str) -> dict | None:
    """从模型输出宽松解析首个 JSON 对象（容忍围栏/前后噪声）。"""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _coerce_params(obj: dict, baseline: ProcessParams) -> ProcessParams:
    """以基准配方为底，用 obj 中的已知字段覆盖，构造 ProcessParams（缺字段沿用基准；坏类型抛错）。"""
    data: dict[str, Any] = baseline.model_dump()
    for k in _PARAM_FIELDS:
        if k in obj:
            data[k] = obj[k]
    return ProcessParams(**data)


def suggest(
    context_text: str,
    baseline: ProcessParams,
    *,
    thresholds: dict | None = None,
    llm: Any | None = None,
) -> TeacherSuggestion:
    """炉况 + 基准 → Teacher 产参 → 解析 → 过 constraints.validate（不自动修越界）。"""
    if llm is None:
        llm = load_teacher_llm()  # config/llm.yaml 选型（D1 拍板：14B 4bit GPU）
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.replace("{context}", context_text).replace("{baseline}", baseline.model_dump_json())
    raw = chat(llm, prompt, max_tokens=1024)

    obj = _parse_json_object(raw)
    if obj is None or "params" not in obj or not isinstance(obj["params"], dict):
        return TeacherSuggestion(None, "", None, raw, error="未解析出 params JSON 对象")

    rationale = str(obj.get("rationale", ""))
    try:
        params = _coerce_params(obj["params"], baseline)
    except ValidationError as exc:
        return TeacherSuggestion(None, rationale, None, raw, error=f"参数结构校验失败：{exc}")

    check = validate(params.to_param_set(), thresholds=thresholds)
    return TeacherSuggestion(params, rationale, check, raw)


def suggest_and_translate(
    context_text: str,
    baseline: ProcessParams,
    *,
    thresholds: dict | None = None,
    llm: Any | None = None,
) -> tuple[TeacherSuggestion, TranslationResult | None]:
    """完整链路：Teacher 产参 → 过闸门 → 翻译官出中文（解析失败则翻译为 None）。

    翻译用 polish=False（确定性骨架，避免链路中再调一次 LLM）；越界时翻译如实呈现 violations。
    """
    s = suggest(context_text, baseline, thresholds=thresholds, llm=llm)
    if s.params is None or s.check is None:
        return s, None
    tr = translate_params(s.params, s.check, polish=False)
    return s, tr

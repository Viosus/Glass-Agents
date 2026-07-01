"""参数翻译官：把**已过 tools.constraints.validate 闸门**的工艺参数转成中文操作建议。

定位：纯展示层。规则 > AI（铁律#3）——
- 数值由 render_skeleton **确定性**填入中文骨架（数值的唯一来源），LLM 只润色措辞；
- within_limits=False 时如实呈现 violations、不得给"照做"建议；
- LLM 润色后过**数值守卫**：输出中的数字必须 ⊆ 骨架数字，否则回退确定性骨架（绝不展示杜撰数值）。

Prompt 模板：config/prompt_param_translator.md（占位 {skeleton} {context}）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_roles._llm import chat, load_llm

if TYPE_CHECKING:
    from schemas.process_params import ProcessParams
    from tools.constraints import CheckResult

_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_PATH = _ROOT / "config" / "prompt_param_translator.md"

_RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_GLASS_ZH = {"ultra_clear": "超白", "clear": "普白"}
_MODE_ZH = {"high_quality": "高质量", "high_efficiency": "高效率"}


@dataclass
class TranslationResult:
    """翻译结果。"""

    advice_zh: str                                   # 中文操作建议（最终）
    is_actionable: bool                              # within_limits 才为 True
    used_llm: bool                                   # 是否经 LLM 润色（守卫触发/越界则 False）
    warnings: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> set[float]:
    """提取文本中所有数字并归一为 float 集合（保留 6 位小数比较）。"""
    return {round(float(m), 6) for m in _RE_NUMBER.findall(text)}


def render_skeleton(params: ProcessParams) -> str:
    """纯确定性：把参数渲染成带单位的中文骨架（数值的唯一来源），不调用任何模型。"""
    p = params
    glass = f"{_GLASS_ZH.get(p.glass_type, p.glass_type)}({p.glass_type})"
    mode = f"{_MODE_ZH.get(p.quality_mode, p.quality_mode)}({p.quality_mode})"
    temps = ", ".join(str(t) for t in p.zone_temps)
    roles = ", ".join(p.zone_roles)
    lines = [
        f"- 厚度: {p.thickness_mm} mm；品类: {glass}；质量模式: {mode}",
        f"- 分区温度(℃): [{temps}]（角色: {roles}）",
        f"- 上炉温: {p.temp_upper} ℃；下炉温: {p.temp_lower} ℃",
        f"- 对流风速: {p.convection_speed}；上下对流配比: {p.convection_ratio_upper_lower}",
        f"- 摆动速度: {p.oscillation_speed}；摆动幅度: {p.oscillation_amplitude}",
        f"- 加热时长: {p.heating_duration_s} s",
    ]
    return "\n".join(lines)


def _render_rejected(check: CheckResult) -> str:
    """within_limits=False：呈现 violations，明确禁止照做（不出操作步骤）。"""
    head = "⚠️ 该参数集未通过安全闸门，禁止照此操作。需先解决以下问题："
    items = "\n".join(f"- {v}" for v in check.violations) or "- （未提供具体 violations）"
    return f"{head}\n{items}"


def translate_params(
    params: ProcessParams,
    check: CheckResult,
    *,
    context: dict | None = None,
    polish: bool = True,
    llm: Any | None = None,
) -> TranslationResult:
    """把(已校验)参数转中文操作建议。逻辑见模块 docstring。"""
    # 未过闸门：只呈现 violations，不给照做建议
    if not check.within_limits:
        return TranslationResult(
            advice_zh=_render_rejected(check),
            is_actionable=False,
            used_llm=False,
        )

    skeleton = render_skeleton(params)
    if not polish:
        return TranslationResult(advice_zh=skeleton, is_actionable=True, used_llm=False)

    # LLM 润色 + 数值守卫
    if llm is None:
        llm = load_llm()
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    ctx_text = "（无）" if not context else "；".join(f"{k}={v}" for k, v in context.items())
    prompt = template.replace("{skeleton}", skeleton).replace("{context}", ctx_text)
    polished = chat(llm, prompt)

    if extract_numbers(polished) <= extract_numbers(skeleton):
        return TranslationResult(advice_zh=polished, is_actionable=True, used_llm=True)
    # 守卫触发：LLM 引入了骨架外数字 → 回退确定性骨架
    return TranslationResult(
        advice_zh=skeleton,
        is_actionable=True,
        used_llm=False,
        warnings=["数值守卫触发：LLM 输出含骨架外数字，已回退确定性骨架"],
    )

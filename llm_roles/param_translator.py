"""参数翻译官（接口桩，暂不实现业务逻辑）。

职责：把**已过 tools.constraints.validate 闸门**的工艺参数转成中文操作建议。
定位：纯展示层。规则 > AI（铁律#3）——
- 数值由代码确定性填入中文模板（render_skeleton），LLM 只润色措辞，绝不改/增数字；
- within_limits=False 时如实呈现 violations、不得给“照做”建议；
- 经 LLM 润色后过“数值守卫”：输出中的数字必须 ⊆ 骨架数字，否则回退确定性骨架。

Prompt 模板：config/prompt_param_translator.md（占位 {skeleton} {context}）。
本轮只定接口；实现与测试点见各函数 docstring 的 TODO。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from schemas.process_params import ProcessParams
    from tools.constraints import CheckResult


@dataclass
class TranslationResult:
    """翻译结果。"""

    advice_zh: str                                   # 中文操作建议（最终）
    is_actionable: bool                              # within_limits 才为 True
    used_llm: bool                                   # 是否经 LLM 润色（守卫触发/越界则 False）
    warnings: list[str] = field(default_factory=list)  # 回退原因、守卫命中等


def render_skeleton(params: ProcessParams) -> str:
    """纯确定性：把参数字段渲染成带单位的中文骨架（数值的唯一来源）。

    单位按 CONVENTIONS.md：温度 ℃ / 时长 s / 厚度 mm。此函数不调用任何模型。
    TODO(impl)：逐字段 f-string 拼装；后续作为数值守卫的“允许数字集合”来源。
    """
    raise NotImplementedError("render_skeleton 待实现（确定性填数）")


def translate_params(
    params: ProcessParams,
    check: CheckResult,
    *,
    context: dict | None = None,
    polish: bool = True,
    llm: Any | None = None,
) -> TranslationResult:
    """把(已校验)参数转中文操作建议。

    参数：
      params  —— 已过结构校验的工艺参数；
      check   —— tools.constraints.validate 的结果（含 within_limits/violations）；
      context —— 厚度/品类/质量模式等，仅供措辞，不参与产数；
      polish  —— False 时只返回 render_skeleton 的确定性骨架，不调 LLM；
      llm     —— 注入 llm_roles._llm 的实例；None 时按需加载。

    约定（实现时遵守）：
      - check.within_limits=False → advice_zh 呈现 violations + “未过安全闸门，禁止照此操作”，
        is_actionable=False，不产出照做步骤；
      - within_limits=True → render_skeleton →（polish 时）LLM 润色 → 数值守卫
        （输出数字 ⊄ 骨架数字则回退骨架、used_llm=False）。
    TODO(impl)：见上；测试点：输出数字集合 == 输入数字集合。
    """
    raise NotImplementedError("translate_params 待实现（展示层 + 数值守卫）")

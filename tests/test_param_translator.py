"""参数翻译官单测：确定性骨架、数值守卫、越界不给照做建议（注入假 llm，不加载 GGUF）。"""

from llm_roles.param_translator import extract_numbers, render_skeleton, translate_params
from schemas.process_params import ProcessParams
from tools.constraints import CheckResult


class FakeLLM:
    """假 llm：create_chat_completion 返回预设文本（避免 pytest 加载 2GB 模型）。"""

    def __init__(self, content: str) -> None:
        """存下预设返回文本。"""
        self.content = content

    def create_chat_completion(self, messages, temperature=0.0, max_tokens=1024):
        """返回 llama_cpp 同构的输出结构。"""
        return {"choices": [{"message": {"content": self.content}}]}


def _params() -> ProcessParams:
    """构造一组结构合法的参数。"""
    return ProcessParams(
        zone_temps=[100.0, 102.0],
        zone_roles=["center", "center"],
        temp_upper=700.0,
        temp_lower=650.0,
        convection_speed=1.5,
        convection_ratio_upper_lower=1.2,
        oscillation_speed=1.0,
        oscillation_amplitude=1.0,
        heating_duration_s=200.0,
        glass_type="clear",
        thickness_mm=6.0,
        quality_mode="high_quality",
    )


def _ok() -> CheckResult:
    """一个通过闸门的 CheckResult。"""
    return CheckResult(within_limits=True, blow_up_risk=False, gradient_ok=True, violations=[])


def test_skeleton_contains_all_param_numbers():
    """确定性骨架应包含各参数数值。"""
    sk = render_skeleton(_params())
    nums = extract_numbers(sk)
    for v in (700.0, 650.0, 1.5, 1.2, 200.0, 6.0, 100.0, 102.0):
        assert v in nums


def test_rejected_params_no_actionable_advice():
    """未过闸门：is_actionable=False，呈现 violations，不经 LLM。"""
    check = CheckResult(within_limits=False, blow_up_risk=True, gradient_ok=False, violations=["安全: 炸板规则缺失"])
    res = translate_params(_params(), check, llm=FakeLLM("不该被调用"))
    assert res.is_actionable is False
    assert res.used_llm is False
    assert "炸板规则缺失" in res.advice_zh
    assert "禁止" in res.advice_zh


def test_polish_false_returns_skeleton():
    """polish=False 时只返回确定性骨架。"""
    res = translate_params(_params(), _ok(), polish=False)
    assert res.used_llm is False
    assert res.advice_zh == render_skeleton(_params())


def test_llm_polish_number_subset_accepted():
    """LLM 输出数字 ⊆ 骨架数字 → 采用润色结果。"""
    sk = render_skeleton(_params())
    res = translate_params(_params(), _ok(), llm=FakeLLM(sk))
    assert res.used_llm is True
    assert res.advice_zh == sk


def test_numeric_guard_rejects_fabricated_number():
    """LLM 引入骨架外数字 → 数值守卫回退确定性骨架。"""
    sk = render_skeleton(_params())
    bad = sk + "\n补充：建议再加 999.9 ℃"
    res = translate_params(_params(), _ok(), llm=FakeLLM(bad))
    assert res.used_llm is False
    assert res.advice_zh == sk
    assert res.warnings

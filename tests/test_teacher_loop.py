"""Teacher 原型单测：产参解析、过闸门、越界不自动修、坏 JSON 容错、链到翻译官（注入假 llm）。"""

import json

from llm_roles.teacher_loop import suggest, suggest_and_translate
from schemas.process_params import ProcessParams

FULL_THR = {
    "gradient": {"adjacent_zone_max_delta_c": 5, "single_step_max_delta_c": 3},
    "thickness_duration": {"6": [100, 300]},
    "convection": {"clear": [1.0, 2.0]},
    "safety": {"blowup_rule": "rule_v0", "max_gradient": 50},
}


class FakeLLM:
    """假 llm：返回预设文本（避免加载 GGUF）。"""

    def __init__(self, content: str) -> None:
        """存下预设返回文本。"""
        self.content = content

    def create_chat_completion(self, messages, temperature=0.0, max_tokens=1024):
        """返回 llama_cpp 同构输出。"""
        return {"choices": [{"message": {"content": self.content}}]}


def _baseline() -> ProcessParams:
    """一组结构合法、在 FULL_THR 下可放行的基准配方。"""
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


def _teacher_json(params: dict) -> str:
    """构造 Teacher 输出的 JSON（params + rationale）。"""
    return json.dumps({"params": params, "rationale": "据炉况微调"})


def test_suggest_valid_passes_gate():
    """合法参数在 FULL_THR 下过闸门。"""
    base = _baseline()
    s = suggest("炉况: 略", base, thresholds=FULL_THR, llm=FakeLLM(_teacher_json(base.model_dump())))
    assert s.params is not None
    assert s.check is not None and s.check.within_limits is True
    assert s.rationale


def test_suggest_out_of_limit_blocked_not_fixed():
    """越界参数(上炉温<下炉温)被闸门拦下，且不自动修正。"""
    base = _baseline()
    bad = base.model_dump()
    bad["temp_upper"], bad["temp_lower"] = 600.0, 650.0
    s = suggest("炉况: 略", base, thresholds=FULL_THR, llm=FakeLLM(_teacher_json(bad)))
    assert s.params is not None
    assert s.params.temp_upper == 600.0                # 原样保留，未被修
    assert s.check is not None and s.check.within_limits is False
    assert s.check.violations


def test_suggest_malformed_json():
    """坏 JSON → params/check 为 None，error 有值。"""
    s = suggest("炉况: 略", _baseline(), thresholds=FULL_THR, llm=FakeLLM("这不是 JSON"))
    assert s.params is None
    assert s.check is None
    assert s.error


def test_chain_translates_when_actionable():
    """链路：合法参数过闸门 → 翻译官给可照做的中文建议。"""
    base = _baseline()
    s, tr = suggest_and_translate("炉况: 略", base, thresholds=FULL_THR, llm=FakeLLM(_teacher_json(base.model_dump())))
    assert s.check is not None and s.check.within_limits is True
    assert tr is not None and tr.is_actionable is True


def test_chain_not_actionable_when_blocked():
    """链路：越界 → 翻译官呈现 violations、不给照做建议。"""
    base = _baseline()
    bad = base.model_dump()
    bad["temp_upper"], bad["temp_lower"] = 600.0, 650.0
    s, tr = suggest_and_translate("炉况: 略", base, thresholds=FULL_THR, llm=FakeLLM(_teacher_json(bad)))
    assert tr is not None and tr.is_actionable is False

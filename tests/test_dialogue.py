"""对话壳单测：确定性路由/抽数反问/改参过闸门/多轮 prev 推进/诊断只报数/兜底分类。不加载 GGUF。"""

from datetime import datetime

import pytest

from llm_roles.dialogue import (
    DialogueState,
    load_dialogue_rules,
    parse_param_updates,
    respond,
    route_intent,
    state_from_sample,
)
from llm_roles.kb_qa import write_kb
from schemas.archive import ArchiveSample, MetricRecord
from schemas.kb import QAPair
from schemas.process_params import ProcessParams

FULL_THR = {
    "gradient": {"adjacent_zone_max_delta_c": 5, "single_step_max_delta_c": 3},
    "thickness_duration": {"6": [100, 300]},
    "convection": {"clear": [1.0, 2.0]},
    "safety": {"blowup_rule": "rule_v0", "max_gradient": 50},
}


def make_params(**kw) -> ProcessParams:
    base = dict(
        zone_temps=[102.0, 100.0],
        zone_roles=["center", "edge"],
        temp_upper=700.0,
        temp_lower=650.0,
        convection_speed=1.0,
        convection_ratio_upper_lower=1.0,
        oscillation_speed=1.0,
        oscillation_amplitude=1.0,
        heating_duration_s=200.0,
        glass_type="clear",
        thickness_mm=6.0,
        quality_mode="high_quality",
    )
    base.update(kw)
    return ProcessParams(**base)


class FakeLLM:
    """假 LLM：固定返回一段文本（测兜底分类，不加载 GGUF）。"""

    def __init__(self, content: str) -> None:
        self.content = content

    def create_chat_completion(self, **kwargs):
        return {"choices": [{"message": {"content": self.content}}]}


class FakeCore:
    """假参数头：固定 Δ 输出（测 model_suggest 接线，不训模型）。"""

    def __init__(self, delta: list[float]) -> None:
        self.delta = delta

    def __call__(self, x):
        import torch

        return {"param_delta": torch.tensor([self.delta], dtype=torch.float32)}


@pytest.fixture(scope="module")
def rules():
    return load_dialogue_rules()


# --------------------------- 路由 --------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("上炉温改到705", "param_edit"),
        ("检查当前参数", "param_check"),
        ("给个建议", "model_suggest"),
        ("模型怎么看", "model_suggest"),          # 含"怎么"但先于 process_qa 命中
        ("这个怎么弄", "process_qa"),             # 普通"怎么"仍走问答
        ("为什么上炉温要高于下炉温", "process_qa"),
        ("诊断一下这片玻璃", "diagnose"),
        ("当前参数", "show_state"),
        ("帮助", "help"),
        ("退出", "exit"),
        ("今天天气不错", "unknown"),          # 否定样本：域外话不猜
    ],
)
def test_route_intent(rules, text, expected):
    assert route_intent(text, rules) == expected


# --------------------------- 抽数与反问 --------------------------- #
def test_parse_updates_single_and_multi(rules):
    updates, questions = parse_param_updates("上炉温改到705", rules)
    assert updates == {"temp_upper": 705.0} and not questions

    updates2, _ = parse_param_updates("上炉温改到705，下炉温调到690", rules)
    assert updates2 == {"temp_upper": 705.0, "temp_lower": 690.0}


def test_parse_no_number_asks_back(rules):
    updates, questions = parse_param_updates("把风速加大", rules)
    assert not updates and any("要改到多少" in q for q in questions)   # 反问，不猜


def test_parse_no_alias_asks_back(rules):
    updates, questions = parse_param_updates("改到42", rules)
    assert not updates and any("没识别出" in q for q in questions)


# --------------------------- 改参 → 闸门 → 多轮 prev --------------------------- #
def test_edit_violating_gate_shows_violations_no_advice(rules):
    state = DialogueState(params=make_params())
    state, reply = respond(state, "上炉温改到640", rules=rules, thresholds=FULL_THR)  # 640 < 下炉温650
    assert "禁止照此操作" in reply and "temp_upper" in reply
    assert state.last_check is not None and not state.last_check.within_limits
    assert "加热时长" not in reply                     # 越界时不给操作骨架


def test_edit_within_gate_updates_and_renders(rules):
    state = DialogueState(params=make_params())
    state, reply = respond(state, "上炉温改到702", rules=rules, thresholds=FULL_THR)
    assert "已按你说的更新" in reply and "702" in reply
    assert state.params is not None and state.params.temp_upper == 702.0
    assert state.last_check is not None and state.last_check.within_limits


def test_multi_turn_prev_propagates(rules):
    state = DialogueState(params=make_params())
    state, _ = respond(state, "上炉温改到702", rules=rules, thresholds=FULL_THR)
    assert state.prev_params is not None and state.prev_params.temp_upper == 700.0
    state, _ = respond(state, "上炉温改到703", rules=rules, thresholds=FULL_THR)
    assert state.prev_params is not None and state.prev_params.temp_upper == 702.0  # prev 随轮推进
    assert len(state.turns) == 2


def test_edit_without_params_asks_for_baseline(rules):
    state, reply = respond(DialogueState(), "上炉温改到702", rules=rules, thresholds=FULL_THR)
    assert "没有参数组" in reply


def test_check_current_params(rules):
    state, reply = respond(DialogueState(params=make_params()), "检查当前参数", rules=rules, thresholds=FULL_THR)
    assert "700.0" in reply and "加热时长" in reply     # 确定性骨架呈现


# --------------------------- 模型建议（人在环，不自动应用） --------------------------- #
def test_model_suggest_zero_delta_renders_and_not_applied(rules):
    state = DialogueState(params=make_params())
    model = FakeCore([0.0] * 6)                     # 零 Δ → 建议 = 当前参数 → 过闸门
    state, reply = respond(state, "给个建议", rules=rules, model=model, thresholds=FULL_THR)
    assert "模型建议" in reply and "仅供参考" in reply
    assert "temp_upper +0.00" in reply              # Δ 明细逐项呈现
    assert "加热时长" in reply                       # 过闸门 → 确定性骨架
    assert state.params is not None and state.params.temp_upper == 700.0  # 未自动应用


def test_model_suggest_bad_delta_shows_violations(rules):
    state = DialogueState(params=make_params())
    model = FakeCore([-60.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # 上炉温压到 640 < 下炉温 650
    state, reply = respond(state, "给个建议", rules=rules, model=model, thresholds=FULL_THR)
    assert "禁止照此操作" in reply and "temp_upper" in reply
    assert state.params is not None and state.params.temp_upper == 700.0  # 越界建议更不会应用


def test_model_suggest_without_model(rules):
    state, reply = respond(DialogueState(params=make_params()), "给个建议", rules=rules, thresholds=FULL_THR)
    assert "无已激活模型版本" in reply


def test_model_suggest_without_params(rules):
    model = FakeCore([0.0] * 6)
    state, reply = respond(DialogueState(), "给个建议", rules=rules, model=model)
    assert "没有参数组" in reply


# --------------------------- 诊断只报数 --------------------------- #
def test_diagnose_without_sample(rules):
    state, reply = respond(DialogueState(), "诊断一下", rules=rules)
    assert "没有加载样片" in reply


def test_diagnose_presents_metrics_no_conclusion(rules):
    metrics = MetricRecord(x0_95_nm=82.0, fringe_score_0_100=76.5)
    state, reply = respond(DialogueState(metrics=metrics), "诊断一下", rules=rules)
    assert "82.0" in reply and "76.5" in reply
    assert "TODO(plant)" in reply and "不下结论" in reply   # 类目未拍板 → 只报数


# --------------------------- 工艺问答（无 LLM 全确定性） --------------------------- #
def test_qa_empty_kb_refuses(rules, tmp_path):
    state, reply = respond(DialogueState(), "为什么上炉温要高于下炉温", rules=rules, kb_path=tmp_path / "no.jsonl")
    assert "无法基于现有资料" in reply                  # 缺依据拒答，不编造


def test_qa_hits_without_llm_shows_source(rules, tmp_path):
    kb = tmp_path / "kb.jsonl"
    write_kb(
        [
            QAPair(
                question_zh="为什么上炉温要高于下炉温",
                answer_zh="C1 梯度温控要求上炉温高于下炉温（原理待 docs/03 细化）",
                sources=["docs/数据填写说明_config与约束代号.md#C1"],
                is_grounded=True,
            )
        ],
        kb,
    )
    state, reply = respond(DialogueState(), "为什么上炉温要高于下炉温", rules=rules, kb_path=kb)
    assert "C1" in reply and "出处" in reply and "未经 LLM 综合" in reply


# --------------------------- LLM 兜底分类 --------------------------- #
def test_llm_fallback_classifies_enum(rules):
    fake = FakeLLM("param_check")
    state, reply = respond(DialogueState(), "这组行不行啊", rules=rules, llm=fake, thresholds=FULL_THR)
    assert "没有参数组" in reply                        # 走了 param_check 分支


def test_llm_fallback_non_enum_treated_unknown(rules):
    fake = FakeLLM("我觉得这是在问问题")
    state, reply = respond(DialogueState(), "这组行不行啊", rules=rules, llm=fake)
    assert "没听懂" in reply                            # 非枚举输出一律 unknown，不猜


# --------------------------- 样本初始化 / 退出 --------------------------- #
def test_state_from_sample():
    sample = ArchiveSample(
        sample_id="s1",
        created_at=datetime(2026, 7, 2, 9, 0, 0),
        source="line1/早班",
        furnace_id="F1",
        thickness_mm=6.0,
        glass_type="clear",
        quality_mode="high_quality",
        params=make_params(temp_upper=705.0),
        baseline_params=make_params(),
        metrics=MetricRecord(x0_95_nm=82.0, fringe_score_0_100=88.0),
    )
    state = state_from_sample(sample)
    assert state.furnace_id == "F1"
    assert state.params is not None and state.params.temp_upper == 705.0
    assert state.prev_params is not None and state.prev_params.temp_upper == 700.0
    assert state.metrics is not None and state.metrics.fringe_score_0_100 == 88.0


def test_exit_intent(rules):
    state, reply = respond(DialogueState(), "退出", rules=rules)
    assert reply.startswith("已结束对话") and len(state.turns) == 1

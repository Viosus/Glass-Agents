"""LLM 后端单测：TransformersChatAdapter 接口契约 + config 选型分发。不加载任何真模型。"""

import pytest
import torch

from llm_roles._llm import TransformersChatAdapter, chat, load_llm_config, load_teacher_llm


class FakeTokenizer:
    """假 tokenizer：记录调用、返回定长张量、解码固定文本。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, *, add_generation_prompt, return_tensors):
        self.calls.append({"messages": messages, "agp": add_generation_prompt})
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids, *, skip_special_tokens):
        return f"回答({len(ids)}新token)  "


class FakeModel:
    """假模型：记录 generate 入参，返回 输入+2 个新 token。"""

    device = "cpu"

    def __init__(self) -> None:
        self.gen_kwargs: dict = {}

    def generate(self, inputs, **kwargs):
        self.gen_kwargs = kwargs
        return torch.cat([inputs, torch.tensor([[4, 5]])], dim=1)


def test_adapter_contract_matches_llama_cpp():
    """create_chat_completion 返回结构与 llama-cpp 同款 → chat() 零改动可用。"""
    tok, model = FakeTokenizer(), FakeModel()
    adapter = TransformersChatAdapter(model, tok)
    out = adapter.create_chat_completion(messages=[{"role": "user", "content": "你好"}], temperature=0.0)
    assert out["choices"][0]["message"]["content"].startswith("回答(2新token)")  # 只解码新生成段
    assert tok.calls[0]["agp"] is True

    text = chat(adapter, "你好")                       # 经统一入口，strip 生效
    assert text == "回答(2新token)"


def test_adapter_greedy_at_temperature_zero():
    """temperature=0 → do_sample=False 且不传 temperature（贪心，确定性任务禁随机）。"""
    model = FakeModel()
    adapter = TransformersChatAdapter(model, FakeTokenizer())
    adapter.create_chat_completion(messages=[{"role": "user", "content": "q"}], temperature=0.0)
    assert model.gen_kwargs["do_sample"] is False and "temperature" not in model.gen_kwargs

    adapter.create_chat_completion(messages=[{"role": "user", "content": "q"}], temperature=0.7)
    assert model.gen_kwargs["do_sample"] is True and model.gen_kwargs["temperature"] == 0.7


def test_load_llm_config_missing_returns_empty(tmp_path):
    assert load_llm_config(tmp_path / "不存在.yaml") == {}


def test_teacher_backend_dispatch_errors(tmp_path):
    bad = tmp_path / "llm.yaml"
    bad.write_text("teacher:\n  backend: cloud_api\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知 LLM 后端"):
        load_teacher_llm(bad)

    no_source = tmp_path / "llm2.yaml"
    no_source.write_text("teacher:\n  backend: transformers\n  local_dir: models/不存在\n", encoding="utf-8")
    with pytest.raises(ValueError, match="无法加载"):
        load_teacher_llm(no_source)                    # 本地目录缺 + 无 model_id → 明确报错不猜

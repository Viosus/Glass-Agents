"""知识库问答单测：词法检索、生成器 TODO(plant) 保守标记、缺值/无依据拒答（注入假 llm）。"""

from llm_roles.kb_qa import answer_question, generate_qa_pairs, load_kb, retrieve, write_kb
from schemas.kb import QAPair


class FakeLLM:
    """假 llm：返回预设文本（避免加载 GGUF）。"""

    def __init__(self, content: str) -> None:
        """存下预设返回文本。"""
        self.content = content

    def create_chat_completion(self, messages, temperature=0.0, max_tokens=1024):
        """返回 llama_cpp 同构输出。"""
        return {"choices": [{"message": {"content": self.content}}]}


class RaisingLLM:
    """一旦被调用即抛错的假 llm，用于断言"快速拒答通路未调用 LLM"。"""

    def create_chat_completion(self, messages, temperature=0.0, max_tokens=1024):
        """被调用即失败。"""
        raise AssertionError("不应调用 LLM")


def _kb(tmp_path):
    """在临时目录写一个含 2 条问答的知识库并返回路径。"""
    pairs = [
        QAPair(
            question_zh="相邻分区温差上限是多少",
            answer_zh="相邻分区温差不超过 5℃。",
            sources=["docs/数据填写说明.md#C1"],
            is_grounded=True,
            tags=["温差", "C1"],
        ),
        QAPair(
            question_zh="8mm 玻璃的 X0.95 A 级上限是多少",
            answer_zh="该值为 TODO(plant)，源文档暂缺。",
            sources=["docs/01#缺"],
            is_grounded=False,
            needs_plant_value=True,
            tags=["X0.95", "厚度"],
        ),
    ]
    path = tmp_path / "kb.jsonl"
    write_kb(pairs, path)
    return path


def test_write_read_roundtrip(tmp_path):
    """写库后能读回且条数一致。"""
    path = _kb(tmp_path)
    assert len(load_kb(path)) == 2


def test_retrieve_ranks_relevant_pair(tmp_path):
    """按关键词检索命中相关问答，无关问题返回空。"""
    path = _kb(tmp_path)
    hits = retrieve("相邻分区温差", path)
    assert hits and "温差" in hits[0].question_zh
    assert retrieve("苹果香蕉批发", path) == []       # 与库无共享字符 → 空


def test_answer_fast_refuse_when_only_todo_hits(tmp_path):
    """命中项仅 needs_plant_value（如问 X0.95）→ 快速拒答，且不调用 LLM。"""
    path = _kb(tmp_path)
    ans = answer_question("X0.95", kb_path=path, llm=RaisingLLM())
    assert ans.is_answerable is False
    assert "无法基于现有资料" in ans.answer_zh


def test_answer_refuses_via_sentinel(tmp_path):
    """检索到可答项但 LLM 判证据不足返回拒答哨兵 → is_answerable=False。"""
    path = _kb(tmp_path)
    ans = answer_question(
        "相邻分区温差上限", kb_path=path,
        llm=FakeLLM("无法基于现有资料回答（可能为 TODO(plant) 或源文档暂缺）。"),
    )
    assert ans.is_answerable is False
    assert ans.sources == []


def test_answer_uses_retrieved_context(tmp_path):
    """命中可答项 → 用 LLM 作答，标 is_answerable 与出处。"""
    path = _kb(tmp_path)
    ans = answer_question("相邻分区温差上限", kb_path=path, llm=FakeLLM("相邻温差≤5℃。出处：C1"))
    assert ans.is_answerable is True
    assert "5℃" in ans.answer_zh
    assert ans.sources


def test_generate_forces_needs_plant_on_todo_chunk(tmp_path):
    """含 TODO(plant) 的片段 → 生成的问答被保守标 needs_plant_value=True。"""
    doc = tmp_path / "src.md"
    doc.write_text(
        "普通片段：相邻分区温差不超过 5℃，这是已知的硬约束条款之一。\n\n"
        "缺值片段：8mm 的 X0.95 A 级上限为 TODO(plant)，源文档 docs/01 暂缺待补。\n",
        encoding="utf-8",
    )
    fake = FakeLLM('[{"question_zh":"问","answer_zh":"答","is_grounded":true,"needs_plant_value":false,"tags":["t"]}]')
    pairs = generate_qa_pairs([doc], llm=fake)
    assert len(pairs) == 2                              # 两个片段各一条
    todo_pairs = [p for p in pairs if p.needs_plant_value]
    assert len(todo_pairs) == 1                         # 仅缺值片段被强制标记

"""工艺知识库问答：链A 生成 Q&A 对建库 + 链B 基于库的词法检索应答（RAG）。

红线（铁律#5/#8）：docs/01·docs/03 缺失——
- 仅据片段/检索结果作答并标注出处；
- 片段含 TODO(plant) 的问答一律 needs_plant_value=True（保守，不得编造数字）；
- 检索为空 / 命中均 needs_plant_value → 拒答（is_answerable=False）。

Prompt 模板：config/prompt_kb_generate.md（链A）、config/prompt_kb_answer.md（链B）。
检索用**词法**（CJK 单字 + ASCII 词 的重叠打分），零额外依赖；嵌入检索留作后续可插点。
KB 落 data/kb/qa_pairs.jsonl（jsonl，写读校验）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llm_roles._llm import chat, load_llm
from schemas.kb import QAAnswer, QAPair

_ROOT = Path(__file__).resolve().parents[1]
_GEN_PROMPT_PATH = _ROOT / "config" / "prompt_kb_generate.md"
_ANS_PROMPT_PATH = _ROOT / "config" / "prompt_kb_answer.md"
DEFAULT_KB_PATH = _ROOT / "data" / "kb" / "qa_pairs.jsonl"

_REFUSAL = "无法基于现有资料回答（可能为 TODO(plant) 或源文档暂缺）。"
_QA_FIELDS = ("question_zh", "answer_zh", "sources", "is_grounded", "needs_plant_value", "tags")


# ---- 共用：分词 / 切片 / JSON 解析 ----
def _tokens(text: str) -> set[str]:
    """词法分词：CJK 单字 + ASCII 词（小写）。"""
    ascii_words = re.findall(r"[a-z0-9_]+", text.lower())
    cjk = re.findall(r"[一-鿿]", text)
    return set(ascii_words) | set(cjk)


def _chunk(text: str) -> list[str]:
    """按空行切片，丢弃过短块（<20 字符）。"""
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if len(b.strip()) >= 20]


def _parse_json_array(raw: str) -> list[dict]:
    """从模型输出中宽松解析 JSON 数组（容忍代码围栏/前后噪声）。"""
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


# ---- 链A：生成器 ----
def _coerce_qapair(obj: dict, source_id: str, plant_value_in_chunk: bool) -> QAPair | None:
    """把模型给的 dict 收敛成 QAPair（只取已知字段；含 TODO(plant) 片段则强制 needs_plant_value）。"""
    data: dict[str, Any] = {k: obj[k] for k in _QA_FIELDS if k in obj}
    sources = list(data.get("sources") or [])
    if source_id not in sources:
        sources.append(source_id)
    data["sources"] = sources
    if plant_value_in_chunk:                       # 保守：片段缺值 → 所有问答标记依赖缺值
        data["needs_plant_value"] = True
    try:
        return QAPair(**data)
    except ValidationError:
        return None


def generate_qa_pairs(
    source_paths: list[Path],
    *,
    max_pairs_per_chunk: int = 3,
    llm: Any | None = None,
) -> list[QAPair]:
    """从源文档片段生成 Q&A 对（仅据片段；片段含 TODO(plant) → needs_plant_value=True）。"""
    if llm is None:
        llm = load_llm()
    template = _GEN_PROMPT_PATH.read_text(encoding="utf-8")
    pairs: list[QAPair] = []
    for path in source_paths:
        text = Path(path).read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk(text)):
            source_id = f"{Path(path).name}#{i}"
            has_plant = "TODO(plant)" in chunk
            prompt = template.replace("{source_text}", chunk).replace("{source_id}", source_id)
            raw = chat(llm, prompt, max_tokens=1024)
            for obj in _parse_json_array(raw)[:max_pairs_per_chunk]:
                qa = _coerce_qapair(obj, source_id, has_plant)
                if qa is not None:
                    pairs.append(qa)
    return pairs


def write_kb(pairs: list[QAPair], out_path: Path = DEFAULT_KB_PATH) -> None:
    """写 jsonl 知识库（一行一条），并逐行读回校验。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [p.model_dump_json() for p in pairs]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    for line in lines:                             # 读回即校验（脏行抛 ValidationError）
        QAPair.model_validate_json(line)


def load_kb(kb_path: Path = DEFAULT_KB_PATH) -> list[QAPair]:
    """读回 jsonl 知识库（逐行校验）。文件不存在则返回空。"""
    kb_path = Path(kb_path)
    if not kb_path.exists():
        return []
    out: list[QAPair] = []
    for line in kb_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(QAPair.model_validate_json(line))
    return out


# ---- 链B：应答引擎 ----
def retrieve(question_zh: str, kb_path: Path = DEFAULT_KB_PATH, *, top_k: int = 5) -> list[QAPair]:
    """词法检索：按问题与 (问题+答案+标签) 的 token 重叠打分，返回 top_k（分>0，稳定序）。"""
    q = _tokens(question_zh)
    scored: list[tuple[int, int, QAPair]] = []
    for idx, pair in enumerate(load_kb(kb_path)):
        text = pair.question_zh + " " + pair.answer_zh + " " + " ".join(pair.tags)
        score = len(q & _tokens(text))
        if score > 0:
            scored.append((-score, idx, pair))     # -score 升序=分数降序；idx 稳定 tie-break
    scored.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in scored[:top_k]]


def answer_question(
    question_zh: str,
    *,
    kb_path: Path = DEFAULT_KB_PATH,
    top_k: int = 5,
    llm: Any | None = None,
) -> QAAnswer:
    """基于检索片段作答；检索为空/命中均 needs_plant_value → 拒答。"""
    hits = retrieve(question_zh, kb_path, top_k=top_k)
    answerable_hits = [h for h in hits if not h.needs_plant_value]
    retrieved_ids = [sid for h in hits for sid in h.sources]

    if not answerable_hits:                        # 无依据 / 全是缺值 → 拒答
        return QAAnswer(answer_zh=_REFUSAL, sources=[], is_answerable=False, retrieved=retrieved_ids)

    if llm is None:
        llm = load_llm()
    template = _ANS_PROMPT_PATH.read_text(encoding="utf-8")
    context_chunks = "\n\n".join(
        f"[{', '.join(h.sources)}] Q:{h.question_zh}\nA:{h.answer_zh}" for h in answerable_hits
    )
    prompt = template.replace("{question}", question_zh).replace("{context_chunks}", context_chunks)
    answer = chat(llm, prompt, max_tokens=512)
    # 哨兵识别：LLM 判定检索证据不足而返回固定拒答语 → 标不可答（诚实）
    refused = "无法基于现有资料" in answer
    sources = [] if refused else [sid for h in answerable_hits for sid in h.sources]
    return QAAnswer(answer_zh=answer, sources=sources, is_answerable=not refused, retrieved=retrieved_ids)

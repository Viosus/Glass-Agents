"""工艺知识库问答（接口桩，暂不实现业务逻辑）。

两条链：
- 链 A 生成器：从现有 docs/config 片段批量生成 Q&A 对建库（产物 schemas.kb.QAPair）。
- 链 B 应答引擎：基于库检索作答（RAG，产物 schemas.kb.QAAnswer）。

红线（铁律#5/#8）：docs/01·docs/03 缺失——
- 仅据给定片段/检索结果作答并标注出处；
- 问到 TODO(plant)/缺失常量一律拒答或标 needs_plant_value，绝不编造数字；
- 生成的问答对未经专家复核（is_reviewed=False）不作真值。

Prompt 模板：config/prompt_kb_generate.md（链 A）、config/prompt_kb_answer.md（链 B）。
检索先用词法（零额外依赖）；嵌入检索（sentence-transformers）留作后续可插点，本轮不做。
KB 落 data/kb/qa_pairs.jsonl。本轮只定接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from schemas.kb import QAAnswer, QAPair


# ---- 链 A：生成器（docs → Q&A 对建库）----
def generate_qa_pairs(
    source_paths: list[Path],
    *,
    max_pairs_per_chunk: int = 3,
    llm: Any | None = None,
) -> list[QAPair]:
    """从源文档片段生成 Q&A 对。

    约定：仅据片段生成、每条须能在片段中找到依据；片段中为 TODO(plant) 或未给出的
    数值/国标/安全常量不得编造，对应 QAPair.needs_plant_value=True、is_grounded 据实。
    TODO(impl)：切片 → 调 config/prompt_kb_generate.md → 解析 JSON → 装 QAPair。
    """
    raise NotImplementedError("generate_qa_pairs 待实现（链 A 生成器）")


def write_kb(pairs: list[QAPair], out_path: Path) -> None:
    """把 Q&A 对写入 jsonl 知识库（逐条写读校验，默认 data/kb/qa_pairs.jsonl）。"""
    raise NotImplementedError("write_kb 待实现（jsonl 落库）")


# ---- 链 B：应答引擎（RAG over KB）----
def retrieve(question_zh: str, kb_path: Path, *, top_k: int = 5) -> list[QAPair]:
    """词法检索：按问题召回最相关的若干 QAPair（嵌入检索为后续可插点）。"""
    raise NotImplementedError("retrieve 待实现（词法检索）")


def answer_question(
    question_zh: str,
    *,
    kb_path: Path,
    top_k: int = 5,
    llm: Any | None = None,
) -> QAAnswer:
    """基于检索片段作答。

    约定：只用检索片段作答并给出处；检索为空或命中均 needs_plant_value →
    is_answerable=False，拒答文案含「TODO(plant)/源文档暂缺」；严禁用片段外知识补数字。
    TODO(impl)：retrieve → 调 config/prompt_kb_answer.md → 装 QAAnswer。
    """
    raise NotImplementedError("answer_question 待实现（链 B 应答）")

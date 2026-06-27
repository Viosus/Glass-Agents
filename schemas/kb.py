"""工艺知识库问答的数据契约（pydantic 校验版）。

对齐 schemas/process_params.py 风格：写入即校验，脏字段拒绝。
红线（铁律#5/#8）：答案依赖 TODO(plant)/缺失常量时不得编造数字，须置 needs_plant_value=True；
生成的问答对在专家复核前（is_reviewed=False）不作真值。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QAPair(BaseModel):
    """知识库一条问答对（生成器链 A 的产物）。"""

    model_config = ConfigDict(extra="forbid")  # 脏字段拒绝

    question_zh: str                                 # 中文问题
    answer_zh: str                                   # 中文答案（仅据出处，不编造）
    sources: list[str] = Field(default_factory=list)  # 出处，如 "docs/xxx.md#C1"
    is_grounded: bool = False                        # 是否有出处支撑
    needs_plant_value: bool = False                  # 答案依赖 TODO(plant)→True
    is_reviewed: bool = False                        # 专家复核前不作真值
    tags: list[str] = Field(default_factory=list)    # 厚度/约束代号/术语 等


class QAAnswer(BaseModel):
    """应答引擎（链 B）对单个问题的回答。"""

    model_config = ConfigDict(extra="forbid")

    answer_zh: str                                   # 中文答案 或 拒答文案
    sources: list[str] = Field(default_factory=list)  # 引用出处
    is_answerable: bool = False                      # False=拒答（无依据/缺值）
    retrieved: list[str] = Field(default_factory=list)  # 命中片段 id，便于审计

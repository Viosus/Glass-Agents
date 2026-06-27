"""llm_roles/ —— 本地 LLM 角色层（与纯确定性工具 tools/ 分开）。

当前角色：
- param_translator：参数翻译官（已校验参数 JSON → 中文操作建议，展示层）。
- kb_qa：工艺知识库问答（docs→Q&A 对生成器 + 基于库的 RAG 应答引擎）。
共享：_llm（本地 GGUF 客户端）。安全闸门永远在模型之外（tools/constraints）。
"""

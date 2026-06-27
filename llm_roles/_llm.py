"""共享本地 LLM 客户端：加载本地 GGUF 模型并做一次对话补全。

供 llm_roles 下各角色（参数翻译官 / 知识库问答）复用，避免重复加载逻辑。
全程本地推理（llama-cpp + 本地 GGUF），不联网，符合“本地无云”原则。
后端（CPU/GPU）由 llama-cpp 编译版本决定；当前环境为 CPU wheel。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 默认模型路径：仓库内本地 GGUF（见 VERSION.md）
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"


def detect_gpu_layers() -> int:
    """探测可用后端：llama-cpp 为 GPU 构建且 torch 报告 CUDA 可用才 offload(-1)，否则 CPU(0)。"""
    try:
        import llama_cpp  # 本地库

        if not llama_cpp.llama_supports_gpu_offload():
            return 0
        import torch  # 本地库

        return -1 if torch.cuda.is_available() else 0
    except Exception:
        return 0


def load_llm(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    *,
    n_ctx: int = 8192,
    gpu_layers: int | None = None,
) -> Any:
    """加载本地 GGUF 模型，返回 llama_cpp.Llama 实例（无类型 stub，故标 Any）。"""
    from llama_cpp import Llama  # 本地库；仅在需要时导入，便于无依赖时给清晰报错

    layers = gpu_layers if gpu_layers is not None else detect_gpu_layers()
    return Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=layers, verbose=False)


def chat(llm: Any, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """对单条 prompt 做一次对话补全，返回模型输出文本（已 strip）。

    temperature 默认 0：角色多为确定性任务（翻译/受控问答），禁随机。
    """
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return out["choices"][0]["message"]["content"].strip()

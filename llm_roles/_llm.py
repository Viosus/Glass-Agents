"""共享本地 LLM 客户端：两个本地后端，统一 create_chat_completion 接口。

- **gguf**（llama-cpp）：轻角色默认（翻译官/知识库/审查）；当前 CPU wheel。
- **transformers**（bitsandbytes 4bit，GPU）：Teacher 正式选型（2026-07-02 拍板
  Qwen2.5-14B-Instruct，~9GB 显存）；经 TransformersChatAdapter 适配成 llama-cpp
  同款接口，chat() 与各角色零改动通用。
后端选择数据驱动：config/llm.yaml（load_teacher_llm 按需读取）。
全程本地推理，不联网，符合"本地无云推理"原则。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 默认模型路径：仓库内本地 GGUF（见 VERSION.md）
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
_LLM_CONFIG = ROOT / "config" / "llm.yaml"


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
    llm 可为 llama_cpp.Llama 或 TransformersChatAdapter（同款接口）。
    """
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return out["choices"][0]["message"]["content"].strip()


class TransformersChatAdapter:
    """transformers 后端适配器：对外暴露 llama-cpp 同款 create_chat_completion 接口。

    这样 chat() 与所有角色（teacher/翻译官/kb_qa/对话壳兜底）无需感知后端差异。
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        """持有已加载的 HF model 与 tokenizer（加载在 load_llm_transformers 完成）。"""
        self.model = model
        self.tokenizer = tokenizer

    def create_chat_completion(
        self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024, **kwargs: Any
    ) -> dict:
        """chat 模板 → generate → 只解码新生成段。temperature=0 → 贪心（do_sample=False）。"""
        import torch  # 本地库；延迟导入与 gguf 路径解耦

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(inputs, **gen_kwargs)
        text = self.tokenizer.decode(out[0][inputs.shape[1] :], skip_special_tokens=True)
        return {"choices": [{"message": {"content": text}}]}


def load_llm_transformers(model_path: Path | str, *, quant_4bit: bool = True) -> TransformersChatAdapter:
    """加载 HF 权重（默认 bitsandbytes nf4 4bit，12GB 显存可装 14B）→ 适配器。

    model_path 可为本地目录（推荐，先下载落盘）或 HF 模型 id（需已缓存，不在此处联网）。
    """
    import torch  # 本地库
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # 本地库

    quant_cfg = None
    if quant_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16 if quant_cfg is None else None,
    )
    model.eval()
    return TransformersChatAdapter(model, tokenizer)


def load_llm_config(path: Path | None = None) -> dict:
    """读 config/llm.yaml（每次调用按需读取；不存在返回空 dict）。"""
    import yaml  # 本地库

    p = Path(path) if path is not None else _LLM_CONFIG
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_teacher_llm(config_path: Path | None = None) -> Any:
    """按 config/llm.yaml 的 teacher 段加载 Teacher LLM（选型拍板：14B 4bit GPU）。

    backend=transformers：local_dir 存在用本地目录，否则用 model_id（须已在本机缓存）；
    backend=gguf：走 load_llm；未配置时回退默认 GGUF（诚实降级，开发机可用）。
    """
    cfg = (load_llm_config(config_path).get("teacher") or {})
    backend = cfg.get("backend", "gguf")
    if backend == "gguf":
        return load_llm(cfg.get("model_path", DEFAULT_MODEL_PATH))
    if backend == "transformers":
        local_dir = cfg.get("local_dir")
        local = ROOT / str(local_dir) if local_dir else None
        source = local if (local is not None and local.exists()) else cfg.get("model_id")
        if not source:
            raise ValueError("config/llm.yaml teacher.local_dir 不存在且未配 model_id，无法加载")
        return load_llm_transformers(source, quant_4bit=bool(cfg.get("quant_4bit", True)))
    raise ValueError(f"未知 LLM 后端: {backend}（支持 gguf / transformers）")

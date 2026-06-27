# VERSION

记录项目使用的外部模型权重 / 大文件版本（这些文件不入 git，见 `.gitignore`）。

## 本地模型（models/）

### Qwen2.5-3B-Instruct (GGUF, q4_k_m)
- **用途**：阶段 A Teacher 冷启动的本地 LLM 推理（无训练）。
- **文件**：`models/qwen2.5-3b-instruct-q4_k_m.gguf`
- **大小**：2,104,932,768 字节（≈ 1.96 GiB）
- **量化**：Q4_K_M
- **来源仓库**：`Qwen/Qwen2.5-3B-Instruct-GGUF`
- **下载镜像**：hf-mirror（`https://hf-mirror.com`）
- **SHA256**：`626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`
- **下载日期**：2026-06-27

> ⚠️ GGUF 为 llama.cpp 推理格式，**不能**用 transformers+peft QLoRA 栈微调；仅作 Teacher 本地推理。
> 运行需另装 GGUF 运行时（llama-cpp-python / ollama），当前环境尚未安装。

#### 重新下载（hf-mirror）
```bash
curl -L --fail -o "models/qwen2.5-3b-instruct-q4_k_m.gguf" \
  "https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
```

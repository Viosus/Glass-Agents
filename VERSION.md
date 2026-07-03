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

### Qwen2.5-14B-Instruct (HF safetensors, bf16)
- **用途**：Teacher 正式选型（2026-07-02 拍板，信息需求清单 D1）——bitsandbytes nf4 4bit 量化加载到 GPU
  （`llm_roles/_llm.py::load_teacher_llm`，选型配置 `config/llm.yaml`）。
- **目录**：`models/qwen2.5-14b-instruct/`（8 个 safetensors 分片 + tokenizer/config，共 ≈27.5 GB）
- **来源仓库**：`Qwen/Qwen2.5-14B-Instruct`（直连 huggingface.co 下载；本机 hf-mirror 有重定向问题）
- **下载日期**：2026-07-02
- **实测**：transformers 4.57.6 + bitsandbytes 0.49.2，RTX 5070 12GB 加载 ≈30s，显存占用 ≈10 GB
  （nf4+double quant，溢出模块允许回落 CPU）。⚠️ transformers 5.12 加载会原生崩溃，见 requirements.txt 锁定说明。
- **分片 SHA256**（与官方 LFS oid 逐一核对一致）：

| 分片 | SHA256 |
|---|---|
| model-00001-of-00008 | `b477be7572f0ab3ae3cbba38d508cc33e70600b2045669c4ad848051c3432094` |
| model-00002-of-00008 | `eb356aacae443e30f52712b1e98fadf206976365e2f5ee886321b0bb38c7cea8` |
| model-00003-of-00008 | `cee9dbdf738ce61adb21c70ad5c5585747caaab6457c9ffc3bfa20d73127c2c6` |
| model-00004-of-00008 | `b1201a6edcac8ef96a37040524fe37a5a84e446513ed40491e764a37302385fa` |
| model-00005-of-00008 | `fbbda3cdee31895d0c57d617449af5579d219df34f3d63f7a23acbfe7fb0a647` |
| model-00006-of-00008 | `f7ac652aa101717c97cfeccf5109985cca5b04529b57d3f4d2ec7a802266296b` |
| model-00007-of-00008 | `e9115e615b575acb6e8b3a06524b32cbf7f680f07517125e4334f62c3f4acb0d` |
| model-00008-of-00008 | `a77f603d8368749c19d3ac9cf2a300e476edcc794ea41af8e473ee12935eadb8` |

#### 重新下载（直连）
```powershell
& ".venv\Scripts\hf.exe" download Qwen/Qwen2.5-14B-Instruct --local-dir "models/qwen2.5-14b-instruct"
```

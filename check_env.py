import importlib

mods = [
    # 通用
    "numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn",
    "jupyterlab", "tensorboard", "wandb",
    # LLM
    "transformers", "datasets", "accelerate", "peft", "trl",
    "bitsandbytes", "sentencepiece", "safetensors", "tokenizers",
    "evaluate", "einops", "huggingface_hub",
    # CV
    "timm", "cv2", "albumentations", "skimage", "ultralytics", "pycocotools",
]

ok, bad = [], []
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        ok.append(f"{m} {v}")
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")

print("=== OK (%d) ===" % len(ok))
for s in ok:
    print(" ", s)
if bad:
    print("\n=== FAILED (%d) ===" % len(bad))
    for s in bad:
        print(" ", s)
else:
    print("\nAll imports OK.")

# bitsandbytes GPU check
try:
    import bitsandbytes as bnb
    import torch
    if torch.cuda.is_available():
        a = torch.randn(64, 64, device="cuda")
        print("\nbitsandbytes", bnb.__version__, "loaded; CUDA tensor ok on", torch.cuda.get_device_name(0))
except Exception as e:
    print("\nbitsandbytes GPU check failed:", e)

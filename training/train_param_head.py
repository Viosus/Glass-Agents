"""单头训练入口（真数据路径）：data/archive 真样本 → 只训参数头 → 版本门 → checkpoint。

与 train.py（模拟器脚手架）的分工：本文件走**真数据**，只对参数头反传
（质量/能耗/归因头等各自标签攒够后再开）；样本量不足门槛时诚实报错退出，
**绝不用模拟数据顶**。出参安全仍由 tools/constraints 在门内强制（规则 > AI）。

用法：& "D:\\Glass Agents\\.venv\\Scripts\\python.exe" training\\train_param_head.py ^
        --archive data\\archive [--steps 500] [--out runs\\param_head]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 支持直接 `python training/train_param_head.py`

import torch  # noqa: E402
from torch import nn  # noqa: E402

from schemas.archive import load_all  # noqa: E402
from schemas.bucketing import Bucket  # noqa: E402
from tools.eval_gate import GateResult, evaluate_param_gate  # noqa: E402
from training.dataset import time_ordered_split  # noqa: E402
from training.features import feature_dim  # noqa: E402
from training.model import MultiHeadCore  # noqa: E402
from training.targets import (  # noqa: E402
    PARAM_TARGET_FIELDS,
    ParamTrainingSet,
    build_param_training_set,
    delta_fields_sha256,
    feature_schema_sha256,
    load_training_config,
)


def _todo_or_float(value: object) -> float | None:
    """config 值 → float；TODO(plant)/None → None（如实降级，不填占位数）。"""
    if value is None:
        return None
    if isinstance(value, str):
        return None if value.strip().startswith("TODO(plant)") else float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def train_param_head(
    train_set: ParamTrainingSet,
    train_idx: list[int],
    *,
    steps: int = 500,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str | None = None,
    seed: int = 0,
) -> tuple[MultiHeadCore, list[float]]:
    """只训参数头：对 param_delta 算 MSE 反传（其余三头不参与 loss）。返回 (模型, loss 曲线)。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x = train_set.features[train_idx].to(device)
    y = train_set.deltas[train_idx].to(device)

    model = MultiHeadCore(feature_dim(), param_dim=len(PARAM_TARGET_FIELDS)).to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    n = x.shape[0]
    step = 0
    while step < steps:
        for start in range(0, n, batch_size):  # 不打乱（时间序铁律）
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            out = model(xb)
            loss = mse(out["param_delta"], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append(float(loss.detach().cpu()))
            step += 1
            if step >= steps:
                break
    return model, history


def per_bucket_metrics(
    model: MultiHeadCore, train_set: ParamTrainingSet, idx: list[int]
) -> dict[Bucket, dict[str, float]]:
    """指定切分上按桶报参数头 MAE（哪些桶学得好/差，一目了然）。"""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(train_set.features[idx].to(device))["param_delta"].cpu()
    target = train_set.deltas[idx]

    by_bucket: dict[Bucket, list[float]] = {}
    for row, i in enumerate(idx):
        mae = float((pred[row] - target[row]).abs().mean())
        by_bucket.setdefault(train_set.buckets[i], []).append(mae)
    return {b: {"mae_param": sum(v) / len(v), "n": float(len(v))} for b, v in sorted(by_bucket.items())}


def save_checkpoint(model: MultiHeadCore, out_dir: Path, meta: dict) -> Path:
    """存 checkpoint：weights.pt + meta.json（特征/Δ 契约指纹、样本数、时间）——版本门与模型包铺路。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / "weights.pt"
    torch.save(model.state_dict(), weights)
    full_meta = {
        "feature_schema_sha256": feature_schema_sha256(),
        "delta_fields_sha256": delta_fields_sha256(),
        "param_target_fields": list(PARAM_TARGET_FIELDS),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(full_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return weights


def run_gate_on_split(
    model: MultiHeadCore, train_set: ParamTrainingSet, idx: list[int], *, max_mae: float | None
) -> GateResult:
    """在指定切分上跑参数头版本门（逐样本各自 baseline 解码 → constraints）。"""
    return evaluate_param_gate(
        model,
        train_set.features[idx],
        train_set.deltas[idx],
        [train_set.baselines[i] for i in idx],
        max_mae=max_mae,
    )


def main() -> int:
    """CLI：读档 → 构训练集 → 门槛检查 → 训练 → 按桶指标 + 版本门 → checkpoint。"""
    ap = argparse.ArgumentParser(description="参数头单头训练（真数据；样本不足则诚实退出）")
    ap.add_argument("--archive", type=Path, default=Path("data/archive"), help="归档目录")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("runs/param_head"), help="checkpoint 输出目录")
    args = ap.parse_args()

    if not args.archive.exists():
        print(f"归档目录不存在：{args.archive}（先跑 tools/ingest_annotations.py 导入标注表）")
        return 1
    cfg = load_training_config()
    train_set = build_param_training_set(load_all(args.archive))
    n = train_set.features.shape[0]
    min_n = int(cfg.get("min_train_samples", 30))
    print(f"可训样本 {n}（剔除 {train_set.dropped}：非真值/无基准）；门槛 min_train_samples={min_n}")
    if n < min_n:
        print("样本量不足，诚实退出——绝不用模拟数据顶（等老师傅标注攒够再训）。")
        return 1

    tr, va, te = time_ordered_split(n, float(cfg.get("val_frac", 0.2)), float(cfg.get("test_frac", 0.2)))
    model, history = train_param_head(train_set, tr, steps=args.steps, lr=args.lr)
    model = model.cpu()
    print(f"loss: 首 {history[0]:.4f} → 末 {history[-1]:.4f}（{len(history)} 步）")

    print("验证集按桶 MAE：")
    for bucket, m in per_bucket_metrics(model, train_set, va).items():
        print(f"  {bucket}: MAE={m['mae_param']:.4f} n={int(m['n'])}")

    max_mae = _todo_or_float((cfg.get("gate") or {}).get("max_param_mae"))
    gate = run_gate_on_split(model, train_set, te, max_mae=max_mae)
    print(f"版本门(test): passed={gate.passed} 零违规={gate.constraints_ok} MAE={gate.metrics['mae_param']:.4f}")
    for d in gate.details:
        print(f"  - {d}")

    weights = save_checkpoint(
        model,
        args.out,
        {"train_sample_count": n, "gate_passed": gate.passed, "gate_metrics": gate.metrics},
    )
    print(f"checkpoint 已存 {weights}（含契约指纹 meta.json）")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    sys.exit(main())

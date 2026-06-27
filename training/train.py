"""N4 训练脚手架入口：在**已知 DGP 模拟器**上跑通管线并**验证可学**（非真训练）。

真训练受 N2 数据门、N3 决策门约束；I/O 维度为临时契约。
规则永不进权重：模型出参 → 基准+Δ(decode) → 过 tools/constraints 闸门(eval_gate/safety_gate)。

用法：& "D:\\Glass Agents\\.venv\\Scripts\\python.exe" training\\train.py --synthetic --steps 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 支持直接 `python training/train.py`

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from schemas.process_params import ProcessParams  # noqa: E402
from tools.constraints import CheckResult, validate  # noqa: E402
from tools.eval_gate import evaluate_gate, head_metrics  # noqa: E402
from training.dataset import SampleDataset, time_ordered_split  # noqa: E402
from training.decode import decode_params  # noqa: E402
from training.losses import MultiHeadLoss  # noqa: E402
from training.model import MultiHeadCore  # noqa: E402
from training.simulator import make_simulated  # noqa: E402

IN_DIM = 12
PARAM_DIM = 6
ATTR_DIM = 4


def baseline_recipe(zone_temps: list[float] | None = None) -> ProcessParams:
    """一个结构合法的基准配方（占位；真实基准来自现场配方库）。"""
    zt = zone_temps or [100.0, 102.0]
    return ProcessParams(
        zone_temps=zt,
        zone_roles=["center"] * len(zt),
        temp_upper=700.0,
        temp_lower=650.0,
        convection_speed=1.0,
        convection_ratio_upper_lower=1.0,
        oscillation_speed=1.0,
        oscillation_amplitude=1.0,
        heating_duration_s=200.0,
        glass_type="clear",
        thickness_mm=6.0,
        quality_mode="high_quality",
    )


def _load_split(n: int, seed: int = 0):
    """生成模拟数据并按时间序切分，返回 (features, targets, (tr, va, te))。"""
    features, targets = make_simulated(n=n, in_dim=IN_DIM, param_dim=PARAM_DIM, attr_dim=ATTR_DIM, seed=seed)
    tr, va, te = time_ordered_split(n)
    return features, targets, (tr, va, te)


def smoke_train(steps: int = 5, n: int = 256, device: str | None = None, seed: int = 0):
    """在模拟器数据上训练若干步，返回 (model, loss_history, split_idx)。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    features, targets, (tr, va, te) = _load_split(n, seed=seed)
    ds = SampleDataset(features[tr], {k: v[tr] for k, v in targets.items()})
    dl = DataLoader(ds, batch_size=16, shuffle=False)  # 不打乱（时间序）

    model = MultiHeadCore(IN_DIM, param_dim=PARAM_DIM, attr_dim=ATTR_DIM).to(device)
    loss_fn = MultiHeadLoss().to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)

    history: list[float] = []
    step = 0
    while step < steps:
        for x, y in dl:
            x = x.to(device)
            out = model(x)
            losses = {
                "param": mse(out["param_delta"], y["param"].to(device)),
                "quality": mse(out["quality"], y["quality"].to(device)),
                "energy": mse(out["energy"], y["energy"].to(device)),
                "attribution": mse(out["attribution"], y["attribution"].to(device)),
            }
            total = loss_fn(losses)
            opt.zero_grad()
            total.backward()
            opt.step()
            history.append(float(total.detach().cpu()))
            step += 1
            if step >= steps:
                break
    return model, history, (tr, va, te)


def safety_gate(param_delta, baseline: ProcessParams, thresholds: dict | None = None) -> CheckResult:
    """规则永不进权重：Δ → decode 出参 → 过 tools.constraints.validate（安全闸门外挂于模型）。"""
    return validate(decode_params(param_delta, baseline).to_param_set(), thresholds=thresholds)


def main() -> int:
    ap = argparse.ArgumentParser(description="N4 训练脚手架（模拟器数据，验证可学）")
    ap.add_argument("--synthetic", action="store_true", help="用模拟器数据跑通并验证（当前唯一支持）")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    if not args.synthetic:
        print("当前仅支持 --synthetic（真数据待 N2 数据门 / N3 决策门）。")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history, splits = smoke_train(steps=args.steps, n=args.n)
    model = model.cpu()  # 训练后统一回 cpu 做评测，避免设备不一致
    tr, va, te = splits
    features, targets, _ = _load_split(args.n)

    # 训练前后 loss + 验证集各头指标（证"确实在学"）
    print(f"设备: {device}；切分(train/val/test)={tuple(len(s) for s in splits)}")
    print(f"loss: 首 {history[0]:.4f} → 末 {history[-1]:.4f}（{len(history)} 步）")
    with torch.no_grad():
        out_va = model(features[va])
    m = head_metrics(out_va, {k: v[va] for k, v in targets.items()})
    print(
        f"验证集指标: R²(quality)={m['r2_quality']:.3f}  "
        f"R²(energy)={m['r2_energy']:.3f}  MSE(param)={m['mse_param']:.3f}"
    )

    # 验证门（默认 config 多 TODO → 预期因出参违规而不放行）
    gate = evaluate_gate(model, features[te], {k: v[te] for k, v in targets.items()}, baseline_recipe())
    print(f"验证门: passed={gate.passed} 质量不回归={gate.quality_not_regressed} 零违规={gate.constraints_ok}")
    for d in gate.details:
        print(f"  - {d}")
    print("注：脚手架/模拟器数据，非真训练；系数虚构、I/O 临时契约待 N3。")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    sys.exit(main())

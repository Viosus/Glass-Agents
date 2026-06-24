"""N4 训练脚手架入口：合成数据冒烟，证明"整条管线连通"。

**不做真训练/收敛声明**；真训练受 N2 数据门、N3 决策门约束。I/O 维度为临时契约。
规则永不进权重：模型出参 → 基准+Δ → 过 tools/constraints 闸门。

用法：& "D:\\Glass Agents\\.venv\\Scripts\\python.exe" training\\train.py --synthetic --steps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 支持直接 `python training/train.py`

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tools.constraints import CheckResult, ParamSet, validate  # noqa: E402
from training.dataset import SampleDataset, time_ordered_split  # noqa: E402
from training.losses import MultiHeadLoss  # noqa: E402
from training.model import MultiHeadCore  # noqa: E402
from training.synthetic import make_synthetic  # noqa: E402

IN_DIM = 12
PARAM_DIM = 6
ATTR_DIM = 4


def smoke_train(steps: int = 5, n: int = 64, device: str | None = None):
    """合成数据冒烟训练若干步，返回 (model, loss_history, split_idx)。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    features, targets = make_synthetic(n=n, in_dim=IN_DIM, param_dim=PARAM_DIM, attr_dim=ATTR_DIM)
    tr, va, te = time_ordered_split(n)
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


def safety_gate(param_delta, baseline_temps) -> CheckResult:
    """规则永不进权重：把模型 Δ 叠加到基准配方（占位映射）→ 过 tools.constraints.validate。

    占位映射仅示意（真实 Δ→参数映射待 N3）；要点是 Student 出参**必过硬约束闸门**。
    """
    p = ParamSet(
        zone_temps=list(baseline_temps),
        zone_roles=["center"] * len(baseline_temps),
        temp_upper=700.0 + float(param_delta[0]),
        temp_lower=650.0 + float(param_delta[1]),
        convection_speed=1.0,
        convection_ratio_upper_lower=1.0,
        oscillation_speed=1.0,
        oscillation_amplitude=1.0,
        heating_duration_s=200.0,
        glass_type="clear",
        thickness_mm=6.0,
        quality_mode="high_quality",
    )
    return validate(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="N4 训练脚手架（仅合成数据）")
    ap.add_argument("--synthetic", action="store_true", help="用合成数据跑通管线（当前唯一支持）")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    if not args.synthetic:
        print("当前仅支持 --synthetic（真数据待 N2 数据门 / N3 决策门）。")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history, splits = smoke_train(steps=args.steps)
    print(f"设备: {device}；切分(train/val/test)={tuple(len(s) for s in splits)}")
    print(f"loss 轨迹({len(history)} 步): {[round(h, 4) for h in history]}")

    res = safety_gate(np.zeros(PARAM_DIM), [100.0, 102.0])
    print(
        "规则闸门 demo（默认 config 多 TODO → 预期拦截）："
        f"within_limits={res.within_limits}, #violations={len(res.violations)}"
    )
    print("注：脚手架/合成数据，非真训练；I/O 临时契约待 N3。")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    sys.exit(main())

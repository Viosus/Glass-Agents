"""评测门（N5 验证门预演）：各头指标 + "质量不回归 + 出参零违规"放行判定。

两块：
1) head_metrics —— 各头回归指标（param MSE/MAE、quality/energy 的 R²）。
2) evaluate_gate —— 放行逻辑：质量 R² 不低于门槛(不回归) **且** 解码出参全部过 constraints.validate(零违规)。
安全判定仍由 tools.constraints 强制（规则 > AI）；本模块只做"是否可放行"的汇总判定。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from schemas.process_params import ProcessParams
from tools.constraints import validate
from training.decode import decode_params


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """决定系数 R² = 1 - SS_res/SS_tot；target 方差为 0 时返回 0.0。"""
    pred = pred.detach().flatten()
    target = target.detach().flatten()
    ss_tot = float(((target - target.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return 0.0
    ss_res = float(((target - pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def head_metrics(out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, float]:
    """各头指标。out 用模型键(param_delta/quality/energy/attribution)，targets 用 param/quality/energy/attribution。"""
    pd = out["param_delta"].detach()
    return {
        "mse_param": float(((pd - targets["param"]) ** 2).mean()),
        "mae_param": float((pd - targets["param"]).abs().mean()),
        "r2_quality": r2_score(out["quality"], targets["quality"]),
        "r2_energy": r2_score(out["energy"], targets["energy"]),
    }


@dataclass
class GateResult:
    """验证门结果。"""

    passed: bool
    quality_r2: float
    quality_not_regressed: bool
    constraints_ok: bool
    n_constraint_violations: int    # 解码出参未过闸门的样本数
    metrics: dict[str, float] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)


def evaluate_param_gate(
    model,
    features: torch.Tensor,
    delta_targets: torch.Tensor,
    baselines: list[ProcessParams],
    *,
    thresholds: dict | None = None,
    max_mae: float | None = None,
) -> GateResult:
    """参数头版本门（真数据路径）：逐样本用**各自的** baseline 解码出参过闸门。

    与 evaluate_gate 的差异：真数据每片基准不同（baselines 与 features 行一一对应）；
    只评参数头——放行 = (max_mae 未配置 或 val MAE ≤ max_mae) 且 全部出参 within_limits。
    max_mae 未配置（config gate.max_param_mae=TODO(plant)）时如实降级为只查违规并写入 details。
    复用 GateResult：quality_r2 置 0、quality_not_regressed 置 True（本门不评质量头）。
    """
    if features.shape[0] != len(baselines):
        raise ValueError(f"features({features.shape[0]}) 与 baselines({len(baselines)}) 数量不一致")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(features.to(device))
    deltas = out["param_delta"].detach().cpu()
    delta_targets = delta_targets.cpu()

    mae = float((deltas - delta_targets).abs().mean()) if deltas.numel() else 0.0
    mae_ok = True
    details: list[str] = []
    if max_mae is None:
        details.append("gate.max_param_mae 未配置（TODO(plant)）→ 本门只查出参违规")
    else:
        mae_ok = mae <= max_mae
        if not mae_ok:
            details.append(f"参数头 MAE 超限：{mae:.4f} > {max_mae}")

    n_bad = 0
    for i in range(deltas.shape[0]):
        params = decode_params(deltas[i], baselines[i])
        res = validate(params.to_param_set(), thresholds=thresholds)
        if not res.within_limits:
            n_bad += 1
    constraints_ok = n_bad == 0
    if not constraints_ok:
        details.append(f"出参违规：{n_bad}/{deltas.shape[0]} 条未过 constraints.validate")

    return GateResult(
        passed=mae_ok and constraints_ok,
        quality_r2=0.0,
        quality_not_regressed=True,
        constraints_ok=constraints_ok,
        n_constraint_violations=n_bad,
        metrics={"mae_param": mae},
        details=details,
    )


def evaluate_gate(
    model,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    baseline: ProcessParams,
    *,
    thresholds: dict | None = None,
    min_quality_r2: float = 0.0,
) -> GateResult:
    """跑模型 → 算指标 → 解码每条出参过闸门 → 汇总放行判定。

    放行 = (quality R² ≥ min_quality_r2) 且 (所有解码出参 within_limits)。
    thresholds 可注入（默认读 config，含多 TODO → 预期非零违规、不放行）。
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(features.to(device))
    out = {k: v.detach().cpu() for k, v in out.items()}      # 统一回 cpu，避免与 targets 设备不一致
    targets = {k: v.cpu() for k, v in targets.items()}

    metrics = head_metrics(out, targets)
    quality_r2 = metrics["r2_quality"]
    quality_not_regressed = quality_r2 >= min_quality_r2

    # 逐样本解码 Δ→参数 → 过硬约束闸门，统计未过数
    n_bad = 0
    deltas = out["param_delta"].detach()
    for i in range(deltas.shape[0]):
        params = decode_params(deltas[i], baseline)
        res = validate(params.to_param_set(), thresholds=thresholds)
        if not res.within_limits:
            n_bad += 1
    constraints_ok = n_bad == 0

    details: list[str] = []
    if not quality_not_regressed:
        details.append(f"质量回归：R²={quality_r2:.3f} < 门槛 {min_quality_r2}")
    if not constraints_ok:
        details.append(f"出参违规：{n_bad}/{deltas.shape[0]} 条未过 constraints.validate")

    return GateResult(
        passed=quality_not_regressed and constraints_ok,
        quality_r2=quality_r2,
        quality_not_regressed=quality_not_regressed,
        constraints_ok=constraints_ok,
        n_constraint_violations=n_bad,
        metrics=metrics,
        details=details,
    )

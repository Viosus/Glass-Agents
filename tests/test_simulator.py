"""模拟器 + 评测门单测：可学性（模型确实从 DGP 学到信号）、时间漂移、门能拦回归。"""

import torch

from tools.eval_gate import evaluate_gate, head_metrics, r2_score
from training.simulator import make_simulated
from training.train import baseline_recipe, smoke_train


def test_simulator_shapes_and_determinism():
    f1, t1 = make_simulated(n=32, seed=0)
    f2, t2 = make_simulated(n=32, seed=0)
    assert f1.shape == (32, 12)
    assert t1["param"].shape == (32, 6)
    assert t1["attribution"].shape == (32, 4)
    assert torch.equal(f1, f2)                    # 同 seed → 确定性
    assert torch.equal(t1["quality"], t2["quality"])


def test_drift_shifts_distribution():
    # 概念漂移：尾部样本第 0 维均值应明显高于头部（按时间序的分布偏移）
    f, _ = make_simulated(n=200, seed=1, drift=5.0)
    head_mean = float(f[:50, 0].mean())
    tail_mean = float(f[-50:, 0].mean())
    assert tail_mean - head_mean > 1.0


def test_model_actually_learns_on_simulator():
    # 在已知 DGP 上训练足够步，验证集 quality 的 R² 应显著为正（区别于随机不可学）
    model, history, splits = smoke_train(steps=400, n=256, device="cpu", seed=0)
    _, va, _ = splits
    features, targets = make_simulated(n=256, seed=0)
    with torch.no_grad():
        out = model(features[va])
    m = head_metrics(out, {k: v[va] for k, v in targets.items()})
    assert history[-1] < history[0]               # loss 下降
    assert m["r2_quality"] > 0.5                  # 确实学到信号


def test_r2_zero_variance_target():
    assert r2_score(torch.zeros(5), torch.ones(5)) == 0.0  # target 无方差 → 0.0


def test_gate_blocks_quality_regression():
    # 故意只训 1 步（欠拟合）→ 设高门槛 → 质量回归 → 门拦下
    model, _, splits = smoke_train(steps=1, n=64, device="cpu", seed=0)
    _, _, te = splits
    features, targets = make_simulated(n=64, seed=0)
    gate = evaluate_gate(
        model, features[te], {k: v[te] for k, v in targets.items()},
        baseline_recipe(), min_quality_r2=0.99,
    )
    assert gate.quality_not_regressed is False
    assert gate.passed is False


def test_gate_blocks_on_constraint_violation_default_config():
    # 默认 config 多 TODO(plant) → 解码出参必违规 → 门因零违规不成立而拦下
    model, _, splits = smoke_train(steps=5, n=64, device="cpu", seed=0)
    _, _, te = splits
    features, targets = make_simulated(n=64, seed=0)
    gate = evaluate_gate(
        model, features[te], {k: v[te] for k, v in targets.items()},
        baseline_recipe(), min_quality_r2=-1.0,   # 故意放低质量门槛，隔离出"违规"这一因素
    )
    assert gate.constraints_ok is False
    assert gate.n_constraint_violations > 0
    assert gate.passed is False

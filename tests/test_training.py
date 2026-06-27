"""N4 训练脚手架单测：管线连通、损失可算、一步训练能跑、切分无泄漏、出参过闸门。"""

import numpy as np
import torch

from tools.constraints import CheckResult
from training.dataset import time_ordered_split
from training.losses import MultiHeadLoss
from training.model import MultiHeadCore
from training.train import ATTR_DIM, IN_DIM, PARAM_DIM, baseline_recipe, safety_gate, smoke_train


def test_model_forward_shapes():
    """各头输出形状正确。"""
    m = MultiHeadCore(IN_DIM, param_dim=PARAM_DIM, attr_dim=ATTR_DIM)
    out = m(torch.randn(4, IN_DIM))
    assert out["param_delta"].shape == (4, PARAM_DIM)
    assert out["quality"].shape == (4, 1)
    assert out["energy"].shape == (4, 1)
    assert out["attribution"].shape == (4, ATTR_DIM)


def test_multihead_loss_is_scalar():
    loss_fn = MultiHeadLoss()
    losses = {h: torch.tensor(1.0) for h in ("param", "quality", "energy", "attribution")}
    total = loss_fn(losses)
    assert total.ndim == 0  # 标量
    assert torch.isfinite(total)


def test_smoke_train_runs_steps():
    """冒烟训练能跑指定步数且 loss 有限。"""
    _, history, _ = smoke_train(steps=2, n=32, device="cpu")
    assert len(history) == 2
    assert all(np.isfinite(h) for h in history)


def test_time_ordered_split_no_leakage():
    tr, va, te = time_ordered_split(100, val_frac=0.2, test_frac=0.2)
    assert len(tr) + len(va) + len(te) == 100
    assert set(tr) & set(va) == set()
    assert set(va) & set(te) == set()
    assert set(tr) & set(te) == set()
    # 时间序：train 全在 val 前、val 全在 test 前（无未来泄漏）
    assert max(tr) < min(va) < min(te)
    assert max(va) < min(te)


def test_model_output_passes_safety_gate():
    model, _, _ = smoke_train(steps=1, n=32, device="cpu")
    delta = model(torch.randn(1, IN_DIM))["param_delta"].detach().numpy().ravel()
    res = safety_gate(delta, baseline_recipe())
    assert isinstance(res, CheckResult)         # 规则闸门外挂于模型，被应用
    assert res.within_limits is False           # 默认 config 多 TODO → 保守拦截（规则不在权重里）

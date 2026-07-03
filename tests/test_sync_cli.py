"""sync_cli 登记/晋级/激活接线单测：register-model 正路 + 留存验证门 + TODO config 下的诚实拒绝。

不训真模型（随机初始化即可测接线）；全程 tmp_path，不碰真实 data/ 与 MODELS.md。
"""

import argparse
from datetime import datetime, timedelta

import pytest

from schemas.archive import ArchiveSample, MetricRecord, write_sample
from schemas.process_params import ProcessParams
from tools.model_registry import read_ledger
from tools.sync_cli import _holdout_gate, cmd_activate_model, cmd_promote_model, cmd_register_model
from training.features import feature_dim
from training.model import MultiHeadCore
from training.targets import PARAM_TARGET_FIELDS
from training.train_param_head import save_checkpoint

FULL_THR = {
    "gradient": {"adjacent_zone_max_delta_c": 5, "single_step_max_delta_c": 3},
    "thickness_duration": {"6": [100, 300]},
    "convection": {"clear": [1.0, 2.0]},
    "safety": {"blowup_rule": "rule_v0", "max_gradient": 50},
}

BASE = ProcessParams(
    zone_temps=[102.0, 100.0],
    zone_roles=["center", "edge"],
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


def write_archive(root, n: int = 12) -> None:
    """写 n 条带基准的真值样本（留存验证门的口粮）。"""
    t0 = datetime(2026, 7, 2, 8, 0, 0)
    for i in range(n):
        final = BASE.model_copy(update={"temp_upper": BASE.temp_upper + 1.0})
        write_sample(
            ArchiveSample(
                sample_id=f"s{i:03d}",
                created_at=t0 + timedelta(minutes=i),
                source="synthetic/test",
                furnace_id="F1",
                thickness_mm=6.0,
                glass_type="clear",
                quality_mode="high_quality",
                is_ground_truth=True,
                params=final,
                baseline_params=BASE,
                metrics=MetricRecord(x0_95_nm=80.0),
            ),
            root,
        )


@pytest.fixture
def checkpoint(tmp_path):
    """train_param_head.save_checkpoint 产出的完整 checkpoint 目录（weights.pt + meta.json）。"""
    model = MultiHeadCore(feature_dim(), param_dim=len(PARAM_TARGET_FIELDS))
    ckpt_dir = tmp_path / "ckpt"
    meta = {"train_sample_count": 48, "gate_metrics": {"mae_param": 0.1}, "gate_passed": True}
    save_checkpoint(model, ckpt_dir, meta)
    return ckpt_dir


# --------------------------- register-model --------------------------- #
def test_register_model_cmd(checkpoint, tmp_path):
    reg, md = tmp_path / "registry", tmp_path / "MODELS.md"
    rc = cmd_register_model(argparse.Namespace(checkpoint_dir=checkpoint, registry=reg, models_md=md))
    assert rc == 0
    assert (reg / "param_head-v0001" / "weights.pt").exists()
    assert (reg / "param_head-v0001" / "model_card.json").exists()
    text = md.read_text(encoding="utf-8")
    assert "param_head-v0001" in text and "候选" in text            # 人读账本追加"候选"行
    assert [e["event"] for e in read_ledger(reg)] == ["register"]


def test_register_model_cmd_incomplete_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = cmd_register_model(argparse.Namespace(checkpoint_dir=empty, registry=tmp_path / "r", models_md=tmp_path / "m"))
    assert rc == 1                                                   # 目录不完整 → 拒绝，不猜


# --------------------------- 留存验证门 helper --------------------------- #
def test_holdout_gate_runs_with_injected_thresholds(checkpoint, tmp_path):
    archive = tmp_path / "archive"
    write_archive(archive)
    gate = _holdout_gate(archive, checkpoint / "weights.pt", thresholds=FULL_THR)
    assert gate is not None and "mae_param" in gate.metrics          # 接线通（随机模型不保证过门）


def test_holdout_gate_refuses_without_samples(checkpoint, tmp_path):
    assert _holdout_gate(tmp_path / "不存在", checkpoint / "weights.pt") is None
    empty = tmp_path / "empty_archive"
    empty.mkdir()
    assert _holdout_gate(empty, checkpoint / "weights.pt") is None   # 无样本拒绝盲判


# --------------------------- promote/activate：TODO config 下诚实拒绝 --------------------------- #
def test_promote_and_activate_rejected_under_todo_config(checkpoint, tmp_path):
    """真实 config/thresholds.yaml 多项 TODO(plant) → 出参违规 → 两重门都如实拒绝（预期护栏行为）。"""
    reg, md = tmp_path / "registry", tmp_path / "MODELS.md"
    archive = tmp_path / "archive"
    write_archive(archive)
    assert cmd_register_model(argparse.Namespace(checkpoint_dir=checkpoint, registry=reg, models_md=md)) == 0

    ns = argparse.Namespace(model_id="param_head-v0001", registry=reg, archive=archive, models_md=md)
    assert cmd_promote_model(ns) == 1
    assert cmd_activate_model(ns) == 1
    events = [e["event"] for e in read_ledger(reg)]
    assert "promote_rejected" in events and "activate_rejected" in events
    text = md.read_text(encoding="utf-8")
    assert "晋级" not in text and "激活" not in text                 # 被拒不上人读账本

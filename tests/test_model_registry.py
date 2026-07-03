"""模型版本门单测：登记/晋级三分支/炉侧激活/模型包往返与指纹拒收。全程 tmp_path。"""

import json
import zipfile

import pytest
import torch

from sync.datapack import export_modelpack, import_modelpack
from tools.eval_gate import GateResult
from tools.model_registry import (
    activate,
    append_models_md,
    current_active,
    current_promoted,
    load_model_card,
    promote,
    read_ledger,
    register_candidate,
)
from training.features import feature_dim
from training.model import MultiHeadCore
from training.targets import PARAM_TARGET_FIELDS, delta_fields_sha256, feature_schema_sha256

CFG = {"min_train_samples": 30, "gate": {"regression_tolerance": 0.05}}


def make_meta(**over) -> dict:
    meta = {
        "feature_schema_sha256": feature_schema_sha256(),
        "delta_fields_sha256": delta_fields_sha256(),
        "train_sample_count": 48,
        "gate_metrics": {"mae_param": 0.10},
        "gate_passed": True,
    }
    meta.update(over)
    return meta


def make_gate(passed=True, mae=0.10, details=None) -> GateResult:
    return GateResult(
        passed=passed,
        quality_r2=0.0,
        quality_not_regressed=True,
        constraints_ok=passed,
        n_constraint_violations=0 if passed else 1,
        metrics={"mae_param": mae},
        details=details or ([] if passed else ["出参违规：1/8"]),
    )


@pytest.fixture
def weights(tmp_path):
    """一份真实结构的参数头权重文件。"""
    model = MultiHeadCore(feature_dim(), param_dim=len(PARAM_TARGET_FIELDS))
    p = tmp_path / "weights.pt"
    torch.save(model.state_dict(), p)
    return p


# --------------------------- 登记 --------------------------- #
def test_register_sequential_ids_and_card(weights, tmp_path):
    reg = tmp_path / "registry"
    id1 = register_candidate(weights, make_meta(), reg)
    id2 = register_candidate(weights, make_meta(), reg)
    assert (id1, id2) == ("param_head-v0001", "param_head-v0002")

    card = load_model_card(id1, reg)
    assert len(card["weights_sha256"]) == 64
    assert card["feature_schema_sha256"] == feature_schema_sha256()
    events = read_ledger(reg)
    assert [e["event"] for e in events] == ["register", "register"]


def test_register_missing_fingerprint_rejected(weights, tmp_path):
    meta = make_meta()
    del meta["feature_schema_sha256"]
    with pytest.raises(ValueError, match="缺元数据键"):
        register_candidate(weights, meta, tmp_path / "registry")


# --------------------------- 晋级三分支 --------------------------- #
def test_promote_first_version(weights, tmp_path):
    reg = tmp_path / "registry"
    mid = register_candidate(weights, make_meta(), reg)
    ok, reason = promote(mid, make_gate(), reg, config=CFG)
    assert ok and current_promoted(reg) == mid


def test_promote_rejected_on_gate_or_samples(weights, tmp_path):
    reg = tmp_path / "registry"
    mid = register_candidate(weights, make_meta(), reg)
    ok, reason = promote(mid, make_gate(passed=False), reg, config=CFG)
    assert not ok and "版本门未过" in reason

    mid2 = register_candidate(weights, make_meta(train_sample_count=10), reg)
    ok2, reason2 = promote(mid2, make_gate(), reg, config=CFG)
    assert not ok2 and "样本量不足" in reason2
    assert current_promoted(reg) is None                     # 全被拒 → 无晋级版本


def test_promote_regression_tolerance(weights, tmp_path):
    reg = tmp_path / "registry"
    v1 = register_candidate(weights, make_meta(), reg)
    assert promote(v1, make_gate(mae=0.10), reg, config=CFG)[0]

    v2 = register_candidate(weights, make_meta(), reg)
    ok_bad, reason = promote(v2, make_gate(mae=0.20), reg, config=CFG)   # 回归 100% → 拒
    assert not ok_bad and "回归" in reason

    v3 = register_candidate(weights, make_meta(), reg)
    assert promote(v3, make_gate(mae=0.102), reg, config=CFG)[0]         # 容差 5% 内 → 过
    assert current_promoted(reg) == v3


# --------------------------- 炉侧激活（第二重门） --------------------------- #
def test_activate_second_gate(weights, tmp_path):
    reg = tmp_path / "registry"
    mid = register_candidate(weights, make_meta(), reg)
    ok, _ = activate(mid, make_gate(passed=False), reg)
    assert not ok and current_active(reg) is None            # 本地门未过 → 不激活

    ok2, _ = activate(mid, make_gate(), reg)
    assert ok2 and current_active(reg) == mid


# --------------------------- 模型包往返与拒收 --------------------------- #
def test_modelpack_roundtrip(weights, tmp_path):
    reg = tmp_path / "registry"
    mid = register_candidate(weights, make_meta(), reg)
    pack = export_modelpack(reg / mid, tmp_path / "out")

    reg2 = tmp_path / "registry_furnace"
    manifest = import_modelpack(pack, reg2)
    assert manifest.model_id == mid and manifest.gate_passed
    assert (reg2 / mid / "weights.pt").exists()
    state = torch.load(reg2 / mid / "weights.pt", map_location="cpu", weights_only=True)
    assert any(k.startswith("param_head") for k in state)


def test_modelpack_fingerprint_mismatch_rejected(weights, tmp_path):
    reg = tmp_path / "registry"
    mid = register_candidate(weights, make_meta(feature_schema_sha256="0" * 64), reg)
    pack = export_modelpack(reg / mid, tmp_path / "out")
    with pytest.raises(ValueError, match="特征契约指纹"):
        import_modelpack(pack, tmp_path / "registry2")


def test_modelpack_tampered_weights_rejected(weights, tmp_path):
    reg = tmp_path / "registry"
    mid = register_candidate(weights, make_meta(), reg)
    pack = export_modelpack(reg / mid, tmp_path / "out")

    bad = tmp_path / "tampered.zip"
    with zipfile.ZipFile(pack) as zin, zipfile.ZipFile(bad, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "weights.pt":
                data = data + b"\x00"
            zout.writestr(info.filename, data)
    with pytest.raises(ValueError, match="权重哈希"):
        import_modelpack(bad, tmp_path / "registry2")


def test_models_md_append(tmp_path):
    md = tmp_path / "MODELS.md"
    md.write_text("| a | b |\n|---|---|\n", encoding="utf-8")
    card = {
        "model_id": "param_head-v0001",
        "registered_at": "2026-07-02T10:00:00+00:00",
        "train_sample_count": 48,
        "gate_metrics": {"mae_param": 0.1234},
        "weights_sha256": "ab" * 32,
    }
    append_models_md(card, "晋级", md)
    text = md.read_text(encoding="utf-8")
    assert "param_head-v0001" in text and "0.1234" in text and "晋级" in text
    assert json.dumps(card)                                  # card 未被修改仍可序列化

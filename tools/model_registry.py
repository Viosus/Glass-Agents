"""模型版本登记与晋级/激活门（自我迭代的"版本闸"，规则 > AI 的延伸）。

三种事件（都追加进 models/registry/registry.jsonl，机器可读账本）：
- register：训练产出登记为候选（含契约指纹与门指标快照）；
- promote：中心侧晋级（第一重门）——gate.passed 且样本量达标 且 相对当前版本不回归；
- activate：炉侧激活（第二重门）——导入模型包后用**本地留存验证样本**再过一次门才生效，
  防"中心数据分布 ≠ 本炉分布"的静默回归。
人读账本 MODELS.md（入 git）：只有指标与哈希，无配方数值，可公开。
判据参数（min_train_samples / gate.regression_tolerance）读 config/training.yaml，禁硬编码。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tools.eval_gate import GateResult
from training.targets import load_training_config

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = _ROOT / "models" / "registry"
DEFAULT_MODELS_MD = _ROOT / "MODELS.md"
_LEDGER_NAME = "registry.jsonl"

# 候选登记必须携带的元数据键（缺任何一个都拒绝登记——契约指纹是跨机分发的安全带）
_REQUIRED_META = ("feature_schema_sha256", "delta_fields_sha256", "train_sample_count")


def _ledger_path(registry_dir: Path) -> Path:
    """账本文件路径。"""
    return Path(registry_dir) / _LEDGER_NAME


def append_ledger(entry: dict, ledger_path: Path) -> None:
    """追加一条账本事件（jsonl，一行一事件，自动补 UTC 时间戳）。"""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_ledger(registry_dir: Path = DEFAULT_REGISTRY) -> list[dict]:
    """读回全部账本事件（无账本返回空列表）。"""
    p = _ledger_path(registry_dir)
    if not p.exists():
        return []
    entries = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _next_model_id(registry_dir: Path) -> str:
    """下一个版本号：param_head-vNNNN（扫描已有目录取最大 +1）。"""
    registry_dir = Path(registry_dir)
    max_n = 0
    if registry_dir.exists():
        for d in registry_dir.iterdir():
            if d.is_dir() and d.name.startswith("param_head-v"):
                suffix = d.name.removeprefix("param_head-v")
                if suffix.isdigit():
                    max_n = max(max_n, int(suffix))
    return f"param_head-v{max_n + 1:04d}"


def register_candidate(weights_path: Path, meta: dict, registry_dir: Path = DEFAULT_REGISTRY) -> str:
    """把训练产出登记为候选版本：分配 model_id，存 weights + model_card.json，记账。

    meta 至少含契约指纹与样本量（train_param_head.save_checkpoint 的 meta.json 即满足）；
    缺键直接拒绝——没有指纹的模型不允许进入分发链路。
    """
    missing = [k for k in _REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"候选登记缺元数据键: {missing}（用 train_param_head 的 meta.json）")

    model_id = _next_model_id(registry_dir)
    dest = Path(registry_dir) / model_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, dest / "weights.pt")
    card = {
        "model_id": model_id,
        "weights_sha256": hashlib.sha256((dest / "weights.pt").read_bytes()).hexdigest(),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    (dest / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    append_ledger(
        {"event": "register", "model_id": model_id, "train_sample_count": meta["train_sample_count"]},
        _ledger_path(registry_dir),
    )
    return model_id


def load_model_card(model_id: str, registry_dir: Path = DEFAULT_REGISTRY) -> dict:
    """读某版本的 model_card.json（不存在抛 FileNotFoundError）。"""
    p = Path(registry_dir) / model_id / "model_card.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _last_event(registry_dir: Path, event: str) -> dict | None:
    """账本中最后一条指定类型事件。"""
    for entry in reversed(read_ledger(registry_dir)):
        if entry.get("event") == event:
            return entry
    return None


def current_promoted(registry_dir: Path = DEFAULT_REGISTRY) -> str | None:
    """当前已晋级版本（中心侧视角）；从未晋级返回 None。"""
    e = _last_event(registry_dir, "promote")
    return str(e["model_id"]) if e else None


def current_active(registry_dir: Path = DEFAULT_REGISTRY) -> str | None:
    """当前已激活版本（炉侧视角，过了第二重门）；从未激活返回 None。"""
    e = _last_event(registry_dir, "activate")
    return str(e["model_id"]) if e else None


def promote(
    model_id: str,
    gate: GateResult,
    registry_dir: Path = DEFAULT_REGISTRY,
    *,
    config: dict | None = None,
) -> tuple[bool, str]:
    """第一重门（中心侧晋级）。返回 (是否晋级, 理由)；结果如实记账，不静默。

    判据：gate.passed 且 训练样本量 ≥ min_train_samples 且
    相对当前已晋级版本 MAE 不回归（new ≤ prev × (1 + regression_tolerance)）；首版无前任则免比。
    """
    cfg = config if config is not None else load_training_config()
    card = load_model_card(model_id, registry_dir)
    new_mae = float(gate.metrics.get("mae_param", float("nan")))

    reason = ""
    ok = True
    if not gate.passed:
        ok, reason = False, f"版本门未过: {'; '.join(gate.details) or 'gate.passed=False'}"
    elif int(card.get("train_sample_count", 0)) < int(cfg.get("min_train_samples", 30)):
        ok, reason = False, (
            f"样本量不足: {card.get('train_sample_count')} < min_train_samples={cfg.get('min_train_samples')}"
        )
    else:
        prev = _last_event(registry_dir, "promote")
        if prev is not None and prev.get("mae_param") is not None:
            tol = float((cfg.get("gate") or {}).get("regression_tolerance", 0.05))
            prev_mae = float(prev["mae_param"])
            if new_mae > prev_mae * (1.0 + tol):
                ok, reason = False, (
                    f"相对 {prev['model_id']} 回归: MAE {new_mae:.4f} > {prev_mae:.4f}×(1+{tol})"
                )
    if ok:
        reason = "晋级"

    append_ledger(
        {
            "event": "promote" if ok else "promote_rejected",
            "model_id": model_id,
            "mae_param": None if new_mae != new_mae else new_mae,  # NaN → None
            "reason": reason,
        },
        _ledger_path(registry_dir),
    )
    return ok, reason


def activate(model_id: str, gate: GateResult, registry_dir: Path = DEFAULT_REGISTRY) -> tuple[bool, str]:
    """第二重门（炉侧激活）：本地留存验证样本的门结果 passed 才激活；结果如实记账。"""
    ok = gate.passed
    reason = "本地验证通过，激活" if ok else f"本地验证未过: {'; '.join(gate.details) or 'gate.passed=False'}"
    append_ledger(
        {
            "event": "activate" if ok else "activate_rejected",
            "model_id": model_id,
            "mae_param": gate.metrics.get("mae_param"),
            "reason": reason,
        },
        _ledger_path(registry_dir),
    )
    return ok, reason


def append_models_md(card: dict, status: str, path: Path = DEFAULT_MODELS_MD) -> None:
    """人读账本 MODELS.md 追加一行（只有指标与哈希，无配方数值，可入 git）。"""
    mae = (card.get("gate_metrics") or {}).get("mae_param")
    line = (
        f"| {card.get('model_id')} | {card.get('registered_at', '')[:10]} "
        f"| {card.get('train_sample_count')} | {'' if mae is None else f'{mae:.4f}'} "
        f"| `{str(card.get('weights_sha256', ''))[:12]}…` | {status} |\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

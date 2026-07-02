"""同步 CLI：数据包/模型包的导出、导入、投递、激活与状态看板（不联网也完整可用）。

子命令：
  export-data    本炉归档 → 数据包（默认落 data/outbox/）
  import-data    数据包 → 本地归档（幂等去重；冲突上报不覆盖）
  export-model   注册表版本目录 → 模型包
  import-model   模型包 → 注册表候选（校验哈希与契约指纹）
  activate-model 用本地留存验证样本过第二重门，通过才激活
  push / pull    经文件投递通道（config/sync.yaml drop_dir）收发包
  status         归档按炉/按桶计数 + 当前晋级/激活版本

用法示例：& "D:\\Glass Agents\\.venv\\Scripts\\python.exe" tools\\sync_cli.py export-data --furnace-id F1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 支持直接 `python tools/sync_cli.py`

import torch  # noqa: E402

from schemas.archive import load_all  # noqa: E402
from schemas.bucketing import bucket_table  # noqa: E402
from sync.datapack import export_datapack, export_modelpack, import_datapack, import_modelpack  # noqa: E402
from sync.transport import FileDropTransport  # noqa: E402
from tools.eval_gate import evaluate_param_gate  # noqa: E402
from tools.model_registry import (  # noqa: E402
    DEFAULT_REGISTRY,
    activate,
    current_active,
    current_promoted,
)
from training.dataset import time_ordered_split  # noqa: E402
from training.features import feature_dim  # noqa: E402
from training.model import MultiHeadCore  # noqa: E402
from training.targets import PARAM_TARGET_FIELDS, build_param_training_set, load_training_config  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_ARCHIVE = _ROOT / "data" / "archive"
_OUTBOX = _ROOT / "data" / "outbox"
_INBOX = _ROOT / "data" / "inbox"


def _load_param_head(weights: Path) -> MultiHeadCore:
    """从权重文件恢复参数头模型（结构由当前代码契约决定）。"""
    model = MultiHeadCore(feature_dim(), param_dim=len(PARAM_TARGET_FIELDS))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model


def cmd_export_data(args: argparse.Namespace) -> int:
    """导出本炉数据包到 outbox。"""
    pack = export_datapack(args.archive, args.furnace_id, args.out, include_images=args.include_images)
    print(f"已导出数据包: {pack}")
    return 0


def cmd_import_data(args: argparse.Namespace) -> int:
    """导入数据包到本地归档，打印报告。"""
    report = import_datapack(args.pack, args.archive)
    print(f"新增 {report.added}  重复跳过 {report.skipped_duplicates}")
    for c in report.conflicts:
        print(f"  ⚠ 冲突(未覆盖): {c}")
    for r in report.rejected:
        print(f"  ✗ 拒收: {r}")
    return 0 if not report.rejected else 1


def cmd_export_model(args: argparse.Namespace) -> int:
    """把注册表中某版本打成模型包。"""
    pack = export_modelpack(Path(args.registry) / args.model_id, args.out)
    print(f"已导出模型包: {pack}")
    return 0


def cmd_import_model(args: argparse.Namespace) -> int:
    """导入模型包为候选（第二重门 activate-model 通过后才生效）。"""
    manifest = import_modelpack(args.pack, args.registry)
    print(
        f"已导入候选 {manifest.model_id}（中心侧门 passed={manifest.gate_passed}，"
        f"样本量 {manifest.train_sample_count}）"
    )
    print("下一步：activate-model 用本地留存验证样本过第二重门后才激活。")
    return 0


def cmd_activate_model(args: argparse.Namespace) -> int:
    """第二重门：本地留存验证样本（时间序尾部 val+test 段）跑参数门，通过才激活。"""
    weights = Path(args.registry) / args.model_id / "weights.pt"
    if not weights.exists():
        print(f"注册表中无该版本权重: {weights}（先 import-model）")
        return 1
    if not Path(args.archive).exists():
        print(f"本地归档不存在: {args.archive}——无留存验证样本，拒绝盲激活")
        return 1

    ts = build_param_training_set(load_all(Path(args.archive)))
    n = ts.features.shape[0]
    if n == 0:
        print("本地无可用真值样本（is_ground_truth+baseline），拒绝盲激活")
        return 1
    cfg = load_training_config()
    _, va, te = time_ordered_split(n, float(cfg.get("val_frac", 0.2)), float(cfg.get("test_frac", 0.2)))
    holdout = va + te
    model = _load_param_head(weights)
    gate = evaluate_param_gate(
        model, ts.features[holdout], ts.deltas[holdout], [ts.baselines[i] for i in holdout]
    )
    ok, reason = activate(args.model_id, gate, args.registry)
    print(f"第二重门: passed={gate.passed} MAE={gate.metrics['mae_param']:.4f} → {reason}")
    for d in gate.details:
        print(f"  - {d}")
    return 0 if ok else 1


def cmd_push(args: argparse.Namespace) -> int:
    """经文件投递通道送出一个包。"""
    dest = FileDropTransport.from_config().push(args.pack)
    print(f"已投递: {dest}")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """从文件投递通道收包进 inbox。"""
    received = FileDropTransport.from_config().pull(args.dest)
    if not received:
        print("无新包")
    for p in received:
        print(f"收到: {p}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """状态看板：归档按炉/按桶计数、outbox/inbox、当前晋级/激活版本。"""
    archive = Path(args.archive)
    if archive.exists():
        samples = load_all(archive)
        by_furnace: dict[str, int] = {}
        for s in samples:
            by_furnace[s.furnace_id] = by_furnace.get(s.furnace_id, 0) + 1
        per_furnace = ", ".join(f"{k}={v}" for k, v in sorted(by_furnace.items())) or "无"
        print(f"归档样本 {len(samples)}；按炉: {per_furnace}")
        for bucket, total, gt in bucket_table(samples):
            print(f"  {bucket}: 共 {total}，真值 {gt}")
    else:
        print(f"归档目录不存在: {archive}")
    for name, box in (("outbox", _OUTBOX), ("inbox", _INBOX)):
        packs = sorted(box.glob("*.zip")) if box.exists() else []
        print(f"{name}: {len(packs)} 个包" + ("".join(f"\n  {p.name}" for p in packs)))
    print(f"当前晋级版本: {current_promoted(args.registry) or '无'}")
    print(f"当前激活版本: {current_active(args.registry) or '无'}")
    return 0


def main() -> int:
    """CLI 入口：子命令分发。"""
    ap = argparse.ArgumentParser(description="数据包/模型包同步（文件投递通道；云通道待评审）")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export-data", help="本炉归档 → 数据包")
    p.add_argument("--furnace-id", required=True)
    p.add_argument("--archive", type=Path, default=_ARCHIVE)
    p.add_argument("--out", type=Path, default=_OUTBOX)
    p.add_argument("--include-images", action="store_true")
    p.set_defaults(func=cmd_export_data)

    p = sub.add_parser("import-data", help="数据包 → 本地归档")
    p.add_argument("pack", type=Path)
    p.add_argument("--archive", type=Path, default=_ARCHIVE)
    p.set_defaults(func=cmd_import_data)

    p = sub.add_parser("export-model", help="注册表版本 → 模型包")
    p.add_argument("model_id")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--out", type=Path, default=_OUTBOX)
    p.set_defaults(func=cmd_export_model)

    p = sub.add_parser("import-model", help="模型包 → 注册表候选")
    p.add_argument("pack", type=Path)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.set_defaults(func=cmd_import_model)

    p = sub.add_parser("activate-model", help="本地留存验证过第二重门后激活")
    p.add_argument("model_id")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--archive", type=Path, default=_ARCHIVE)
    p.set_defaults(func=cmd_activate_model)

    p = sub.add_parser("push", help="投递一个包（drop_dir 见 config/sync.yaml）")
    p.add_argument("pack", type=Path)
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("pull", help="从投递目录收包进 inbox")
    p.add_argument("--dest", type=Path, default=_INBOX)
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("status", help="归档/包/版本状态看板")
    p.add_argument("--archive", type=Path, default=_ARCHIVE)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    sys.exit(main())

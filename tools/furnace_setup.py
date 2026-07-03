"""炉子初始化向导：问答式录入炉体身份 → 写 config/furnaces.yaml + config/sync.yaml。

设计：
- 一次性低频操作走专门入口（不走对话壳）；回车跳过 = 诚实缺值（写 TODO(plant)/留空），带类型校验循环重问。
- 两个 yaml 均**整文件再生成**（头注释模板在本文件内=单一来源；pyyaml 原地编辑会丢注释）；
  furnaces.yaml 保留其他炉条目原样，sync.yaml 的 cloud 段三键原样保留（含 TODO 字符串）。
- 投产日期/上次大修是**炉龄特征的原料**（training/features.py v2），务必尽量填。

用法：
  交互：  & "D:\\Glass Agents\\.venv\\Scripts\\python.exe" tools\\furnace_setup.py
  非交互：& ... tools\\furnace_setup.py --from-json 炉信息.json [--force] [--skip-sync]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 支持直接 `python tools/furnace_setup.py`

_ROOT = Path(__file__).resolve().parents[1]
_FURNACES = _ROOT / "config" / "furnaces.yaml"
_SYNC = _ROOT / "config" / "sync.yaml"
_TODO = "TODO(plant)"

# 可缺省字段（缺 → 写 TODO(plant) 占位，保持模板样貌；nameplate 缺 → {}）
_OPTIONAL_KEYS = ("zone_count", "zone_layout", "fan_count", "commissioning_date", "last_overhaul_date")

_FURNACES_HEADER = """\
# 炉体登记表：每台目标钢化炉一条（跨炉迁移的数据地基）。
# 拿到铭牌照片/现场确认即填；未知项保持 TODO(plant)，代码按缺失处理，绝不猜测。
# 由 schemas/furnace.py::load_furnace_registry 每次调用时按需读取（改这里即时生效）。
# 推荐用初始化向导录入：& .venv\\Scripts\\python.exe tools\\furnace_setup.py
#
# 字段：
#   furnace_id          炉子唯一标识（导入标注表时 --furnace-id 必须与此对得上）
#   zone_count          分区数（铭牌/现场）
#   zone_layout         分区布局描述，如 "2x6"（真实二维邻接关系另行 TODO(plant)）
#   fan_count           风机数
#   nameplate           铭牌原文键值对（型号/厂家/额定功率等，原样抄录）
#   commissioning_date  投产日期（ISO，台账/铭牌——炉龄特征的原料，务必填）
#   last_overhaul_date  上次大修日期（ISO——距大修天数特征的原料）
#   recorded_at         快照采集时间（ISO 格式）

"""

_SYNC_HEADER = """\
# 同步配置（多炉云共享·中心汇聚）。缺值一律 TODO(plant)，代码遇 TODO 明确报错，不猜。
# 机密边界：数据包含配方 → 只经工厂自有通道上自家云，绝不入 git（data/outbox|inbox 已 gitignore）。
# furnace_id / drop_dir 可用初始化向导写入：tools/furnace_setup.py

"""


def _load_raw(path: Path) -> dict:
    """原样读 yaml（保留 TODO 字符串等原值）；不存在返回空 dict。"""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_furnace_info(raw: dict) -> dict:
    """校验并规范化一条炉子信息（缺值留 None，不猜；坏值抛 ValueError）。"""
    fid = str(raw.get("furnace_id") or "").strip()
    if not fid or fid.startswith(_TODO):
        raise ValueError("furnace_id 必填（如 F1/F2，命名规则见信息需求清单 D6）")
    entry: dict = {"furnace_id": fid}
    for key in ("zone_count", "fan_count"):
        v = raw.get(key)
        if v is not None:
            n = int(v)
            if n <= 0:
                raise ValueError(f"{key} 须为正整数: {v!r}")
            entry[key] = n
    if raw.get("zone_layout"):
        entry["zone_layout"] = str(raw["zone_layout"])
    for key in ("commissioning_date", "last_overhaul_date"):
        v = raw.get(key)
        if v is not None:
            entry[key] = v if isinstance(v, date) else date.fromisoformat(str(v))
    nameplate = raw.get("nameplate") or {}
    if not isinstance(nameplate, dict):
        raise ValueError("nameplate 须为键值对")
    entry["nameplate"] = {str(k): str(v) for k, v in nameplate.items()}
    return entry


def _entry_for_yaml(entry: dict) -> dict:
    """写盘形态：缺省字段补 TODO(plant) 占位（保持模板样貌），日期转 ISO 字符串。"""
    out: dict = {"furnace_id": entry["furnace_id"]}
    for key in _OPTIONAL_KEYS:
        v = entry.get(key)
        out[key] = v.isoformat() if isinstance(v, date) else (v if v is not None else _TODO)
    out["nameplate"] = entry.get("nameplate") or {}
    out["recorded_at"] = datetime.now(timezone.utc).isoformat()
    return out


def write_furnaces_yaml(entry: dict, path: Path = _FURNACES, *, force: bool = False) -> None:
    """把本条目并入炉体登记表（其余炉原样保留）；同 id 已存在且未 --force → 拒绝。"""
    raw = _load_raw(path)
    existing: list = [e for e in (raw.get("furnaces") or []) if isinstance(e, dict)]
    ids = [str(e.get("furnace_id")) for e in existing]
    if entry["furnace_id"] in ids and not force:
        raise ValueError(f"炉 {entry['furnace_id']} 已登记（--force 才覆盖）")
    merged = [e for e in existing if str(e.get("furnace_id")) != entry["furnace_id"]]
    merged.append(_entry_for_yaml(entry))
    text = _FURNACES_HEADER + yaml.safe_dump({"furnaces": merged}, allow_unicode=True, sort_keys=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def write_sync_yaml(furnace_id: str, drop_dir: str | None, path: Path = _SYNC) -> None:
    """更新本机 sync.yaml 的 furnace_id / drop_dir；cloud 段三键原样保留（含 TODO 字符串）。"""
    raw = _load_raw(path)
    cloud = raw.get("cloud") or {"provider": _TODO, "endpoint": _TODO, "auth_ref": _TODO}
    data = {
        "furnace_id": furnace_id,
        "drop_dir": drop_dir if drop_dir else raw.get("drop_dir", _TODO),
        "cloud": cloud,
    }
    text = _SYNC_HEADER + yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _ask(prompt: str, *, required: bool = False, parser=None) -> object | None:
    """交互问一项：回车跳过（诚实缺值）；解析失败循环重问。"""
    while True:
        text = input(prompt).strip()
        if not text:
            if required:
                print("  该项必填。")
                continue
            return None
        if parser is None:
            return text
        try:
            return parser(text)
        except (ValueError, TypeError) as e:
            print(f"  无效: {e}")


def collect_interactive() -> tuple[dict, str | None]:
    """逐项问答收集炉子信息 + drop_dir。返回 (炉信息 dict, drop_dir)。"""
    print("=== 炉子初始化向导（回车跳过=如实缺值，之后可再补）===")
    info: dict = {"furnace_id": _ask("炉号（必填，如 F1）: ", required=True)}
    info["zone_count"] = _ask("分区数: ", parser=int)
    info["zone_layout"] = _ask("分区布局（如 2x6）: ")
    info["fan_count"] = _ask("风机数: ", parser=int)
    info["commissioning_date"] = _ask("投产日期（YYYY-MM-DD，炉龄特征原料）: ", parser=date.fromisoformat)
    info["last_overhaul_date"] = _ask("上次大修日期（YYYY-MM-DD）: ", parser=date.fromisoformat)
    nameplate: dict[str, str] = {}
    model = _ask("炉型号: ")
    maker = _ask("制造厂家: ")
    if model:
        nameplate["model"] = str(model)
    if maker:
        nameplate["manufacturer"] = str(maker)
    info["nameplate"] = nameplate
    drop_dir = _ask("同步投递目录 drop_dir（共享盘/U盘路径）: ")
    return info, (str(drop_dir) if drop_dir else None)


def _checklist(furnace_id: str) -> str:
    """写盘后的上炉检查清单（指路，不重复正文）。"""
    return (
        f"\n✅ 炉 {furnace_id} 已登记。接下来（上炉流程阶段1，详见 docs/自我迭代与上炉流程.md）：\n"
        "  1. 核对/补齐本炉安全限值 config/thresholds.yaml（信息需求清单 A3–A6）\n"
        "  2. 落实图像↔标注对齐约定与检测软件元数据（清单 B6–B8）\n"
        "  3. 导入本炉基准配方库（清单 B2）\n"
        "  4. 标注表导入用：tools/ingest_annotations.py <表.csv> --furnace-id " + furnace_id
    )


def main() -> int:
    """CLI 入口：交互或 --from-json 收集 → 校验 → 写两个 yaml → 打印检查清单。"""
    ap = argparse.ArgumentParser(description="炉子初始化向导（写 furnaces.yaml + sync.yaml，缺值不猜）")
    ap.add_argument(
        "--from-json", type=Path, default=None, help="非交互：从 JSON 读炉信息（键=FurnaceConfig 字段+drop_dir）"
    )
    ap.add_argument("--furnaces", type=Path, default=_FURNACES)
    ap.add_argument("--sync", type=Path, default=_SYNC)
    ap.add_argument("--force", action="store_true", help="同 id 已存在时允许覆盖")
    ap.add_argument("--skip-sync", action="store_true", help="不改 sync.yaml（中心侧代登记他炉时用）")
    args = ap.parse_args()

    if args.from_json is not None:
        raw = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        drop_dir = raw.pop("drop_dir", None)
        info = raw
    else:
        info, drop_dir = collect_interactive()

    entry = parse_furnace_info(info)
    write_furnaces_yaml(entry, args.furnaces, force=args.force)
    if not args.skip_sync:
        write_sync_yaml(entry["furnace_id"], drop_dir, args.sync)
    print(_checklist(entry["furnace_id"]))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    sys.exit(main())

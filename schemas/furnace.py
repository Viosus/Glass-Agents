"""炉体配置快照：铭牌与结构信息（跨炉迁移的数据地基）。

原则（铁律 #5/#8）：未知项一律 None，**绝不猜测**；登记表 config/furnaces.yaml 中
值为 TODO(plant) 的炉子跳过（不构造假配置）。样本入库时把当时的炉体配置快照
冻结进 ArchiveSample.furnace_config，保证多炉数据汇聚后仍可按炉体特征建模。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "furnaces.yaml"

# 缺省炉体身份：旧数据/未登记时的诚实占位（非安全限值，故不用 TODO(plant) 字串）
UNKNOWN_FURNACE_ID = "unknown"


class FurnaceConfig(BaseModel):
    """一台钢化炉的配置快照（铭牌/分区/风机）。未知项 None，不猜。"""

    model_config = ConfigDict(extra="forbid")

    furnace_id: str
    zone_count: int | None = Field(default=None, gt=0)       # 分区数（铭牌/现场确认）
    zone_layout: str | None = None                           # 布局描述如 "2x6"；二维邻接 TODO(plant)
    fan_count: int | None = Field(default=None, gt=0)        # 风机数
    nameplate: dict[str, str] = Field(default_factory=dict)  # 铭牌原文键值对（原样记录）
    commissioning_date: date | None = None                   # 投产日期（炉龄特征的原料；台账/铭牌）
    last_overhaul_date: date | None = None                   # 上次大修日期（距大修天数特征的原料）
    recorded_at: datetime | None = None                      # 快照采集时间


def _is_todo(value: object) -> bool:
    """判断 yaml 值是否为 TODO(plant) 占位（未填真值）。"""
    return isinstance(value, str) and value.strip().startswith("TODO(plant)")


def load_furnace_registry(path: Path | None = None) -> dict[str, FurnaceConfig]:
    """读 config/furnaces.yaml 登记表 → {furnace_id: FurnaceConfig}。

    每次调用按需读取（改 yaml 即时生效）。字段值为 TODO(plant) 的按缺失（None）处理；
    整条炉子无 furnace_id 的跳过。文件不存在返回空表（不猜、不报假配置）。
    """
    p = Path(path) if path is not None else _CONFIG
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    registry: dict[str, FurnaceConfig] = {}
    for entry in raw.get("furnaces") or []:
        if not isinstance(entry, dict):
            continue
        fid = entry.get("furnace_id")
        if not fid or _is_todo(fid):
            continue  # 无身份的条目不登记
        cleaned: dict[str, object] = {"furnace_id": str(fid)}
        optional_keys = (
            "zone_count", "zone_layout", "fan_count",
            "commissioning_date", "last_overhaul_date", "recorded_at",
        )
        for key in optional_keys:
            v = entry.get(key)
            if v is not None and not _is_todo(v):
                cleaned[key] = v
        nameplate = entry.get("nameplate")
        if isinstance(nameplate, dict):
            cleaned["nameplate"] = {str(k): str(v) for k, v in nameplate.items() if not _is_todo(v)}
        registry[str(fid)] = FurnaceConfig(**cleaned)  # type: ignore[arg-type]
    return registry

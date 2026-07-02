"""初始化向导单测：from-json 写双 yaml、读回校验、重复 id 拒绝/覆盖、cloud 段保留。全程 tmp_path。"""

from datetime import date

import pytest
import yaml

from schemas.furnace import load_furnace_registry
from tools.furnace_setup import parse_furnace_info, write_furnaces_yaml, write_sync_yaml

INFO = {
    "furnace_id": "F2",
    "zone_count": 12,
    "zone_layout": "2x6",
    "fan_count": 8,
    "commissioning_date": "2016-07-02",
    "last_overhaul_date": "2026-06-02",
    "nameplate": {"model": "炉型X", "manufacturer": "厂家Y"},
}


def test_parse_and_roundtrip(tmp_path):
    entry = parse_furnace_info(dict(INFO))
    assert entry["commissioning_date"] == date(2016, 7, 2)

    furnaces = tmp_path / "furnaces.yaml"
    write_furnaces_yaml(entry, furnaces)
    reg = load_furnace_registry(furnaces)
    f2 = reg["F2"]
    assert f2.zone_count == 12 and f2.zone_layout == "2x6"
    assert f2.commissioning_date == date(2016, 7, 2)          # 日期读回=炉龄特征的原料
    assert f2.last_overhaul_date == date(2026, 6, 2)
    assert f2.nameplate["model"] == "炉型X"
    assert f2.recorded_at is not None                          # 快照时间自动打


def test_missing_optional_written_as_todo(tmp_path):
    entry = parse_furnace_info({"furnace_id": "F3"})           # 只有必填项
    furnaces = tmp_path / "furnaces.yaml"
    write_furnaces_yaml(entry, furnaces)

    raw = yaml.safe_load(furnaces.read_text(encoding="utf-8"))
    e = raw["furnaces"][0]
    assert e["zone_count"] == "TODO(plant)"                    # 缺值写占位，不猜
    reg = load_furnace_registry(furnaces)
    assert reg["F3"].zone_count is None                        # 读回按缺失


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="furnace_id 必填"):
        parse_furnace_info({})
    with pytest.raises(ValueError, match="正整数"):
        parse_furnace_info({"furnace_id": "F9", "zone_count": -2})
    with pytest.raises(ValueError):
        parse_furnace_info({"furnace_id": "F9", "commissioning_date": "不是日期"})


def test_duplicate_id_needs_force_and_preserves_others(tmp_path):
    furnaces = tmp_path / "furnaces.yaml"
    write_furnaces_yaml(parse_furnace_info({"furnace_id": "F1", "zone_count": 4}), furnaces)
    write_furnaces_yaml(parse_furnace_info(dict(INFO)), furnaces)          # F2 并入

    with pytest.raises(ValueError, match="--force"):
        write_furnaces_yaml(parse_furnace_info({"furnace_id": "F2"}), furnaces)

    write_furnaces_yaml(parse_furnace_info({"furnace_id": "F2", "zone_count": 16}), furnaces, force=True)
    reg = load_furnace_registry(furnaces)
    assert reg["F2"].zone_count == 16                          # 覆盖生效
    assert reg["F1"].zone_count == 4                           # 其他炉原样保留


def test_sync_yaml_updates_identity_keeps_cloud(tmp_path):
    sync = tmp_path / "sync.yaml"
    sync.write_text(
        "furnace_id: TODO(plant)\ndrop_dir: TODO(plant)\n"
        "cloud:\n  provider: TODO(plant)\n  endpoint: TODO(plant)\n  auth_ref: 已填的值\n",
        encoding="utf-8",
    )
    write_sync_yaml("F2", r"\\share\glass_drop", sync)
    raw = yaml.safe_load(sync.read_text(encoding="utf-8"))
    assert raw["furnace_id"] == "F2"
    assert raw["drop_dir"] == r"\\share\glass_drop"
    assert raw["cloud"]["auth_ref"] == "已填的值"               # cloud 段原样保留（含已填值）
    assert raw["cloud"]["provider"] == "TODO(plant)"

    write_sync_yaml("F2", None, sync)                          # 不给 drop_dir → 保留现值
    raw2 = yaml.safe_load(sync.read_text(encoding="utf-8"))
    assert raw2["drop_dir"] == r"\\share\glass_drop"

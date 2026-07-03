"""数据包单测：往返/幂等/冲突不覆盖/篡改拒收/版本拒收/按炉过滤 + 文件投递通道。全程 tmp_path。"""

import json
import zipfile
from datetime import datetime, timedelta

import pytest

from schemas.archive import ArchiveSample, MetricRecord, load_all, write_sample
from schemas.process_params import ProcessParams
from sync.datapack import export_datapack, import_datapack
from sync.transport import FileDropTransport, make_transport


def make_params() -> ProcessParams:
    return ProcessParams(
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


def make_sample(sample_id: str, furnace_id: str = "F1", minute: int = 0, **kw) -> ArchiveSample:
    return ArchiveSample(
        sample_id=sample_id,
        created_at=datetime(2026, 7, 2, 9, 0, 0) + timedelta(minutes=minute),
        source="line1/早班",
        furnace_id=furnace_id,
        thickness_mm=6.0,
        glass_type="clear",
        quality_mode="high_quality",
        is_ground_truth=True,
        params=make_params(),
        baseline_params=make_params(),
        metrics=MetricRecord(x0_95_nm=80.0),
        **kw,
    )


@pytest.fixture
def archive(tmp_path):
    """3 条 F1 + 1 条 F2 的源归档。"""
    root = tmp_path / "src_archive"
    for i in range(3):
        write_sample(make_sample(f"a{i:03d}", minute=i), root)
    write_sample(make_sample("b000", furnace_id="F2"), root)
    return root


def rebuild_pack(src, dst, transform):
    """复制 zip 并对指定成员做字节变换（构造篡改/版本包）。"""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            zout.writestr(info.filename, transform(info.filename, zin.read(info.filename)))


# --------------------------- 导出 --------------------------- #
def test_export_filters_by_furnace(archive, tmp_path):
    pack = export_datapack(archive, "F1", tmp_path / "out")
    with zipfile.ZipFile(pack) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        names = zf.namelist()
    assert len(manifest["samples"]) == 3                    # F2 的样本不进 F1 的包
    assert "samples/b000.json" not in names
    assert manifest["furnace_id"] == "F1" and manifest["pack_kind"] == "archive_samples"


def test_export_no_samples_raises(archive, tmp_path):
    with pytest.raises(ValueError, match="无可导出"):
        export_datapack(archive, "F9", tmp_path / "out")


# --------------------------- 导入：往返 / 幂等 / 冲突 --------------------------- #
def test_roundtrip_and_idempotent(archive, tmp_path):
    pack = export_datapack(archive, "F1", tmp_path / "out")
    dest = tmp_path / "dest_archive"

    rep1 = import_datapack(pack, dest)
    assert rep1.added == 3 and not rep1.rejected and not rep1.conflicts
    assert {s.sample_id for s in load_all(dest)} == {"a000", "a001", "a002"}

    rep2 = import_datapack(pack, dest)                      # 重复导入 → 幂等
    assert rep2.added == 0 and rep2.skipped_duplicates == 3


def test_conflict_reported_not_overwritten(archive, tmp_path):
    pack = export_datapack(archive, "F1", tmp_path / "out")
    dest = tmp_path / "dest_archive"
    import_datapack(pack, dest)

    local = make_sample("a000", rationale="本地已被人工修订")   # 同 id 异内容
    write_sample(local, dest)
    rep = import_datapack(pack, dest)
    assert any("a000" in c for c in rep.conflicts)
    loaded = {s.sample_id: s for s in load_all(dest)}
    assert loaded["a000"].rationale == "本地已被人工修订"      # 绝不静默覆盖


# --------------------------- 导入：篡改 / 版本 --------------------------- #
def test_tampered_sample_rejected(archive, tmp_path):
    pack = export_datapack(archive, "F1", tmp_path / "out")
    bad = tmp_path / "tampered.zip"

    def tamper(name, data):
        if name == "samples/a001.json":
            return data.replace(b"line1", b"lineX")         # 内容变但 manifest 哈希没变
        return data

    rebuild_pack(pack, bad, tamper)
    rep = import_datapack(bad, tmp_path / "dest")
    assert rep.added == 2
    assert any("a001" in r and "哈希" in r for r in rep.rejected)


def test_newer_schema_version_rejected(archive, tmp_path):
    pack = export_datapack(archive, "F1", tmp_path / "out")
    bad = tmp_path / "future.zip"

    def bump(name, data):
        if name == "manifest.json":
            m = json.loads(data)
            m["archive_schema_version"] += 1
            return json.dumps(m).encode("utf-8")
        return data

    rebuild_pack(pack, bad, bump)
    with pytest.raises(ValueError, match="升级代码"):
        import_datapack(bad, tmp_path / "dest")


# --------------------------- 文件投递通道 --------------------------- #
def test_file_drop_push_pull(tmp_path):
    drop = tmp_path / "drop"
    transport = FileDropTransport(drop)
    pack = tmp_path / "x.zip"
    pack.write_bytes(b"zipbytes")

    dest_desc = transport.push(pack)
    assert (drop / "x.zip").exists() and "x.zip" in dest_desc

    inbox = tmp_path / "inbox"
    assert [p.name for p in transport.pull(inbox)] == ["x.zip"]
    assert transport.pull(inbox) == []                       # 已存在 → 不重复收


def test_transport_todo_config_refuses(tmp_path):
    with pytest.raises(ValueError, match="TODO\\(plant\\)"):
        FileDropTransport.from_config({"drop_dir": "TODO(plant)"})
    with pytest.raises(NotImplementedError, match="评审"):
        make_transport("cloud")
    with pytest.raises(ValueError, match="未知传输类型"):
        make_transport("carrier_pigeon")

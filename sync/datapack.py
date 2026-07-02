"""数据包/模型包的导出与导入（zip，stdlib，零联网——不联网也完整可用，U盘/内网共享即可跑通）。

流程（中心汇聚）：各炉 export_datapack → 自有通道送中心 → 中心 import_datapack 汇聚重训
→ export_modelpack 分发 → 各炉 import_modelpack（校验契约指纹）→ 本地再过一次门才激活（双重门）。
防篡改：逐样本/权重 sha256 与 manifest 比对，不符拒收；同键异内容绝不静默覆盖。
"""

from __future__ import annotations

import hashlib
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from schemas.archive import ARCHIVE_SCHEMA_VERSION, ArchiveSample
from schemas.datapack import DataPackManifest, ImportReport, ModelPackManifest, SampleDigest

_MANIFEST_NAME = "manifest.json"


def _sha256_bytes(data: bytes) -> str:
    """字节内容 → sha256 十六进制。"""
    return hashlib.sha256(data).hexdigest()


def _producer_tag() -> str:
    """产包端标识：项目名 + 归档 schema 版本（排障时判断产包端代码代际）。"""
    return f"glass-agents/archive-v{ARCHIVE_SCHEMA_VERSION}"


def export_datapack(
    archive_root: Path,
    furnace_id: str,
    out_dir: Path,
    *,
    include_images: bool = False,
    images_root: Path | None = None,
) -> Path:
    """把 archive_root 下属于 furnace_id 的样本打成数据包。返回包路径。

    只导出该炉样本（按样本内 furnace_id 字段过滤，身份从第一天就带）；
    include_images 时把 stress_image 引用的图像本体一并入包（按 sha256 命名）。
    """
    archive_root = Path(archive_root)
    entries: list[tuple[str, bytes]] = []          # (zip 内路径, 内容)
    digests: list[SampleDigest] = []
    image_files: dict[str, Path] = {}              # sha256 → 磁盘路径

    for p in sorted(archive_root.glob("*.json")):
        raw = p.read_bytes()
        sample = ArchiveSample.model_validate_json(raw.decode("utf-8"))  # 出包前校验
        if sample.furnace_id != furnace_id:
            continue
        digests.append(
            SampleDigest(
                sample_id=sample.sample_id,
                furnace_id=sample.furnace_id,
                content_sha256=_sha256_bytes(raw),
            )
        )
        entries.append((f"samples/{sample.sample_id}.json", raw))
        if include_images and sample.stress_image is not None:
            img = Path(sample.stress_image.path)
            if not img.is_absolute() and images_root is not None:
                img = Path(images_root) / img
            if img.exists():
                image_files[sample.stress_image.sha256] = img

    if not digests:
        raise ValueError(f"归档中没有 furnace_id={furnace_id} 的样本，无可导出")

    manifest = DataPackManifest(
        pack_id=str(uuid.uuid4()),
        furnace_id=furnace_id,
        created_at=datetime.now(timezone.utc),
        archive_schema_version=ARCHIVE_SCHEMA_VERSION,
        samples=digests,
        includes_images=bool(image_files),
        producer=_producer_tag(),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = manifest.created_at.strftime("%Y%m%dT%H%M%SZ")
    pack_path = out_dir / f"datapack_{furnace_id}_{stamp}.zip"
    with zipfile.ZipFile(pack_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_MANIFEST_NAME, manifest.model_dump_json(indent=2))
        for name, raw in entries:
            zf.writestr(name, raw)
        for sha, img in image_files.items():
            zf.writestr(f"images/{sha}{img.suffix}", img.read_bytes())
    return pack_path


def import_datapack(
    pack_path: Path,
    archive_root: Path,
    *,
    images_root: Path | None = None,
) -> ImportReport:
    """导入数据包：逐样本校验+哈希核验 → 幂等去重 → 冲突上报（绝不静默覆盖）→ 落库。

    包 schema 版本高于本地代码 → 整包拒收（ValueError，提示升级代码）。
    """
    archive_root = Path(archive_root)
    report = ImportReport()

    with zipfile.ZipFile(pack_path) as zf:
        manifest = DataPackManifest.model_validate_json(zf.read(_MANIFEST_NAME).decode("utf-8"))
        if manifest.archive_schema_version > ARCHIVE_SCHEMA_VERSION:
            raise ValueError(
                f"包 schema v{manifest.archive_schema_version} 高于本地代码 v{ARCHIVE_SCHEMA_VERSION}，"
                "请先升级代码再导入"
            )

        archive_root.mkdir(parents=True, exist_ok=True)
        for digest in manifest.samples:
            member = f"samples/{digest.sample_id}.json"
            try:
                raw = zf.read(member)
            except KeyError:
                report.rejected.append(f"{digest.sample_id}: 包内缺文件 {member}")
                continue
            if _sha256_bytes(raw) != digest.content_sha256:
                report.rejected.append(f"{digest.sample_id}: 内容哈希与 manifest 不符（疑似篡改/损坏）")
                continue
            try:
                sample = ArchiveSample.model_validate_json(raw.decode("utf-8"))
            except ValidationError as e:
                report.rejected.append(f"{digest.sample_id}: 样本校验失败 {e.error_count()} 处")
                continue
            if sample.sample_id != digest.sample_id or sample.furnace_id != digest.furnace_id:
                report.rejected.append(f"{digest.sample_id}: 样本身份与 manifest 摘要不一致")
                continue

            dest = archive_root / f"{sample.sample_id}.json"
            if dest.exists():
                if _sha256_bytes(dest.read_bytes()) == digest.content_sha256:
                    report.skipped_duplicates += 1
                else:
                    report.conflicts.append(f"{sample.sample_id}: 本地已有同名样本且内容不同（人工裁决）")
                continue
            dest.write_bytes(raw)
            report.added += 1

        if manifest.includes_images and images_root is not None:
            img_dir = Path(images_root)
            img_dir.mkdir(parents=True, exist_ok=True)
            for name in zf.namelist():
                if name.startswith("images/") and not name.endswith("/"):
                    target = img_dir / Path(name).name
                    if not target.exists():
                        target.write_bytes(zf.read(name))
    return report


# --------------------------------------------------------------------------- #
# 模型包（中心 → 各炉分发）
# --------------------------------------------------------------------------- #
def export_modelpack(version_dir: Path, out_dir: Path) -> Path:
    """把模型版本目录（weights.pt + model_card.json）打成模型包。返回包路径。

    model_card.json 由 tools.model_registry.register_candidate 生成，
    含 model_id / 契约指纹 / 门指标；这里只做打包与权重哈希封存。
    """
    version_dir = Path(version_dir)
    weights = version_dir / "weights.pt"
    card_path = version_dir / "model_card.json"
    if not weights.exists() or not card_path.exists():
        raise ValueError(f"版本目录不完整（需 weights.pt + model_card.json）：{version_dir}")
    card = json.loads(card_path.read_text(encoding="utf-8"))

    raw_weights = weights.read_bytes()
    manifest = ModelPackManifest(
        pack_id=str(uuid.uuid4()),
        model_id=str(card["model_id"]),
        created_at=datetime.now(timezone.utc),
        weights_sha256=_sha256_bytes(raw_weights),
        feature_schema_sha256=str(card["feature_schema_sha256"]),
        delta_fields_sha256=str(card["delta_fields_sha256"]),
        train_sample_count=int(card.get("train_sample_count", 0)),
        gate_metrics={k: float(v) for k, v in (card.get("gate_metrics") or {}).items()},
        gate_passed=bool(card.get("gate_passed", False)),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_path = out_dir / f"modelpack_{manifest.model_id}.zip"
    with zipfile.ZipFile(pack_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_MANIFEST_NAME, manifest.model_dump_json(indent=2))
        zf.writestr("weights.pt", raw_weights)
        zf.writestr("model_card.json", json.dumps(card, ensure_ascii=False, indent=2))
    return pack_path


def import_modelpack(pack_path: Path, registry_dir: Path) -> ModelPackManifest:
    """导入模型包为**候选**（未激活）：校验权重哈希 + 特征/Δ 契约指纹，不符即拒。

    第二重门在导入之后：用本地留存验证样本过 evaluate_param_gate 才可激活
    （tools.model_registry.activate），防"中心分布 ≠ 本炉分布"的静默回归。
    """
    from training.targets import delta_fields_sha256, feature_schema_sha256

    with zipfile.ZipFile(pack_path) as zf:
        manifest = ModelPackManifest.model_validate_json(zf.read(_MANIFEST_NAME).decode("utf-8"))
        raw_weights = zf.read("weights.pt")
        card_raw = zf.read("model_card.json")

    if _sha256_bytes(raw_weights) != manifest.weights_sha256:
        raise ValueError("权重哈希与 manifest 不符（疑似篡改/损坏），拒收")
    if manifest.feature_schema_sha256 != feature_schema_sha256():
        raise ValueError("特征契约指纹与本地代码不符（FEATURE_NAMES 已变），拒收——先对齐代码版本")
    if manifest.delta_fields_sha256 != delta_fields_sha256():
        raise ValueError("Δ 字段契约指纹与本地代码不符（DELTA_FIELDS 已变），拒收——先对齐代码版本")

    dest = Path(registry_dir) / manifest.model_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "weights.pt").write_bytes(raw_weights)
    (dest / "model_card.json").write_bytes(card_raw)
    return manifest

"""数据包/模型包契约（多炉云共享的文件格式，中心汇聚·文件包先行）。

包 = zip（stdlib，零联网依赖）：manifest.json + samples/*.json（+ 可选 images/）。
版本与防篡改：
- manifest 带 archive_schema_version（样本本体不带版本，包级声明更诚实）；
  导入端代码版本低于包版本 → 整包拒收提示升级。
- 每个样本记 content_sha256；模型包记权重哈希 + 特征/Δ 契约指纹，不匹配拒导入。
机密边界：包内含配方（params/baseline_params）→ 只落 data/outbox|inbox（.gitignore），
只经工厂自有通道上自家云，**绝不入 git**。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_SHA_PATTERN = r"^[0-9a-fA-F]{64}$"


class SampleDigest(BaseModel):
    """包内单样本摘要：去重键 (furnace_id, sample_id) + 内容哈希。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    furnace_id: str
    content_sha256: str = Field(min_length=64, max_length=64, pattern=_SHA_PATTERN)


class DataPackManifest(BaseModel):
    """数据包清单：一台炉的一批归档样本。"""

    model_config = ConfigDict(extra="forbid")

    pack_id: str                                             # uuid4
    pack_kind: Literal["archive_samples"] = "archive_samples"
    furnace_id: str
    created_at: datetime
    archive_schema_version: int                              # = schemas.archive.ARCHIVE_SCHEMA_VERSION
    samples: list[SampleDigest]
    includes_images: bool = False
    producer: str                                            # 产包端标识（代码版本等）


class ModelPackManifest(BaseModel):
    """模型包清单：中心训练产出的一个模型版本（分发给各炉）。"""

    model_config = ConfigDict(extra="forbid")

    pack_id: str
    pack_kind: Literal["model_version"] = "model_version"
    model_id: str                                            # 如 param_head-v0003
    created_at: datetime
    weights_sha256: str = Field(min_length=64, max_length=64, pattern=_SHA_PATTERN)
    feature_schema_sha256: str = Field(min_length=64, max_length=64, pattern=_SHA_PATTERN)
    delta_fields_sha256: str = Field(min_length=64, max_length=64, pattern=_SHA_PATTERN)
    train_sample_count: int = Field(ge=0)
    gate_metrics: dict[str, float] = Field(default_factory=dict)   # 中心侧门指标快照
    gate_passed: bool                                        # 中心侧门结果（第一重门）


class ImportReport(BaseModel):
    """导入报告：幂等/冲突/拒收明细（同键异内容绝不静默覆盖）。"""

    model_config = ConfigDict(extra="forbid")

    added: int = 0
    skipped_duplicates: int = 0
    conflicts: list[str] = Field(default_factory=list)       # 同 (furnace_id, sample_id) 不同内容
    rejected: list[str] = Field(default_factory=list)        # 校验失败/哈希不符

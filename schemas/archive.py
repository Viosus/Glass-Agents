"""归档样本 L5：一炉玻璃的完整记录 + JSON 落库（写入即校验）。

图像以"路径 + sha256 哈希"引用，与结构化数据分离落库（图像本体不进 JSON）。
分桶键：thickness_mm / glass_type / quality_mode。专家复核后置 is_ground_truth=True。
缺失指标/判级一律 None（不杜撰）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.process_params import ProcessParams

Grade = Literal["A", "B", "C"]


class ImageRef(BaseModel):
    """应力斑图像引用：路径 + 内容哈希 + 尺度，与结构化数据分离。"""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    mm_per_px: float = Field(gt=0)


class MetricRecord(BaseModel):
    """三法指标值与判级（缺则 None；CCP 带 is_calibrated）。"""

    model_config = ConfigDict(extra="forbid")

    x0_95_nm: float | None = None
    x0_95_grade: Grade | None = None
    iso_t_pct: float | None = None
    iso_t_grade: Grade | None = None
    ccp_value: float | None = None
    ccp_is_calibrated: bool = False
    ccp_grade: Grade | None = None


class ConstraintSummary(BaseModel):
    """tools.constraints.CheckResult 的归档摘要。"""

    model_config = ConfigDict(extra="forbid")

    within_limits: bool
    blow_up_risk: bool
    gradient_ok: bool
    violations: list[str] = Field(default_factory=list)


class ArchiveSample(BaseModel):
    """归档样本 L5。写入即校验；未知字段拒绝。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    created_at: datetime
    source: str                                              # 产线/班次/teacher 版本等来源

    # ---- 分桶键 ----
    thickness_mm: float = Field(gt=0)
    glass_type: Literal["ultra_clear", "clear"]
    quality_mode: Literal["high_quality", "high_efficiency"]
    is_ground_truth: bool = False                            # 专家复核后 True

    # ---- 图像（路径+哈希，与结构化数据分离）----
    stress_image: ImageRef | None = None

    # ---- 结构化 ----
    params: ProcessParams
    metrics: MetricRecord = Field(default_factory=MetricRecord)
    constraint: ConstraintSummary | None = None

    # ---- outcome 回填（出炉实测）----
    measured_quality_grade: Grade | None = None
    measured_energy_kwh: float | None = Field(default=None, ge=0)
    rationale: str | None = None                             # Teacher 理由/专家备注（归因弱标签）

    @property
    def bucket(self) -> tuple[float, str, str]:
        return (self.thickness_mm, self.glass_type, self.quality_mode)


# --------------------------------------------------------------------------- #
# JSON 落库（data/archive/<sample_id>.json）
# --------------------------------------------------------------------------- #
def write_sample(sample: ArchiveSample, root: Path) -> Path:
    """落库一条样本（已是校验过的模型）。返回写入路径。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sample.sample_id}.json"
    path.write_text(sample.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_sample(path: Path) -> ArchiveSample:
    """读回一条样本，读入即校验（脏 JSON 抛 ValidationError）。"""
    return ArchiveSample.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_all(root: Path) -> list[ArchiveSample]:
    """读回目录下全部样本（按文件名排序）。"""
    return [read_sample(p) for p in sorted(Path(root).glob("*.json"))]

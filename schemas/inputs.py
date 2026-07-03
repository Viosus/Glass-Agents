"""五类输入 schema —— 字段源自《钢化炉多AGENT分层AI调参小模型架构（讨论稿）.doc》§2.1–2.5。

源文档状态（2026-07-02）：正式 `docs/04` 仍缺，但架构讨论稿已给出五类输入的字段级描述，
按"真值先落源文档再实现"落地如下；正式规格出台后逐字段复核。原则不变：**未知值一律
None，不猜**；数值型字段只做结构合法性校验，工艺判定仍归 tools/constraints（规则 > AI）。

五类对照（讨论稿 §2）：
- 2.1 图像类   → 应力斑分数由外部检测软件送来（MetricRecord.fringe_score_0_100）；
                光焦度/弓波的**输出契约**见 OpticalFeatureSlots（算法与图像源未接，插槽先行）。
- 2.2 设备类   → 静态铭牌 = schemas/furnace.FurnaceConfig；动态运行 = EquipmentUsage。
- 2.3 工艺配方 → ProcessParams（schemas/process_params.py）。
- 2.4 动态调整 → 老师傅标注表管线（tools/ingest_annotations.py，Δ=最终−基准）。
- 2.5 环境辅助 → EnvironmentInput。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from schemas.process_params import ProcessParams

# 已具体定义的输入：工艺参数（讨论稿 §2.3）
ProcessParamsInput = ProcessParams


class EnvironmentInput(BaseModel):
    """环境辅助输入（讨论稿 §2.5）：车间温湿度 / 进炉玻璃温度 / 季节工况。全部可缺省。"""

    model_config = ConfigDict(extra="forbid")

    workshop_temp: float | None = None                        # 车间温度（℃，项目约定温度不带后缀）
    workshop_humidity_pct: float | None = Field(default=None, ge=0, le=100)  # 车间相对湿度 %
    glass_inlet_temp: float | None = None                     # 进炉玻璃温度（℃）
    season_note: str | None = None                            # 季节/工况备注（形态待现场，先自由文本）


class OpticalFeatureSlots(BaseModel):
    """光学类图像特征插槽（讨论稿 §2.1/§3.1.1 的输出契约）——算法与图像源未接，先定字段。

    应力斑分数不在此处（由外部检测软件评好送来）；本插槽只留 光焦度畸变 / 弓波平整度
    两类待接输出。图像源与标定到位前恒为 None（不猜、不臆造数值）。
    """

    model_config = ConfigDict(extra="forbid")

    optical_power_mdpt: float | None = None                   # 光焦度畸变（毫屈光度 mdpt）
    bow_height_mm: float | None = None                        # 弓高（mm）
    wave_amplitude_mm: float | None = None                    # 波浪幅值（mm）


class EquipmentUsage(BaseModel):
    """设备运行数据（讨论稿 §2.2 动态部分）：维保线（advisor/maintenance）的输入载体。

    数据源 = PLC/MES 台账（信息需求清单 B4）或人工填报；未知一律 None。
    """

    model_config = ConfigDict(extra="forbid")

    furnace_id: str | None = None                             # 对应 config/furnaces.yaml 登记
    run_hours: float | None = Field(default=None, ge=0)       # 累计运行小时
    changeover_count: int | None = Field(default=None, ge=0)  # 累计换产次数
    load_frac: float | None = Field(default=None, ge=0, le=1)  # 平均负荷系数（0~1）
    source_note: str | None = None                            # 数据出处备注（PLC/台账/人工）


__all__ = ["ProcessParamsInput", "EnvironmentInput", "OpticalFeatureSlots", "EquipmentUsage"]

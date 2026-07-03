"""工艺参数输入（pydantic 校验版）—— 镜像 tools.constraints.ParamSet。

写入即校验：枚举值、长度一致、正值约束。单位：温度 ℃ / 时长 s / 厚度 mm。
注意：本层只做"结构合法性"校验；硬约束/安全判定仍由 tools/constraints.validate 负责（规则 > AI）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from tools.constraints import ParamSet


class ProcessParams(BaseModel):
    """一组钢化炉工艺参数（校验版）。"""

    model_config = ConfigDict(extra="forbid")  # 脏字段拒绝

    zone_temps: list[float]                                  # ℃，按分区顺序
    zone_roles: list[str]                                    # 与 zone_temps 等长，center/edge/...
    temp_upper: float                                        # ℃
    temp_lower: float                                        # ℃
    convection_speed: float = Field(ge=0)
    convection_ratio_upper_lower: float = Field(gt=0)
    oscillation_speed: float = Field(ge=0)
    oscillation_amplitude: float = Field(ge=0)
    heating_duration_s: float = Field(gt=0)                  # s
    glass_type: Literal["ultra_clear", "clear"]
    thickness_mm: float = Field(gt=0)                        # mm
    quality_mode: Literal["high_quality", "high_efficiency"]
    # ---- 架构讨论稿 §4.2-2 补充字段（可选；约束规则待 docs/03，None 不拦）----
    convection_temp: float | None = None                     # 对流风温（℃）
    fan_startup_logic: str | None = None                     # 风机启动逻辑（形态待现场，自由文本）

    @model_validator(mode="after")
    def _check_lengths(self) -> ProcessParams:
        """结构自洽校验：zone_temps 与 zone_roles 等长且非空。"""
        if len(self.zone_temps) != len(self.zone_roles):
            raise ValueError("zone_temps 与 zone_roles 必须等长")
        if not self.zone_temps:
            raise ValueError("zone_temps 不能为空")
        return self

    def to_param_set(self) -> ParamSet:
        """转为 tools.constraints.ParamSet 以过硬约束闸门（规则 > AI）。"""
        from tools.constraints import ParamSet

        return ParamSet(**self.model_dump())

"""研判输出层的报告契约：AdvisoryReport = 架构讨论稿 §4.2 六项输出的统一装配。

设计原则（与全仓一致）：
- 每个 section 带 SectionStatus——缺真值/数据时 ok=False 并列明缺什么（对应
  《信息需求清单》编号），**绝不用占位数字装作可判定**（铁律 #5）；
- 未标定参考实现的产出一律 is_calibrated=False，不当真值下发（同 CCP 模式）；
- 数值全部来自确定性模块，LLM 只在上游产参/下游润色，不进本层。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schemas.inputs import EnvironmentInput, OpticalFeatureSlots
from schemas.process_params import ProcessParams
from tools.constraints import CheckResult


@dataclass
class SectionStatus:
    """单个输出 section 的可判定状态：ok=False 时 missing 列出缺的真值/数据。"""

    ok: bool
    missing: list[str] = field(default_factory=list)


@dataclass
class ParamsAdvice:
    """§4.2-1/2/3：全套工艺参数建议 + 安全闸门结果 + 理由（参数须已过 constraints）。"""

    status: SectionStatus
    params: ProcessParams | None = None
    check: CheckResult | None = None
    rationale: str = ""


@dataclass
class EnergyAdvice:
    """§4.2-5：能耗估计与最低能耗方案。estimate_kwh 为代理公式产出（可能未标定）。"""

    status: SectionStatus
    estimate_kwh: float | None = None
    is_calibrated: bool = False          # 系数来自 config 真值=True；注入参考=False（不当真值）
    plan_note: str = ""                  # 最低能耗方案：真值到位前说明缺口，不臆造方案


@dataclass
class ComponentWear:
    """单个部件的老化研判：wear_frac∈[0,1]，service_due=达到维保阈值。

    est_hours_to_service = §3.1.4"精准维保时间"：按"换产频次与负荷维持现状、
    老化随运行小时线性累积"外推的剩余运行小时；已达阈值=0.0；小时权重为 0
    （无法外推）= None。外推假设写死在此注释，产出随 is_calibrated 标注。
    """

    component: str
    wear_frac: float
    service_due: bool
    est_hours_to_service: float | None = None


@dataclass
class MaintenanceAdvice:
    """§4.2-6：部件老化度、维保时点与检修项目清单。"""

    status: SectionStatus
    components: list[ComponentWear] = field(default_factory=list)
    service_items: list[str] = field(default_factory=list)   # 建议检修的部件名
    is_calibrated: bool = False


@dataclass
class LoadingAdvice:
    """§4.2-4：摆炉（装载排布）建议——每床片数与排布（网格策略）。"""

    status: SectionStatus
    sheets_per_bed: int | None = None
    layout: str | None = None            # 排布描述，如 "3行×4列"
    gap_mm: float | None = None          # 采用的片间距


@dataclass
class Suspect:
    """一条归因线索：疑似原因 + 触发证据 + 疑似分区（A9 邻接未定前恒 None）。"""

    issue: str
    evidence: str
    zone: int | None = None


@dataclass
class AttributionAdvice:
    """§3.1.2：缺陷→工艺归因。violations 透传的线索独立于映射表可用性。"""

    status: SectionStatus
    suspects: list[Suspect] = field(default_factory=list)


@dataclass
class AdvisoryReport:
    """§4.2 六项输出的统一报告（§4.2-1/2/3 合并在 params 一节）。"""

    params: ParamsAdvice
    energy: EnergyAdvice
    maintenance: MaintenanceAdvice
    loading: LoadingAdvice
    attribution: AttributionAdvice
    environment: EnvironmentInput | None = None      # §2.5 环境输入随报告存档
    optical: OpticalFeatureSlots | None = None       # §3.1.1 光焦度/弓波插槽（未接恒 None）

"""研判输出层：把架构讨论稿 §4.2 六项输出装配成统一的 AdvisoryReport。

公共 API：advise() / report_to_text() / AdvisoryReport 及各 section dataclass。
原则：缺真值的节如实 cannot_determine 并列明缺什么；未标定参考值显式标注；
数值全部来自确定性模块，安全闸门在模型之外（规则 > AI）。
"""

from advisor.assemble import advise, report_to_text
from advisor.attribution import attribute
from advisor.energy import estimate_energy
from advisor.loading import suggest_loading
from advisor.maintenance import assess_maintenance
from advisor.report import (
    AdvisoryReport,
    AttributionAdvice,
    ComponentWear,
    EnergyAdvice,
    LoadingAdvice,
    MaintenanceAdvice,
    ParamsAdvice,
    SectionStatus,
    Suspect,
)

__all__ = [
    "advise",
    "report_to_text",
    "attribute",
    "estimate_energy",
    "suggest_loading",
    "assess_maintenance",
    "AdvisoryReport",
    "AttributionAdvice",
    "ComponentWear",
    "EnergyAdvice",
    "LoadingAdvice",
    "MaintenanceAdvice",
    "ParamsAdvice",
    "SectionStatus",
    "Suspect",
]

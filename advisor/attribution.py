"""归因线（讨论稿 §3.1.2）：缺陷/偏差信号 → 疑似工艺原因（规则版）。

两路线索：
① 安全闸门 violations **无条件透传**为线索（确定性输出，不依赖映射表）；
② 质量诊断量/参数残差 按 config/attribution.yaml 映射表出疑似原因——映射行
  TODO(plant) 待工艺专项（D3 诊断类目）敲定；疑似分区依赖 A9 邻接关系，未定前恒 None。

signal 字典的键约定（调用方组装）：
  质量诊断量原名直用（如 fringe_score_0_100 / centrality / fringe_area_frac）；
  参数残差加前缀 delta_（如 delta_heating_duration_s = 最终 − 基准）。
"""

from __future__ import annotations

from pathlib import Path

from advisor.report import AttributionAdvice, SectionStatus, Suspect

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "attribution.yaml"
_OPS = ("gte", "lte")


def _is_todo(v) -> bool:
    """判断配置值是否缺失（None 或 TODO 占位）→ 按无法判定处理。"""
    return v is None or (isinstance(v, str) and v.strip().upper().startswith("TODO"))


def _load_config(path: Path | None = None) -> dict:
    """读 config/attribution.yaml（每次调用按需读盘，改 yaml 即时生效）。"""
    import yaml

    p = Path(path) if path is not None else _CONFIG
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_rules(rules: list, signals: dict) -> list[Suspect]:
    """逐条映射规则对照信号值，命中即产出疑似原因（zone 恒 None，待 A9）。"""
    suspects: list[Suspect] = []
    for rule in rules:
        op = str(rule.get("op", ""))
        if op not in _OPS:
            raise ValueError(f"attribution: 规则 op 须为 {_OPS}，得到 {op!r}")
        key = str(rule.get("signal"))
        value = signals.get(key)
        if value is None:
            continue  # 该信号本次未提供 → 不判
        threshold = float(rule["threshold"])
        hit = value >= threshold if op == "gte" else value <= threshold
        if hit:
            suspects.append(
                Suspect(
                    issue=str(rule["issue"]),
                    evidence=f"{key}={value:.4g} {'≥' if op == 'gte' else '≤'} {threshold:.4g}",
                )
            )
    return suspects


def attribute(
    violations: list[str] | None = None,
    signals: dict | None = None,
    config: dict | None = None,
) -> AttributionAdvice:
    """violations + 信号字典 → 归因建议。映射表缺失 → cannot_determine（violations 线索仍给）。"""
    cfg = config if config is not None else _load_config()
    suspects = [Suspect(issue="硬约束违规（安全闸门）", evidence=v) for v in (violations or [])]

    rules = cfg.get("rules")
    if _is_todo(rules):
        return AttributionAdvice(
            status=SectionStatus(
                ok=False,
                missing=[
                    "config/attribution.yaml: rules 缺陷→原因映射表（信息需求清单 E4 / 拍板 D3）",
                    "A9 二维分区邻接关系（疑似分区定位）",
                ],
            ),
            suspects=suspects,
        )

    suspects += _apply_rules(list(rules or []), signals or {})
    return AttributionAdvice(status=SectionStatus(ok=True), suspects=suspects)

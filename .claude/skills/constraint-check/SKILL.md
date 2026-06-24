---
name: constraint-check
description: 对一组钢化炉工艺参数运行硬约束/安全校验（梯度温控、厚度-时长、对流配比、炸板风险等）。当任务涉及生成或修改工艺参数、参数落地前校验、安全底线、tools/constraints.py 或 M6 安全模块时使用。原则：规则优先级高于 AI；严格依据 docs/03。
---

# 工艺硬约束校验（安全闸门）

任何输出工艺参数的代码路径，落地前必须过本校验。

1. 读 docs/03 的 C1~C5 与安全红线。
2. 构造 ParamSet（含 prev，用于 ±3℃ 调温幅度判断），调 tools/constraints.validate(p, prev)。
3. 读 within_limits / blow_up_risk / gradient_ok / violations；任一不通过 → 拒绝该参数集，原样返回 violations，不得自动放宽。

铁律：阈值缺失（TODO(plant) 未填）→ 该项判"无法判定→不通过"，绝不用占位数字放行。

TODO(plant): docs/03-hard-constraints.md 尚未建立 —— 当前 config/thresholds.yaml 仅 C1 梯度温控有真实值（相邻 5℃ / 单步 3℃），C2 厚度-时长、C3 对流、安全红线（max_gradient / blowup_rule）均为 TODO(plant)，校验会一律判不通过直到补值。

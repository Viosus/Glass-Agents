---
name: glass-metrics
description: 计算钢化玻璃应力斑质量指标（X0.95 / IsoT / CCP）并按国标分级。当任务涉及光程差/延迟量图像、应力斑、质量判级、评估区域掩膜，或需要调用 tools/metrics.py 时使用。严格依据 docs/01，不得自行近似。
---

# 应力斑质量指标计算

1. 先读 docs/01 的分级表、CCP 公式、评估区域几何，不要凭记忆。
2. 生成 M 掩膜（扣除边缘 E、孔洞 H），仅在 M 上统计。
3. 算 X0.95 / IsoT(T=75) / CCP，分级用 config/grading.yaml 查表；多法不一致取最严，除非 grading_method 指定。
4. 实现在 tools/metrics.py，限值/参考常量在 config/，禁止硬编码；临界值必须有单测。

TODO(plant): config/ccp_reference.yaml 的 Cmax/CPmax 待标定；>15mm 限值待商定。
TODO(plant): docs/01-objective-and-metrics.md 尚未建立 —— 当前 grading.yaml 仅有 6mm X0.95 行，IsoT/CCP 分级表与其余厚度待补；补全前 grade() 对缺失项返回 None（无法判级），不得猜测。

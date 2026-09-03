---
name: glass-metrics
description: 计算钢化玻璃应力斑质量指标（X0.95 / IsoT / CCP）并按国标分级。当任务涉及光程差/延迟量图像、应力斑、质量判级、评估区域掩膜，或需要调用 tools/metrics.py 时使用。严格依据 docs/01，不得自行近似。
---

# 应力斑质量指标计算

1. 先读 docs/《钢化玻璃应力斑分级及检测方法》草案2026.5.18(1).pdf 的分级表、CCP 公式、评估区域几何，不要凭记忆。
2. 生成 M 掩膜（扣除边缘 E、孔洞 H），仅在 M 上统计。
3. 算 X0.95 / IsoT(T=75) / CCP，分级用 config/grading.yaml 查表（表1/2/3 已由草案抄入）；多法不一致取最严，除非 grading_method 指定。
4. 实现在 tools/metrics.py，限值/参考常量在 config/，禁止硬编码；临界值必须有单测。
5. **提案指标 W_w / texture_w**（位置加权组合纹理指数，tools/metrics.texture_w —— 非现行草案条文，**零标定常数**；曾名 CP_pos，2026-08-04 更名）：保留 CCP 公式形状与两个纹理分量（GLCM 对比度 Ca + 聚类突出 CPa），把参考样分母 Cmax/CPmax 换成纯数学理论上界（49 与 (2(Ng−1))⁴/12≈3201.33），位置权重 w(r)=1−r 纯几何 → 值域 [0,1)，任何实验室免标定直接可比（评估域须一致：整片矫正图）。可选物理守卫 ε（min_dynamic_range_nm，仪器精度性质）。前身 CCP_pos 工程已于 2026-07-30 整体废弃删除。

TODO(plant): config/ccp_reference.yaml 的 Cmax/CPmax（标定前 ccp() 默认路径拒绝出值；texture_w 无此依赖）与 min_dynamic_range_nm（未定→ε 条款不激活）；>15mm 限值待商定；grading.yaml 的 texture_w 分级表待标准工作组建立（建立前 grade(…,"texture_w") 返回 None，不得猜测）。

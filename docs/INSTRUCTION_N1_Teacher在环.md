# INSTRUCTION（给 Claude Code）· N1：Teacher 在环闭环

> 本文件是 **N1 阶段**的运行手册（草稿）。体例同 [INSTRUCTION_基础工程_hook与skill.md](INSTRUCTION_基础工程_hook与skill.md)。
> **现在不要动工建 N1 本体**——N1 受"开工前拍板 1–4"门槛约束（扫描仪 / Teacher LLM / 取数 / 复核形态）。
> 本草稿先把"要建什么、接什么、何时算完"写清楚，待现场拍板后照此执行。

## 目标
跑通一条完整样本闭环：
**图像 + 上下文 → 经约束校验的参数建议（附理由）→ 老师傅复核修正 → 按 schema 归档（标真值）→ 出炉 outcome 回填。**

## 总则（动手前先读）
- 先读 [CLAUDE.md](../CLAUDE.md)、[项目笔记](钢化炉Agent_项目笔记_2026-06-22.md) §2/§3、本仓库已建的 `tools/`、`schemas/`。
- **规则 > AI**：Teacher（大模型）产出的任何参数，落地前**强制再过** `tools/constraints.validate`（M6 安全闸门）；任一项不通过即拒绝，原样返回 `violations`，**不得自动放宽**。
- **缺值不放行**：阈值为 `TODO(plant)` 未填 → 判"无法判定→不通过"。N1 上线前须先补齐基础阶段登记的 `TODO(plant)`（C2/C3/安全红线等），否则 `validate` 永远拦截，闭环无法产出合规建议。
- **涉密取数走自定义代码，不用 MCP**：PLC/MES/扫描仪接口含工厂敏感信息，用项目内自定义适配器，不接外部 MCP server。
- **本地优先**：默认"本地无云"。若 Teacher 选云 API，须显式评估数据出厂合规边界（见 §5）。

## 前置门槛（必须先由现场拍板，CC 无法代办）
见文末「开工前拍板清单」。**第 1 项（扫描仪）没着落，整条飞轮无从谈起，先解决硬件。**

---

## 步骤 1 — L3 编排（Teacher 产建议 + 强制过闸门）
新建 `llm_roles/teacher_loop.py`（或 `pipeline/`），实现单片玻璃的建议生成：

1. **聚合输入**（输入契约，临时，待 docs/04 §2.1–2.5 定稿 `TODO(plant)`）：
   - 图像派生特征：应力斑指标 `X0.95 / IsoT / CCP`（调 `tools/metrics`，仅在掩膜 M 内统计）；
   - 上下文：规格（厚度/品类）、工况、质量模式（`high_quality`/`high_efficiency`）、基准配方。
2. **调 Teacher 大模型** → 产出一组工艺参数 + **理由**（理由即后期归因头的弱标签，务必留存）。
3. **强制过闸门**：把参数装成 `schemas.ProcessParams` → `.to_param_set()` → `tools.constraints.validate(p, prev)`。
   - `within_limits=False` → **不下发**，把 `violations` 连同 Teacher 理由回给老师傅（或要求 Teacher 重产）。
4. **输出建议**：`{参数, 理由, CheckResult}`。

**输出契约**：建议参数用 `schemas.ProcessParams`；归档用 `schemas.ArchiveSample`（已建）。

## 步骤 2 — L4 复核工具（老师傅修正 → 标真值 → 入库）
形态待拍板 4（CLI / 独立 Web / 嵌 HMI）。功能：
1. 展示：Teacher 建议参数 + 理由 + `violations`（若有）+ 应力斑图像与指标判级。
2. 老师傅**修正参数**；修正后：
   - 再过一次 `validate`（人工修正也不得越安全红线）；
   - 置 `is_ground_truth=true`，连同 `rationale`（专家备注）写入 `schemas.ArchiveSample`；
   - 调 `schemas.archive.write_sample` 落库到 `data/archive/`。
3. **标签一致性**（N2 关键）：记录操作员 id，便于后续监控操作员间方差、统一复核口径。

## 步骤 3 — outcome 回填
出炉后实测：
- 质量：对成品应力斑图像走 `tools/metrics` → 写回 `ArchiveSample.metrics` 与 `measured_quality_grade`；
- 能耗：`measured_energy_kwh` 写回。
- 回填只更新对应样本记录，不改其 `params`/`is_ground_truth`。

## 步骤 4 — 取数适配器（按拍板接口）
新建 `io_adapters/`（自定义代码，不用 MCP）：
- 扫描仪 → 光程差/延迟量图像（按拍板 1 的型号/接口）；图像以**路径 + sha256** 落 `schemas.ImageRef`，本体与结构化数据分离存储。
- PLC/传感器/MES → 炉子工况、配方、能耗（按拍板 3 的接口与采集频率）。

---

## §5 — Teacher LLM 选型说明（列利弊，不替现场决定）
| 方案 | 利 | 弊 |
|---|---|---|
| **云 API**（如 Claude） | 能力强、起步快、免维护 | **数据出厂**，与"本地无云"原则冲突；需评估合规与脱敏 |
| **本地大模型** | 合规（数据不出厂）、可离线 | 吃算力；需 `transformers` 推理，视情 QLoRA（见 CLAUDE.md 训练环境，12GB 可跑 7B QLoRA） |

- 若选云 Claude：模型 id / 价格 / 工具调用细节参考 `claude-api` 技能；务必先过数据脱敏与合规审批。
- 选型决定 N1 集成方式与合规边界，**由现场拍板**。

---

## DoD（供日后执行时逐条核对）
- [ ] 投一片玻璃 → 系统出**经 `validate` 校验**的参数建议（附理由）。
- [ ] 老师傅修正后入库、标 `is_ground_truth=true`、`rationale` 留存，可被 `schemas.bucketing` 统计。
- [ ] 出炉 outcome（质量/能耗）可回填到该样本。
- [ ] 跑通**一条完整样本**（图像→建议→修正→归档→回填）。
- [ ] Teacher 的任何越界参数都被 M6（`tools/constraints`）拦下（构造越界用例实测）。

---

## 开工前拍板清单（1–6，转录自项目计划，执行前对齐）
1. **应力斑图像从哪来？** ⭐最硬前置：各向异性扫描仪有无 / 型号 / 接口。没有 → 质量标签源断，飞轮转不起来。
2. **Teacher 用哪个大模型？** 云 API vs 本地（定集成与合规边界，见 §5）。
3. **怎么取炉子数据？** PLC/传感器/MES 接口与采集频率（可能需 IT/OT 配合）。
4. **老师傅在哪复核？** CLI / 独立 Web / 嵌 HMI（定 L4 形态与上手成本）。
5. **两个开放问题**：有无模块级标注（缺陷→原因、参数→最优能耗）？是否需感知小模型（光焦度畸变/弓波）？
6. **TODO(plant) 数值**：厚度-时长表、对流区间、炸板/梯度阈值、CCP 参考样品 Cmax/CPmax、权衡系数、>15mm 限值等（见基础阶段登记表）。

> 建议：1–4 至少各定一个方向后再动工 N1。准备执行时，可据本草稿细化为逐步代码任务。

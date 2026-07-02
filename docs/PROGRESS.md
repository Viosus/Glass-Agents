# 项目进度 · Glass Agents

> 更新日期：2026-07-02　|　分支：`p2-furnace-sync-dialogue`（自 foundation-and-n4-scaffold 开出）
> 🧭 **最终需求（方向锚点）**：①正向调参（模仿老师傅产参数）②**逆向诊断（最终硬目标）——从玻璃样片反推炉状态**（坏了没/参数没调好/哪个区）。逆诊断从正向模型导出，详见 `.claude/rules/项目架构说明.md` 与 §2.5。
> 终态四要求（2026-07-02 用户提出）：①Student 自我迭代+多炉云共享 ②跨炉迁移 ③对话能力 ④信息需求清单 —— 本轮全部落地地基（§2.8–2.11）。
> 一句话：**P2 三件套+炉体身份、多炉同步文件包+双重版本门、CLI 多轮对话壳、信息需求清单——全部就绪；200 份标注一到即可训参数头并进入自我迭代闭环。**

## 0. 五阶段飞轮 · 当前位置

```
[阶段0 评测靶子]✅ → [无数据期干跑]✅本次 → A 冷启动Teacher(N1) → B 飞轮攒数据(N2) ← 🟡老师傅标注已启动
                                              ⏸待拍板(LLM选型等)      → C 训练(N3/N4) ⏸数据门(200份) → D/E
```

- **已做**：不依赖现场数据的一切——护栏/指标/数据层/训练脚手架 + 本地 LLM 三角色 + 可验证训练管线 + 标注表体系 + **P2 三件套/炉体身份/多炉同步/对话壳（本轮，§2.8–2.11）**。
- **进行中**：老师傅现产现标（表已交付，导入器已就绪）；并行会话在做 `fringe_scoring/` 应力斑分布打分（未提交；本工程只承接其分数 `MetricRecord.fringe_score_0_100`，不管其内部）。
- **未做**：真训练（等 200 份）、N1 本体（待拍板）、诊断类目专项（已定为专项，见 §2.7）、云联网层（待信息+评审，文件包链路已闭环）。

## 1. 门禁状态（工作区实测，2026-07-02）

| 检查 | 结果 |
|---|---|
| `ruff check .` | ✅ All checks passed |
| `pytest tests/` | ✅ **143 passed, 1 skipped**（本轮 +67；唯一 skip = IsoT 判级表，被缺失 docs/01 阻塞。注：并行 fringe 会话施工期间其测试可能瞬态红，属对方工作区） |
| `mypy tools schemas training llm_roles sync` | ✅ 40 文件无错（本轮新增 sync 目标） |
| 提交硬闸门 | ✅ `pre_commit_gate`(Claude 侧) + `.githooks/pre-commit`(agent 无关，每 clone 启用一次) |
| 代码审查铁律#9 | ✅ 每个新建/改 .py 过 `tools/review.py`（本地 GGUF 双通道） |

## 2. ✅ 已完成项（本轮新增，2026-06-27 → 07-02）

### 2.1 本地 LLM 基建
- [x] `models/qwen2.5-3b-instruct-q4_k_m.gguf`（1.96GiB，hf-mirror 下载，SHA256 见 `VERSION.md`；gitignore 不入库）
- [x] `llama-cpp-python==0.3.30` CPU wheel（Blackwell 无匹配 CUDA wheel；装法记录在 `requirements.txt`）
- [x] `tools/review.py` **双通道代码审查器**：确定性(AST+正则，0误报，❌) + 模型补漏(Qwen，⚠️)；规则数据驱动 `config/review_rules.yaml`
- [x] CLAUDE.md **铁律#9**（py 改动必审查）；`AGENTS.md`（Codex 入口）+ git hook 硬闸门（另一会话完成）

### 2.2 llm_roles 三角色（LLM 注入式，pytest 不加载 GGUF）
- [x] `param_translator`：确定性中文骨架(数值唯一来源) + **数值守卫**(LLM 输出数字⊄骨架→回退)；越界只呈现 violations
- [x] `kb_qa`：docs→Q&A 生成(TODO(plant) 片段保守标记) + jsonl 库 + 词法检索 + **缺值/证据不足拒答**
- [x] `teacher_loop`：Teacher 产参→解析→**强制过 constraints.validate**(越界不自动修)→链翻译官；N1 骨架，3B 仅占位
- [x] 实时 Qwen 演示三链全通（守卫实测触发过一次回退——正是设计目的）

### 2.3 工作流B：多头核心可验证（真数据到来前验证架构）
- [x] `training/simulator.py` 已知 DGP(可学信号+时间漂移；**系数虚构不入 config**)
- [x] `training/features.py` ArchiveSample→特征向量契约(22维，缺值填0+presence)
- [x] `training/decode.py` Δ→ProcessParams(复用 apply_residual；安全归 constraints)
- [x] `tools/eval_gate.py` 各头指标 + 验证门(质量不回归 且 出参零违规)
- [x] 实测：400 步 val **R²(quality)=0.956 / R²(energy)=0.975**（确实能学）；默认 config 多 TODO→门如实不放行

### 2.4 老师傅标注体系（P1，已交付开工）🟡
- [x] `data/annotation/标注表模板.xlsx`(39列，枚举下拉/冻结表头/字段字典页) + `.csv`(utf-8-sig)；生成器 `tools/make_annotation_template.py`（列定义单一来源）
- [x] `docs/标注说明.md`：逐列填法 + 分层目标(**200 份深覆盖 4–6 个高产量桶**) + 一致性重复(repeat_group_id) + 红线(缺值 TODO 不猜)
- [x] `schemas/archive.py` 扩：`baseline_params`(Δ=最终−基准)、`expert_quality_grade`(主观≠实测)、`cause_tag`
- [x] `.gitignore`：**填好的标注表=工厂配方，绝不入库**（仅保留模板）

### 2.5 方向决策（已确认）
- [x] **训专用多头回归器，不微调 LLM agent**（LLM 只做语言壳：翻译/知识库/冷启动）
- [x] **感知先用工业视觉/经典 CV 顶**，不训感知模型（除非数据证明不够）
- [x] **逆诊断（样片→炉状态）为最终硬目标**：从正向模型导出——失调=推荐Δ大+归因头；坏了=质量/能耗**残差异常检测**（无需故障标签起步）；空间应力图案（fringe_scoring）是关键诊断信号
- [x] 诊断类目（异常/故障/归因+疑似分区）**走专项**，与维修+工艺现场敲定后再加标注列（2026-07-02 拍板）
- [x] **2026-07-02 拍板批次二**（详见 docs/信息需求清单.md D/C5）：Teacher=**Qwen2.5-14B-Instruct 4bit GPU**（config/llm.yaml 驱动，_llm.py transformers 后端；**权重已落盘并实测跑通**：加载≈30s/显存≈10GB，SHA256 与官方一致见 VERSION.md；transformers 锁 `<5`，5.12 加载器 Windows 崩溃记录见 以往错误.md §5）；复核=**Excel 表+CLI 辅助**；上云范围=**全量含自由文本**（自家受控云）；zone_temps 判据**预注册**（前30条≥20%动过则入Δ，config/training.yaml delta_review）；模块级标注**不加**；炉命名 **F1/F2 顺序号**；诊断类目内容仍待现场、流程已定（文本聚类→座谈→定类目）

### 2.6 文档
- [x] `docs/产品形态与安全.md`（终态形式/文件格式/使用方法/商业安全 + 空白项诚实清单）
- [x] `VERSION.md`（模型权重登记）

### 2.7 历史已完成（2026-06-24 前，详见 git log）
M0 工程地基 / M1 护栏实测 / M2 指标补全(CCP) / M3 数据底座(schemas) / N4 训练脚手架 / N1 手册草稿 / 应力斑图像 ImageRef。

### 2.8 P2 三件套 + 炉体身份（本轮，7342706c）✅
- [x] `schemas/archive.py` **v2**：+`furnace_id`/`furnace_config`（跨炉迁移地基——数据从第一天带炉子身份）+ 标注表三列（operator_id/repeat_group_id/condition_note）+ `MetricRecord.fringe_score_0_100`（承接外部 fringe 分）；全带默认值向后兼容，旧 JSON 读回零改动
- [x] `schemas/furnace.py` + `config/furnaces.yaml`：炉体配置快照（铭牌 TODO(plant) 项按缺失处理不猜）
- [x] `tools/ingest_annotations.py`：CSV→落库；表头漂移守卫(COLUMNS 单一来源)；`--furnace-id` 注入；闸门结果只记录不拒收；坏行拒绝带行号；同 id 异内容拒（--force 才覆盖）
- [x] `training/targets.py`：Δ=final−baseline 标签 + **基准侧特征化防标签泄漏** + 契约指纹
- [x] `training/train_param_head.py`：真数据只训参数头；样本 < min_train_samples 诚实退出；checkpoint 带指纹 meta
- [x] `tools/eval_gate.py::evaluate_param_gate`：逐样本各自 baseline 解码过闸门；max_param_mae=TODO(plant) 时如实降级只查违规

### 2.9 同步与自我迭代：多炉云共享地基（本轮，fe450b27）✅
- [x] 拍板：**中心汇聚·文件包先行**——数据包/模型包（zip+manifest+哈希）零联网跑通全链路；云通道留 `Transport` 接口，端点/鉴权 TODO(plant) 待评审（评审前 review 联网黑名单不动）
- [x] `schemas/datapack.py` + `sync/`：四道锁（版本/篡改/冲突/契约指纹）；`tools/sync_cli.py` 8 子命令
- [x] `tools/model_registry.py` + `MODELS.md`：**双重版本门**——中心 promote（门过+样本量+不回归）/ 炉侧 activate（本地留存验证再过门）；registry.jsonl 机器账本
- [x] 机密边界：数据包含配方 → data/outbox|inbox 进 .gitignore，只经自有通道上自家云

### 2.10 CLI 多轮对话壳（本轮）✅
- [x] `llm_roles/dialogue.py` + `run_dialogue.py`：多轮改参/查合规/看指标/问工艺；确定性关键词路由（词表 `config/dialogue_rules.yaml` 数据驱动）+ LLM 兜底单标签分类（非枚举一律 unknown）
- [x] 红线落实：**数字绝不来自 LLM**（正则抽数，歧义反问）；改参强制 validate(new, prev=旧)；越界只列 violations；诊断只报数（类目 TODO(plant) 不下结论）；`--no-llm` 全程确定性
- [x] REPL 实测通：真实 config 多 TODO → 闸门如实全拦（预期）

### 2.11 文档三件套（本轮）✅
- [x] `docs/信息需求清单.md`：**统一汇总入口**——A 安全真值 / B 炉体现场 / C 云与网络 / D 组织拍板，每项写用处/去向/阻塞
- [x] `docs/同步与自我迭代.md`（包格式/四道锁/双重门/机密边界/runbook）、`docs/对话使用说明.md`
- [x] `CONVENTIONS.md` 名册扩 v2 字段；`产品形态与安全.md` §1 加同步层/对话壳、§4.B **数据出厂边界正式更新**（只上自家受控云，绝不入 git）

## 3. ⬜ 未完成 / 阻塞项

> 🗂 **所有"还差什么信息/文件"已统一汇总到 `docs/信息需求清单.md`**（A 安全真值 / B 炉体现场 / C 云与网络 / D 组织拍板），此处只留工程项。

### 工程项
- [x] ~~中心侧 register/promote 便捷 CLI~~ → `sync_cli register-model / promote-model`（MODELS.md 自动记账；留存验证门与 activate 共用 `_holdout_gate`）
- [x] ~~对话壳接多头核心推理~~ → `model_suggest` 意图（「给个建议」）：激活模型出 Δ → 解码 → 闸门 → 呈现；**不自动应用**，逐项确认再过闸门
- [ ] 参数头**输出字段空间**：看老师傅前 ~20–30 条实际改动分布再定（当前 DELTA_FIELDS 6 标量不含 zone_temps，很可能要加；单一来源 training/decode.py，届时只改一处）⏸等数据
- [ ] `CloudTransport` 联网适配器：等清单 C1–C4 + 安全评审 → 实现 + 调整 review 联网黑名单 ⏸等信息
- [ ] 归因头 `attr_dim=4` 为占位，标签体系未定义（等诊断专项 D3）⏸等拍板

> 至此，**不依赖外部输入/数据的代码工作已全部清空**——剩余项全部在等信息（清单 A/B/C）、等数据（前 20–30 条）或等拍板（清单 D）。

### 待拍板 / 专项（详见清单 D）
- [ ] **诊断体系专项**（D3）；Teacher LLM 选型（D1）；复核工具形态（D2）；模块级标注（D5）

### 被缺失源文档阻塞（清单 A，不杜撰，留 TODO(plant)）
- [ ] **docs/01 / docs/03 / docs/04**；默认 config 多 TODO → constraints 全拦截（预期行为），拿到真值填 `config/*.yaml` 即解锁

## 4. 如何运行 / 验证

```powershell
$py = "D:\Glass Agents\.venv\Scripts\python.exe"
& $py -m pytest -q tests/                        # 143 passed, 1 skipped
& $py -m ruff check . ; & $py -m mypy tools schemas training llm_roles sync
& $py tools\run_eval.py                          # 评测靶子 8/8
& $py training\train.py --synthetic --steps 400  # 模拟器训练+验证门(R²≈0.96)
& $py tools\review.py --no-model <file.py>       # 代码审查(确定性秒级；去掉 --no-model 加模型通道)
& $py tools\make_annotation_template.py          # 重新生成标注表模板
# ---- 本轮新增 ----
& $py tools\ingest_annotations.py 填好的表.csv --furnace-id F1 [--dry-run]   # 标注表导入
& $py training\train_param_head.py --archive data\archive                    # 参数头真数据训练(不足30条诚实退出)
& $py tools\sync_cli.py status                   # 同步看板；export/import-data|model 见 docs/同步与自我迭代.md
& $py tools\sync_cli.py register-model runs\param_head ; & $py tools\sync_cli.py promote-model param_head-v0001  # 登记→第一重门
& $py llm_roles\run_dialogue.py --no-llm         # CLI 对话壳(全确定性秒开)；「给个建议」走激活模型(--model-weights 可调试)；用法见 docs/对话使用说明.md
```

## 5. 提交历史（本轮，新→旧）
```
(本次)   清空可写工程项：register/promote CLI + 对话壳 model_suggest 接模型
b3ff03b0 fringe_scoring 迭代②（并行会话）
9eb61119 P4 信息需求清单+同步/对话文档+名册与产品边界
2fa96015 P3 CLI 多轮对话壳
fe450b27 P2 同步与自我迭代——数据包/模型包+双重版本门
7342706c P1 P2三件套+炉体身份（数据从第一天带炉子身份）
40d3001b P1 标注表模板+说明+schema 扩展（老师傅开工）
523e977d AGENTS.md + git hook 硬闸门（Codex 可接手）
e744acde 工作流C Teacher 在环原型
1633e8b9 工作流A llm_roles 角色实现
fb5e876b 工作流B 多头核心可验证
a6e5f7ba 产品形态与安全 存档
304003c7 llm_roles 角色桩
6ea156a6 双通道代码审查器 review.py
84a71b97 模型大文件管理(VERSION.md)
```

## 6. 下一步
1. 老师傅前 ~20–30 条回来后：`ingest_annotations --dry-run` 试导 → 分析改动分布 → **定参数头输出字段空间**（DELTA_FIELDS 是否加 zone_temps）；异常片自由文本喂诊断专项。
2. 200 份到 → `train_param_head` 真训练 → `model_registry` 登记/晋级 → 进入自我迭代闭环。
3. 追着要 `docs/信息需求清单.md` 各板块：A（安全真值，解锁闸门）最急；B1 炉铭牌（F1 配置落 furnaces.yaml）；C（云信息，解锁联网层）。
4. 诊断体系专项（维修+工艺）→ 定类目 → 标注表加结构化诊断列 + 归因头标签空间 + 对话壳诊断升级"下结论"。
5. fringe 打分侧独立推进；其分数经 `MetricRecord.fringe_score_0_100` 随数据进入本工程（逆诊断空间信号）。
```

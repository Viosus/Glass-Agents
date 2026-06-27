# CLAUDE.md

新 session 的「开机必读」。**🔴 铁律**部分不可违反；细节按类目分流到 `.claude/rules/`，用 `@` 引入。

## 维护元规则
- 本文件 **≤ 200 行**，只留 key concept 与铁律；详细内容放 `.claude/rules/` 下分类文档，用 `@` 引用。
- 五个类目：以往错误 / 用户偏好风格 / 项目架构说明 / 安全与合规要求 / 项目专属上下文知识。
- 新增长内容先判断"属于哪个类目"，写进对应文件，**不要堆进本文件**。

## 项目一句话
Glass Agents = 对流钢化炉 · 本地多 Agent 调参系统（模仿老师傅产出工艺参数）。当前处于阶段 0「评测靶子」。架构见 `@.claude/rules/项目架构说明.md`。

## 命名与字段
标准字段名册与命名铁律见 `@CONVENTIONS.md`（单位入名、限值入 config、缺值 `TODO(plant)`）。术语中英对照见项目笔记 §4。

## 🔴 铁律（违反即出事，新 session 最易忘）
1. **Python 解释器**：只用 venv `D:\Glass Agents\.venv\Scripts\python.exe`（3.12.10）。系统 `python`=3.14（无 ML wheel）、`python3`=Microsoft Store 假桩，**都不能用**。
2. **PyTorch**：RTX 5070 是 Blackwell(sm_120)，**必须 cu128**（当前 torch 2.11.0+cu128）。
3. **规则 > AI**：任何产出工艺参数的代码路径，落地前必须过 `tools/constraints.py`；`TODO(plant)` 缺值项一律判「无法判定→不通过」，**绝不用占位数字放行**。
4. **改值只动 config**：限值 / 分级写进 `config/*.yaml`，**禁止硬编码进 `tools/`**；config 在每次调用时按需读取。
5. **缺值标 TODO(plant)**：安全 / 国标常量未知一律标记，**绝不猜测**。单位固定 `℃ / s / mm / nm`。
6. **提交**：非经明确要求不 commit / push；当前在 `master`，要提交先开分支。commit 会先过硬闸门（pytest 不过则拦）。
7. **测 commit 闸门**：`pre_commit_gate` 对**任何含 "git commit" 子串**的命令触发 —— 离线测试用文件喂 payload，别在命令行直接写该字串。
8. **没有的别装有**：源文档 `docs/01`、`docs/03`、`PLAN.md` 不存在，C4/C5 代号无定义 —— 不臆造数值或含义。
9. **py 改动必审查**：每新建/修改一个 `.py` 后，必须跑 `tools/review.py <该文件>`（本地 GGUF，四项机械规则）并把结果展示给用户；**审查通过（✅）才算该任务完成**。

## 分类文档（@ 引用，按需展开）
- `@.claude/rules/以往错误.md` — 踩过的坑：现象 + 根因 + 正确做法。
- `@.claude/rules/用户偏好风格.md` — 中文、简洁、增量、统计背景、遇阻塞先摆不瞎猜。
- `@.claude/rules/项目架构说明.md` — 多头核心模型、五阶段飞轮、仓库结构。
- `@.claude/rules/安全与合规要求.md` — 规则>AI、缺值不放行、安全红线、约束代号。
- `@.claude/rules/项目专属上下文知识.md` — 训练环境、Windows 限制、重装命令、Claude Code 工程约定。

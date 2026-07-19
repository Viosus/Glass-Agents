# AGENTS.md

Codex（及任何非 Claude Code 的 agent）的开机必读。**Claude Code 用户请以 `CLAUDE.md` 为入口**——本文件是给 Codex 的等价入口，两者规则同源。

## 第一件事：读全套规则
本项目的完整、权威规则在下列文件，**开工前必须通读，并当作对你同样强制生效的约束**：
- `CLAUDE.md` —— 铁律与 key concept（下方已镜像铁律，但以 CLAUDE.md 为准）
- `.claude/rules/*.md` —— 五类细则：以往错误 / 用户偏好风格 / 项目架构说明 / 安全与合规要求 / 项目专属上下文知识
- `CONVENTIONS.md` —— 字段名册与命名铁律（单位入名、限值入 config、缺值 `TODO(plant)`）

## 项目一句话
Glass Agents = 对流钢化炉 · 本地多 Agent 调参系统。最终需求**双向**：①正向调参（模仿老师傅产出工艺参数）②**逆向诊断（最终硬目标）**：从玻璃样片反推炉状态（坏了没/参数没调好/哪个区）。方向锚点见 `.claude/rules/项目架构说明.md`。

## 🔴 铁律（与 CLAUDE.md 同源镜像；改动须两处同步）
1. **Python 解释器**：只用 venv `D:\Glass Agents\.venv\Scripts\python.exe`（3.12.10）。系统 `python`=3.14、`python3`=Store 假桩，**都不能用**。
2. **PyTorch**：RTX 5070 是 Blackwell(sm_120)，**必须 cu128**（torch 2.11.0+cu128）。
3. **规则 > AI**：任何产出工艺参数的代码路径，落地前必须过 `tools/constraints.py`；`TODO(plant)` 缺值项一律判「无法判定→不通过」，**绝不用占位数字放行**。
4. **改值只动 config**：限值/分级写进 `config/*.yaml`，**禁止硬编码进 `tools/`**；config 运行时按需读取。
5. **缺值标 TODO(plant)**：安全/国标常量未知一律标记，**绝不猜测**。单位固定 `℃ / s / mm / nm`。
6. **提交**：非经明确要求不 commit / push；要提交先开分支。提交会过硬闸门（见下）。
7. **没有的别装有**：`docs/01`、`docs/03`、`PLAN.md` 不存在，C4/C5 无定义 —— 不臆造。
8. **py 改动必审查**：每新建/修改一个 `.py` 后，跑 `tools/review.py <该文件>`（本地 GGUF，四项机械规则）并把结果展示给用户；审查通过（✅）才算完成。

## 提交硬闸门（agent 无关，替代 Claude Code 的 hook）
Claude Code 靠 `.claude/hooks/pre_commit_gate.py`（PreToolUse）在提交前跑 pytest 拦截。为让 **Codex / 人手动 / CI 提交同样受拦**，本仓库把同一闸门下沉成真正的 git hook：`.githooks/pre-commit`。

**每个 clone 启用一次**（Claude Code 侧无需此步，其自身 hook 仍照常工作）：
```
git config core.hooksPath .githooks
```
启用后 `git commit` 会先跑 `pytest -q tests/`，不过则退出码非 0 拦下提交。venv 解释器缺失时**拒绝提交而非静默放行**。

## 技能调度（命中触发条件时，先读对应文件再动手）
Codex 无 Claude 的 skill 自动发现机制，改为手动调度：
- 涉及**应力斑质量指标 / X0.95·IsoT·CCP / 质量判级 / 评估区域掩膜 / `tools/metrics.py`**
  → 读 `.claude/skills/glass-metrics/SKILL.md`，严格依据 docs/01，不得自行近似。
- 涉及**生成或修改工艺参数 / 参数落地前校验 / 安全底线 / `tools/constraints.py`**
  → 读 `.claude/skills/constraint-check/SKILL.md`，规则优先级高于 AI，严格依据 docs/03。

## 环境速记
- 激活 venv：`.venv\Scripts\Activate.ps1`；跑测试：`.venv\Scripts\python.exe -m pytest -q tests/`。
- Windows 未装：vLLM / flash-attn / DeepSpeed / triton / xformers（替代方案见 `.claude/rules/项目专属上下文知识.md`）。
- 缺失源文档：`docs/01` `docs/03` `PLAN.md` —— 相关 config 项保持 `TODO(plant)`。

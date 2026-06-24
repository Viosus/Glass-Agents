# INSTRUCTION（给 Claude Code）· 基础工程：配置 Hook 与 Skill

> 这份文件是给你（Claude Code）执行的运行手册。目标是把项目的**护栏层**（hooks + skills）搭起来，并写**刚好够让护栏可验证**的最小代码。**不要**在本次构建六个领域模块、大模型在环、人工复核闭环或任何训练脚手架——那些等护栏稳了再按 `PLAN.md` 往后走。

## 总则（动手前先读）
- 先读 `CLAUDE.md`、`docs/01-objective-and-metrics.md`、`docs/03-hard-constraints.md`。
- **规则 > AI**：任何产出工艺参数的代码路径，落地前必须过 `tools/constraints.py`。
- 单位固定：`nm / ℃ / mm / s`，字段名带单位语义，禁止裸数字。
- **安全相关常量缺失一律标 `TODO(plant)`，绝不猜测填占位数字当真值。**
- 先写**纯函数 + 单元测试**；每步做完跑 `pytest`，对照"验收"勾选后再 commit。
- 提交信息前缀：`feat: / fix: / test: / docs: / chore:`。

## 本次产出
1. 最小可测代码：`tools/constraints.py` + `tools/metrics.py` + `config/` + `tests/`。
2. 两个 Skill：`glass-metrics`、`constraint-check`。
3. 两个 Hook：提交前**硬闸门**（测试不过则拦截 commit）+ 编辑后**快速反馈**。
4. 验证：故意制造失败 → 提交被拦；skill 可发现；测试全绿。

---

## 步骤 1 — 仓库骨架
创建目录与 Python 工程：
```
tools/  tests/  config/  .claude/skills/  .claude/hooks/
```
- `requirements.txt`：`pytest`、`numpy`。
- Hook 脚本用 **Python 写**（避免额外依赖 jq）。确认 `python3` 在 PATH。

## 步骤 2 — config（数据驱动，禁止把限值硬编码进逻辑）
创建三个文件：

**`config/grading.yaml`** —— 把 `docs/01` 的**三张分级表**完整抄成配置。结构按下例（这是 X0.95 的一行示范，其余厚度与 IsoT、CCP 两表照此补全，数值以 docs/01 为准）：
```yaml
x0_95_nm:        # 方法 4.2
  - thickness_mm: 6   # 适用 ≤6
    A_max: 70
    B_max: 95         # A_max < v ≤ B_max 为 B；> B_max 为 C
  # 8 / 10 / 12 / 15 ... 按 docs/01 补全；>15 标 TODO(plant)
iso_t_pct: { ... }   # 方法 4.3，阈值 T=75nm，注意是"≥"方向
ccp: { ... }         # 方法 4.4
```

**`config/thresholds.yaml`** —— 硬约束阈值；C1 有明确值，其余占位：
```yaml
gradient:
  adjacent_zone_max_delta_c: 5     # 相邻分区温差 ≤5℃
  single_step_max_delta_c: 3       # 单次调温 ≤±3℃
thickness_duration: TODO(plant)    # 厚度→[时长下限,上限]
convection: TODO(plant)            # 品类→风速区间；上下配比范围
safety:
  blowup_rule: TODO(plant)         # 炸板风险判定条件
  max_gradient: TODO(plant)        # 温度梯度上限
```

**`config/ccp_reference.yaml`**：
```yaml
c_max: TODO(plant)     # 参考最差样品对比度
cp_max: TODO(plant)    # 参考最差样品聚类突出
```

## 步骤 3 — tools/constraints.py（Hook 守的对象）
实现以下接口；**每条规则一个纯函数**，阈值从 `config/thresholds.yaml` 读：
```python
@dataclass
class ParamSet:
    zone_temps: list[float]          # ℃，按分区顺序
    zone_roles: list[str]            # 与 zone_temps 等长，"center"/"edge"/...
    temp_upper: float; temp_lower: float
    convection_speed: float
    convection_ratio_upper_lower: float
    oscillation_speed: float; oscillation_amplitude: float
    heating_duration_s: float
    glass_type: str                  # ultra_clear | clear
    thickness_mm: float
    quality_mode: str                # high_quality | high_efficiency

@dataclass
class CheckResult:
    within_limits: bool
    blow_up_risk: bool
    gradient_ok: bool
    violations: list[str]            # 人类可读，含触发的规则编号

def validate(p: ParamSet, prev: ParamSet | None) -> CheckResult: ...
```
实现要点：
- **C1 梯度温控**（全部用 config 的值）：`min(center 温度) > max(edge 温度)`；`temp_upper > temp_lower`；相邻分区温差 `≤ adjacent_zone_max_delta_c`（相邻=`zone_temps` 中相邻索引，**简化处理**，真实二维邻接关系标 `TODO(plant)`）；单次调温 `≤ single_step_max_delta_c`（需 `prev`，对位比较 `zone_temps`）。
- **C2/C3/安全红线**：相应阈值为 `TODO(plant)` 未填时，该项判定返回"无法判定 → 视为不通过"，并在 `violations` 写明"需补 TODO(plant)"。**绝不放行。**
- `prev=None` 时跳过"单次调温"规则。

**验收（写进 `tests/test_constraints.py`）：**
- 相邻分区差 **5.0℃** → 该规则通过；**5.01℃** → `within_limits=False`，`violations` 含 C1。
- 单次调温 **+3.0℃** → 通过；**+3.01℃** → 不通过（构造带 `prev` 的用例）。
- 某 center 温度 ≤ 某 edge 温度 → `within_limits=False`，违规指向 C1。
- 把 `safety.max_gradient` 置 `TODO(plant)` → 安全项判"无法判定→不通过"。

## 步骤 4 — tools/metrics.py
实现评估区域掩膜 + 三指标（CCP 先脚手架）：
- **M 掩膜**（`docs/01 §2`）：边缘 E：`La=10%L, Wa=10%W`，下限 50mm；厚度≤8mm 上限 200mm，≥10mm 上限 350mm。孔洞 H：每孔半径 `6×厚度 + 孔半径`。
- `x0_95(retardation_nm_masked) -> float`：M 内升序的 95% 分位。
- `iso_t(retardation_nm_masked, T=75) -> float`：M 内 `<T` 占比（%）。
- `grade(value, thickness_mm, method) -> "A"|"B"|"C"`：查 `config/grading.yaml`。
- `ccp(...)`：**先留脚手架**（签名 + docstring 写明 GLCM Ng=8、归一化 10000px²/100mm、四向、公式 `0.5*(sqrt(Ca/Cmax)+(CPa/CPmax)**0.25)`，`Cmax/CPmax` 读 `config/ccp_reference.yaml`），实现体先 `raise NotImplementedError`，对应测试 `@pytest.mark.skip(reason="CCP 待实现")`。

**验收（`tests/test_metrics.py`）：**
- 6mm：构造光程差数组使 X0.95 落在 **70 与 95 两侧** → `grade` 分别为 A/B/C。
- IsoT：构造已知 `<75nm` 占比的数组 → 数值与判级正确（注意"≥"方向）。
- 掩膜：厚度 **6/8/10/15** 的 E 上下限正确（200 vs 350 分界）；孔洞半径 `6t+r` 正确。

## 步骤 5 — 两个 Skill
**`.claude/skills/glass-metrics/SKILL.md`**：
```markdown
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
```

**`.claude/skills/constraint-check/SKILL.md`**：
```markdown
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
```

## 步骤 6 — 两个 Hook
**`.claude/hooks/pre_commit_gate.py`**（提交前硬闸门，退出码 2 = 拦截）：
```python
#!/usr/bin/env python3
import json, sys, subprocess, os
data = json.load(sys.stdin)
cmd = (data.get("tool_input") or {}).get("command", "")
if "git commit" not in cmd:
    sys.exit(0)
proj = os.environ.get("CLAUDE_PROJECT_DIR", ".")
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"],
                   cwd=proj, capture_output=True, text=True)
if r.returncode != 0:
    sys.stderr.write("提交被拦截：安全/指标测试未通过（规则 > AI）。请先修复再提交。\n")
    sys.stderr.write((r.stdout + r.stderr)[-1500:])
    sys.exit(2)
sys.exit(0)
```

**`.claude/hooks/posttool_tests.py`**（编辑核心代码后快速反馈，非阻塞）：
```python
#!/usr/bin/env python3
import json, sys, subprocess, os
data = json.load(sys.stdin)
path = (data.get("tool_input") or {}).get("file_path", "")
if not any(seg in path for seg in ("/tools/", "/tests/", "/config/")):
    sys.exit(0)
proj = os.environ.get("CLAUDE_PROJECT_DIR", ".")
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"],
                   cwd=proj, capture_output=True, text=True)
if r.returncode != 0:
    ctx = "编辑 %s 后核心测试未通过，请先修复：\n%s" % (path, (r.stdout + r.stderr)[-1200:])
    print(json.dumps({"hookSpecificOutput":
        {"hookEventName": "PostToolUse", "additionalContext": ctx}}))
sys.exit(0)
```

**`.claude/settings.json`**（项目级，提交进仓库全队共享）：
```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash",
        "hooks": [ { "type": "command",
          "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/pre_commit_gate.py\"",
          "timeout": 120 } ] }
    ],
    "PostToolUse": [
      { "matcher": "Write|Edit|MultiEdit",
        "hooks": [ { "type": "command",
          "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/posttool_tests.py\"",
          "timeout": 120 } ] }
    ]
  }
}
```
给两个脚本加可执行权限：`chmod +x .claude/hooks/*.py`。

## 步骤 7 — 验证（必须全部做到）
1. `python -m pytest -q tests/` —— 全绿（CCP 用例为 skip）。
2. 手动测硬闸门（测试绿时应退出 0）：
   ```bash
   echo '{"tool_input":{"command":"git commit -m x"}}' | CLAUDE_PROJECT_DIR=$(pwd) python3 .claude/hooks/pre_commit_gate.py; echo $?
   ```
3. **故意改坏** `tools/constraints.py` 里一条规则 → 再跑步骤 2 的命令 → 应退出 **2** 并打印失败原因（说明真实 commit 会被拦）→ 改回，确认恢复退出 0。
4. 在 Claude Code 里运行 `/hooks`，确认两个 hook 已注册。
5. 触发 skill：编辑 `tools/metrics.py` 时，确认 `glass-metrics` 技能被自动参考。
6. 全部通过后再 `git commit`（此时硬闸门会先跑测试放行）。

## VS Code 提示
- `.claude/` 下的 hooks 与 skills 提交进仓库即全队共享。
- 用 VS Code 集成终端跑 `claude` 最稳；`/hooks` 查配置；改完用步骤 2/3 的管道命令**离线**测 hook，不必每次靠真实提交触发。
- 注：自定义 **subagent** 在扩展面板曾有不被识别的问题，但 **hook / skill 不受此影响**；仍建议用 `/hooks` 确认注册。

## 暂不做（避免越界）
六个领域模块、大模型在环、人工复核闭环、训练脚手架。护栏稳了之后按 `PLAN.md` 阶段 2 起继续。
```

完成后请输出：已建文件清单、`pytest` 结果、步骤 7 第 3 条"改坏→被拦→改回"的实测退出码，以及仍未解决的 `TODO(plant)` 待工艺确认项汇总。

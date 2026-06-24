# 项目进度 · Glass Agents

> 更新日期：2026-06-24　|　分支：`foundation-and-n4-scaffold`
> 一句话：**基础阶段（评测靶子 + 护栏 + 数据层）+ N4 训练脚手架 已完成并实测全绿；真训练与 N1 本体受"数据门/决策门"约束，未动。**

## 0. 五阶段飞轮 · 当前位置

```
[阶段0 评测靶子] ✅本次完成  →  A 冷启动Teacher(N1)  →  B 飞轮攒数据(N2)  →  C 训练Student(N3/N4)  →  D 验证(N5)  →  E 部署(N6)
     M0–M3 + N4脚手架 + N1草稿            ⏸待拍板1–4         ⏸时间门            ⏸数据门/决策门         ⏸质量门        ⏸
```

- **现在能做且已做**：纯软件、不依赖现场数据的一切（地基/护栏/指标/数据层 + 训练管线脚手架）。
- **现在不能做**：真训练（无数据）、N1 本体（待现场拍板扫描仪/LLM/取数/复核形态）。

## 1. 门禁状态（committed 态）

| 检查 | 结果 |
|---|---|
| `ruff check .` | ✅ All checks passed |
| `pytest tests/` | ✅ **42 passed, 1 skipped**（唯一 skip = IsoT 判级表，被缺失的 docs/01 阻塞） |
| `mypy tools schemas training` | ✅ Success, no issues |
| 提交硬闸门 `pre_commit_gate` | ✅ 改坏→退2→改回→退0 实测；5 个 commit 均过闸门 |

## 2. ✅ 已完成项

### M0 · 工程地基与可复现性
- [x] `pyproject.toml`：ruff(E/F/I/**N** 命名) + pytest + mypy(宽松)；pep8-naming 放行科学记号例外
- [x] `CONVENTIONS.md`：唯一字段名册 + 命名铁律（单位入名/限值入 config/缺值 TODO(plant)）
- [x] 版本锁 `requirements.lock.txt`（pip freeze 快照）+ `.python-version`(3.12.10)
- [x] `.gitignore` / `.editorconfig`
- [x] `tests/test_conventions.py`：单位后缀语义自检（linter 兜底）
- [x] 迁移 `pytest.ini` → `pyproject[tool.pytest.ini_options]`，删 `pytest.ini`
- [x] CLAUDE.md 指向 CONVENTIONS.md

### M1 · 护栏验证（已建，仅实测）
- [x] `pytest` 全绿
- [x] 硬闸门"改坏一条 C1 → 退出码 2 拦截 → 改回 → 退出码 0"实测通过
- [x] 两 hook 注册确认（PreToolUse=Bash|PowerShell、PostToolUse=Write|Edit|MultiEdit）
- [x] 记录 Windows stdin 离线测坑（`.claude/rules/以往错误.md`）

### M2 · 指标补全（仅未被阻塞部分）
- [x] **CCP 计算实现**：GLCM Ng=8、四方向(0/45/90/135°)平均、公式 `0.5*(sqrt(Ca/Cmax)+(CPa/CPmax)**0.25)`
- [x] CCP 可注入合成参考（`is_calibrated=False`，显式标注未标定）；默认 config 的 Cmax/CPmax 为 TODO(plant) → 仍抛 `NotImplementedError`（不当真值下发）
- [x] 去 CCP 测试 skip；增合成参考/安全/边界用例
- [x] 约束边界用例补全：center≈edge 边界、C2/C3 缺值拦截、全阈值齐备放行
- [x] `tools/run_eval.py`：评测靶子一键运行（8/8 = 100%）

### M3 · 数据底座
- [x] `schemas/process_params.py`：`ProcessParams`（镜像 ParamSet，枚举/长度/正值校验，`extra=forbid`）
- [x] `schemas/archive.py`：`ArchiveSample`(L5)，图像存 路径+sha256 与结构化数据分离；JSON 写读即校验
- [x] `schemas/bucketing.py`：按 (厚度/品类/质量模式) 统计 `is_ground_truth` 样本量
- [x] `tests/test_schemas.py`：脏数据被拒、写读 roundtrip、分桶计数（5 条假样本端到端）

### N4 · 训练脚手架（合成数据，不跑真训练）
- [x] `training/model.py`：多头核心（共享编码器 + 参数(残差Δ)/质量/能耗/归因头）
- [x] `training/losses.py`：多头损失 + 不确定性加权（副头较小权重，防负迁移）
- [x] `training/dataset.py`：按时间/批次顺序切分，**禁止随机打乱**（防泄漏）
- [x] `training/{synthetic,train}.py`：合成数据 + 冒烟训练（cuda 跑通）；**规则永不进权重**（出参过 `tools.constraints.validate` 闸门）
- [x] `tests/test_training.py`：forward 形状/损失标量/一步训练/切分无泄漏/出参过闸门

### N1 · Teacher 在环（仅起草手册，不建本体）
- [x] `docs/INSTRUCTION_N1_Teacher在环.md`：L3 编排/L4 复核/outcome 回填/取数适配器/LLM 选型/DoD/开工前拍板清单

## 3. ⬜ 未完成 / 阻塞项

### A. 被"缺失源文档"阻塞（按铁律不杜撰，留 TODO(plant)）
- [ ] **docs/01**：X0.95 的 8/10/12/15mm 及 >15mm 限值；IsoT、CCP 两张分级表；CCP 的 `Cmax/CPmax` 标定值 + GLCM 精确口径；掩膜边带 8<t<10mm 分界
- [ ] **docs/03**：C2 厚度→加热时长；C3 风速区间/上下配比；安全 `max_gradient`/`blowup_rule`；C4/C5 定义；二维分区邻接关系（C1 现为一维简化）
- [ ] **docs/04**：五类输入 2.1–2.5 精确字段（当前仅"工艺参数输入"已实现）

### B. 需现场/工艺拍板（CC 无法代办，N1 前置）
- [ ] **应力斑图像来源**（各向异性扫描仪）⭐最硬前置——没有则质量标签源断，飞轮转不起来
- [ ] **Teacher 用哪个大模型**（云 API vs 本地，定合规边界）
- [ ] **取炉子数据接口**（PLC/传感器/MES + 采集频率）
- [ ] **老师傅复核工具形态**（CLI / Web / HMI）
- [ ] 两个开放问题：有无模块级标注（缺陷→原因、参数→最优能耗）？是否需感知小模型（光焦度畸变/弓波）？
- [ ] 目标等级（默认 A）与「质量/能耗/损耗」权衡系数

### C. 后续阶段（受门槛约束，本次不做）
- [ ] **N1 本体**：搭 Teacher 在环闭环（待 B 组拍板 1–4）
- [ ] **N2**：飞轮运行，按桶攒 `is_ground_truth` 样本 + 标签一致性监控（时间门）
- [ ] **N3**：用数据回答两个开放问题；定多头模型 I/O；按时间/桶划分训练集（数据门/决策门）
- [ ] **N4 真训练**：Student 蒸馏（脚手架已就绪，待数据门）
- [ ] **N5**：影子运行验证门（质量不回归 + 约束零违规）
- [ ] **N6**：本地离线部署 + 概念漂移监控 + 周期重训

## 4. 如何运行 / 验证

```powershell
$py = "D:\Glass Agents\.venv\Scripts\python.exe"
& $py -m pytest -q tests/          # 42 passed, 1 skipped
& $py -m ruff check .              # All checks passed
& $py -m mypy tools schemas training
& $py tools\run_eval.py            # 评测靶子 8/8
& $py training\train.py --synthetic --steps 5   # 训练管线冒烟（合成数据）
```

## 5. 提交历史（分支 foundation-and-n4-scaffold）
```
docs: N1 Teacher 在环 INSTRUCTION 草稿
feat: N4 训练脚手架（多头模型 + 合成数据冒烟）
feat: M3 数据底座（pydantic schemas + 分桶 + 写读校验）
feat: M2 指标补全（CCP 计算 + 评测/约束边界用例）
chore: 基线 + M0 工程地基与可复现性
```

## 6. 下一步（要往前走的前提）
1. 先解决 **B 组现场拍板**（尤其扫描仪硬件）——否则飞轮无法启动。
2. 拿到 **A 组 docs/01·03·04 真值** → 填 `config/*.yaml`（改 yaml 即生效，无需改代码）→ 解锁分级表/约束/五类输入。
3. 据 N1 草稿搭 Teacher 在环 → 攒数据 → 才进真训练。

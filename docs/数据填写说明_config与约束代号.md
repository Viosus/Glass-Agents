# 数据填写说明 · config 字段与约束代号

> 用途：拿到真实数据 / 国标值 / 现场标定后，**该填哪个文件、哪个字段**，以及代码里 **C1、C2…** 等代号到底指什么。
> 配套护栏代码见 `tools/constraints.py`、`tools/metrics.py`，权威规格见 [INSTRUCTION_基础工程_hook与skill.md](INSTRUCTION_基础工程_hook与skill.md)。

## 0. 铁律（先读）
- **只改 `config/` 里的 yaml，不要改 `tools/` 代码**。限值是数据驱动的，改 config 即自动生效。
- 缺值占位一律写 `TODO(plant)`。代码遇到 `TODO(plant)` 的项会判**"无法判定 → 不通过"**，绝不放行 —— 这是"规则 > AI"的安全保证。
- 单位固定：温度 `℃`、时长 `s`、长度/厚度 `mm`、光程差/延迟量 `nm`。禁止裸数字，含义跟着字段走。
- 填完务必跑 `pytest`（提交时硬闸门也会强制跑），确保没填崩。

---

## 1. 数据填写速查表

| 数据类型 | 文件 | 关键字段 | 当前状态 |
|---|---|---|---|
| 国标分级表 · X0.95（各厚度 A/B/C 限值） | [config/grading.yaml](../config/grading.yaml) | `x0_95_nm[].A_max / B_max` | 仅 6mm 有值，其余 **TODO** |
| 国标分级表 · IsoT | [config/grading.yaml](../config/grading.yaml) | `iso_t_pct` | **TODO** |
| 国标分级表 · CCP | [config/grading.yaml](../config/grading.yaml) | `ccp` | **TODO** |
| 厚度→加热时长 | [config/thresholds.yaml](../config/thresholds.yaml) | `thickness_duration` | **TODO** |
| 对流风速 / 上下配比 | [config/thresholds.yaml](../config/thresholds.yaml) | `convection` | **TODO** |
| 温度梯度上限（安全红线） | [config/thresholds.yaml](../config/thresholds.yaml) | `safety.max_gradient` | **TODO** |
| 炸板风险判定规则（安全红线） | [config/thresholds.yaml](../config/thresholds.yaml) | `safety.blowup_rule` | **TODO** |
| 相邻分区温差上限 | [config/thresholds.yaml](../config/thresholds.yaml) | `gradient.adjacent_zone_max_delta_c` | 已填 **5℃** |
| 单次调温幅度上限 | [config/thresholds.yaml](../config/thresholds.yaml) | `gradient.single_step_max_delta_c` | 已填 **3℃** |
| CCP 参考最差样品 Cmax / CPmax | [config/ccp_reference.yaml](../config/ccp_reference.yaml) | `c_max / cp_max` | **TODO** |

---

## 2. 约束代号含义（C1 / C2 / C3 / 安全红线）

> 这些代号会出现在校验失败信息 `violations` 里，便于定位是哪条规则拦的。
> 实现见 `tools/constraints.py` 的 `validate()`。

### C1 · 梯度温控（已有真实值，可用）
保证炉内温度场合理、避免热应力失控。含 4 条子规则：
1. **中心 > 边缘**：`min(center 区温度) > max(edge 区温度)`（结构规则，无需阈值）。
2. **上炉温 > 下炉温**：`temp_upper > temp_lower`（结构规则）。
3. **相邻分区温差 ≤ 上限**：默认 `5℃`，读 `gradient.adjacent_zone_max_delta_c`。
   - ⚠️ 当前"相邻"= 分区数组中索引相邻的**一维简化**；真实二维分区邻接关系待补（`TODO(plant)`）。
4. **单次调温幅度 ≤ 上限**：默认 `±3℃`，读 `gradient.single_step_max_delta_c`；需传入上一组参数 `prev`，`prev=None` 时跳过此条。

### C2 · 厚度→加热时长（待填）
不同玻璃厚度对应的加热时长合理区间 `[下限, 上限]`（s）。
字段 `thickness_duration`，当前 `TODO(plant)` → 校验一律不通过，直到填入映射表。

### C3 · 对流（待填）
按品类（超白 / 普白）的风速区间，以及上下对流配比范围。
字段 `convection`，当前 `TODO(plant)` → 不通过，直到填值。

### 安全红线（待填，优先级最高）
独立于上面三类、永远在模型之外强制执行：
- **温度梯度上限** `safety.max_gradient` —— 安全红线，缺值即判不通过。
- **炸板风险判定规则** `safety.blowup_rule` —— 缺规则时保守置 `blow_up_risk=True`（无法排除即视为有风险）。

### C4 / C5 —— 尚未定义 ⚠️
INSTRUCTION 提到"docs/03 的 **C1~C5**"，但 `docs/03-hard-constraints.md` 目前**不存在**，C4、C5 的含义**无任何来源、未实现**。
待你补 `docs/03` 后再定义并落地，**不要**凭空猜测填充。（安全红线两项将来是否归编为 C4/C5，也以 docs/03 为准。）

---

## 3. 质量指标说明（对应 grading.yaml）

实现见 `tools/metrics.py`，仅在**评估区域掩膜 M**（扣除边缘带 E、孔洞 H）内统计。

| 指标 | 含义 | 方向 | 判级查表字段 |
|---|---|---|---|
| **X0.95** | M 内光程差的 95% 分位（nm） | 越**小**越好：`v≤A_max→A`；`A_max<v≤B_max→B`；`v>B_max→C` | `x0_95_nm` |
| **IsoT** | M 内光程差 `< T(=75nm)` 的面积占比（%） | 越**大**越好（"≥"方向） | `iso_t_pct` |
| **CCP** | 聚类对比度参数（GLCM，方法 4.4） | 待定 | `ccp` + `ccp_reference.yaml` |

- X0.95 分级表的厚度匹配：每行 `thickness_mm` 表示"适用 ≤该值"，查表取 `thickness_mm ≥ 查询厚度` 的最小一行。
- CCP 目前是**脚手架**：`ccp()` 在 `Cmax/CPmax` 未标定时抛 `NotImplementedError`，相关测试 `skip`。
- 任何缺失项 `grade()` 返回 `None`（无法判级），不猜测。

---

## 4. 填完怎么验证
```powershell
& "D:\Glass Agents\.venv\Scripts\python.exe" -m pytest -q tests/
```
- 全绿即填写未破坏现有规则。
- 新填的限值建议**补对应临界值单测**（参考 `tests/test_constraints.py`、`tests/test_metrics.py` 的写法）。
- 提交时 `pre_commit_gate` 会自动跑测试，不过则拦截 commit。

---

## 5. TODO(plant) 总清单（待工艺/文档确认）
- [ ] `docs/01-objective-and-metrics.md`（国标分级表、CCP 公式、评估区域几何）—— 源文档待建
- [ ] `docs/03-hard-constraints.md`（C1~C5 完整定义，含 C4/C5）—— 源文档待建
- [ ] X0.95：8 / 10 / 12 / 15mm 各行；>15mm 限值
- [ ] IsoT、CCP 两张分级表
- [ ] 厚度→时长映射（C2）
- [ ] 对流风速区间 / 上下配比（C3）
- [ ] 温度梯度上限、炸板判定规则（安全红线）
- [ ] CCP 的 Cmax / CPmax 标定值
- [ ] 二维分区邻接关系（C1 相邻规则当前为一维简化）
- [ ] 掩膜边带上限在 8<t<10mm（如 9mm）的分界

---

## 6. 与源文档的关系
本文件是**填写指南 + 代号词典**，不是数值真值的出处。正式流程：真值先落进 `docs/01` / `docs/03`（国标 / 安全规程的权威记录），再抄进 `config/` 的 yaml。两者一旦都到位，本说明的"当前状态"列即可全部转绿。

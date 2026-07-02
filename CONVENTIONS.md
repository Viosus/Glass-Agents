# CONVENTIONS · 标准字段名册与命名铁律

> 本文件是本工程的**唯一字段名册**（`docs/00-glossary.md` 不存在，故由此处统一）。
> 术语中英对照（监督学习 / 蒸馏 / 编码器 / 多头…）见 [项目笔记 §4](docs/钢化炉Agent_项目笔记_2026-06-22.md)。
> 命名规则由 `ruff`（pep8-naming N）自动把关；linter 管不了的项目特例由 [tests/test_conventions.py](tests/test_conventions.py) 兜底。

## 1. 命名铁律
- **大小写**：变量/函数 `snake_case`；类 `PascalCase`；模块级常量 `UPPER_SNAKE`。
- **单位入名（禁裸数字）**：携带物理量的字段名**必须带单位后缀**，含义跟着字段走：
  | 量 | 单位 | 后缀 | 例 |
  |---|---|---|---|
  | 光程差 / 延迟量 | nm | `_nm` | `retardation_nm_masked`、`x0_95_nm`、`A_max`(nm) |
  | 长度 / 宽度 / 厚度 / 半径 | mm | `_mm` | `thickness_mm`、`L_mm`、`hole_radius_mm`、`mm_per_px` |
  | 时长 | s | `_s` | `heating_duration_s` |
  | 温差 / 调温幅度上限 | ℃ | `_c` | `adjacent_zone_max_delta_c`、`single_step_max_delta_c` |
  - **温度**统一 ℃（项目约定）：温度场字段 `zone_temps / temp_upper / temp_lower` 不带后缀但语义恒为 ℃；**温差类**才带 `_c`。
- **限值入 config**：分级/阈值一律写 `config/*.yaml`，**禁止硬编码进 `tools/`**（[CLAUDE.md](CLAUDE.md) 铁律 #4）。
- **缺值标 `TODO(plant)`**：安全/国标常量未知一律标记，代码遇 `TODO(plant)` 判「无法判定→不通过」，**绝不猜测**。

## 2. 标准字段名册

### 工艺参数 `ParamSet` / `ProcessParams`（单位：℃ / s / mm）
`zone_temps`(list, ℃)、`zone_roles`(list, 与 zone_temps 等长)、`temp_upper`/`temp_lower`(℃)、
`convection_speed`、`convection_ratio_upper_lower`、`oscillation_speed`、`oscillation_amplitude`、
`heating_duration_s`(s)、`glass_type`、`thickness_mm`(mm)、`quality_mode`。

### 校验结果 `CheckResult`
`within_limits`(bool)、`blow_up_risk`(bool)、`gradient_ok`(bool)、`violations`(list[str]，含规则编号 C1/C2/C3/安全)。

### 指标 / 掩膜（[tools/metrics.py](tools/metrics.py)）
`retardation_nm_masked`(nm)、`mm_per_px`、`L_mm`/`W_mm`、`thickness_mm`、`hole_radius_mm`、
`La`/`Wa`(边缘带宽 mm)、`x0_95`(方法4.2)、`iso_t`(方法4.3, 阈值 `T`=75nm)、`ccp`(方法4.4)、`grade`。

### config 键
- `grading.yaml`：`x0_95_nm[].{thickness_mm,A_max,B_max}`、`iso_t_pct`、`ccp`。
- `thresholds.yaml`：`gradient.{adjacent_zone_max_delta_c,single_step_max_delta_c}`、`thickness_duration`、`convection`、`safety.{blowup_rule,max_gradient}`。
- `ccp_reference.yaml`：`c_max`、`cp_max`。

### 受控枚举值
- `glass_type` ∈ {`ultra_clear`(超白), `clear`(普白)}。
- `quality_mode` ∈ {`high_quality`(高质量), `high_efficiency`(高效率)}。
- `zone_roles` 取值含 {`center`, `edge`, …}（其余角色待 `TODO(plant)` 真实分区定义）。

### 归档样本分桶键（L5 `ArchiveSample`）
`thickness_mm`(厚度)、`glass_type`(品类)、`quality_mode`(质量模式)、`is_ground_truth`(bool)。

### 炉体身份与标注扩展（ArchiveSample v2，2026-07-02）
- `furnace_id`(str)：炉子标识；**身份缺省用 `"unknown"`**（不是安全限值，故不用 TODO(plant) 字串）。
- `furnace_config`(FurnaceConfig|None)：炉体配置快照（`schemas/furnace.py`；铭牌未知项一律 None）。
- `operator_id` / `repeat_group_id` / `condition_note`(str|None)：老师傅工号 / 一致性重复组号 / 工况备注。
- `MetricRecord.fringe_score_0_100`(float|None, 0–100)：**外部**应力斑分布打分（独立功能评好随数据到达，本工程只承接不计算）。
- `ARCHIVE_SCHEMA_VERSION`(模块常量)：只进数据包 manifest，不进样本本体。

### 同步与版本（schemas/datapack.py / tools/model_registry.py）
- 数据包去重键：`(furnace_id, sample_id)` + `content_sha256`；冲突（同键异内容）绝不静默覆盖。
- 契约指纹：`feature_schema_sha256` / `delta_fields_sha256`（training/targets.py），模型包导入必校验。
- config 键：`furnaces.yaml: furnaces[].{furnace_id,zone_count,zone_layout,fan_count,nameplate}`；
  `sync.yaml: {furnace_id,drop_dir,cloud.{provider,endpoint,auth_ref}}`；
  `training.yaml: {grade_scores,min_train_samples,val_frac,test_frac,gate.{max_param_mae,regression_tolerance}}`；
  `dialogue_rules.yaml: {intents,field_aliases}`（顺序即路由优先级）。

## 3. 受批准大写记号（pep8-naming 例外）
国标 / GLCM 约定俗成的大写符号，已在 [pyproject.toml](pyproject.toml) `extend-ignore-names` 放行，**仅限**下表，新增需先登记：

| 记号 | 含义 |
|---|---|
| `L` / `W` | 长度 / 宽度（像素或 mm 上下文内） |
| `T` | IsoT 阈值（75 nm） |
| `L_mm` / `W_mm` | 带单位的长 / 宽参数 |
| `La` / `Wa` | 边缘带宽（沿长 / 沿宽） |
| `R_px` | 像素半径 |
| `Ng` | GLCM 灰度级数（=8） |
| `Ca` / `CPa` | GLCM 对比度 / 聚类突出（四向平均） |
| `Cmax` / `CPmax` | CCP 参考最差样品标定量（`config/ccp_reference.yaml` 的 `c_max`/`cp_max`） |

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
`heating_duration_s`(s)、`glass_type`、`thickness_mm`(mm)、`quality_mode`；
可选字段（架构讨论稿 §4.2-2，2026-07-02 增，约束规则待 docs/03）：`convection_temp`(℃|None,
对流风温)、`fan_startup_logic`(str|None, 风机启动逻辑，形态待现场)。

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
- `glass_type` ∈ {`ultra_clear`(超白), `clear`(普白), `low_e`(Low-E), `coated`(镀膜·其他),
  `enameled`(彩釉), `patterned`(压花), `other`(其他)}。**2026-08-23 由 2 值扩为 7 值**——标注应用
  自 2026-07-26 起支持 7 品类且现场在用，原枚举导致 Low-E/彩釉/压花/镀膜 的 39 列 CSV 被
  `ingest_annotations` 整行拒收，这几类炉次无法回流。品类只是**分桶键**，不参与任何安全判定
  （`constraints.validate` 四条规则一条都不读它），放宽枚举不触及安全闸门。
  - **唯一真值源** = `schemas/process_params.py` 的 `GLASS_TYPES` / `GLASS_TYPE_ZH` / `GlassType`，
    新增品类只改那一处（三者一致性由 `tests/test_schemas.py::test_glass_type_sources_agree` 锁定）。
  - ⚠ `other` 的手输名在标注应用侧是 `glass_type_note`，**39 列表暂无该列** → 选 `other` 时具体
    品类名不会回流，只留 `other`。扩列需改冻结契约，未做。
  - ⚠ `tools/constraints.py:47` 的注释仍写 `# ultra_clear | clear`，**属已知过时**：该文件是安全闸门，
    在 AnnotationApp `server/vendor/` 有逐字节冻结副本，为保住「闸门一个字未改」的机器证明未动它。
    字段本身是 `str`、不校验品类，故无功能影响。
- `quality_mode` ∈ {`high_quality`(高质量), `high_efficiency`(高效率)}。
- `zone_roles` 取值含 {`center`, `edge`, …}（其余角色待 `TODO(plant)` 真实分区定义）。

### 归档样本分桶键（L5 `ArchiveSample`）
`thickness_mm`(厚度)、`glass_type`(品类)、`quality_mode`(质量模式)、`is_ground_truth`(bool)。

### 炉体身份与标注扩展（ArchiveSample v2，2026-07-02）
- `furnace_id`(str)：炉子标识；**身份缺省用 `"unknown"`**（不是安全限值，故不用 TODO(plant) 字串）。
- `furnace_config`(FurnaceConfig|None)：炉体配置快照（`schemas/furnace.py`；铭牌未知项一律 None）。
  含 `commissioning_date`/`last_overhaul_date`(date|None)——老化特征的原料，用 `tools/furnace_setup.py` 向导录入。
- **特征契约 26 维**（2026-07-02 尾部追加 4 维，勿插队）：`furnace_age_years`/`furnace_age_present`/
  `days_since_overhaul`/`overhaul_present`；日期缺失或晚于样本时刻（脏）→ 0+presence=0。
- **人读输出拍板**：工艺参数=Excel 参数单（`tools/param_sheet.py`），建议/说明=中文文本；未过闸门的单子标注"禁止照此操作"。
- `operator_id` / `repeat_group_id` / `condition_note`(str|None)：老师傅工号 / 一致性重复组号 / 工况备注。
- `MetricRecord.fringe_score_0_100`(float|None, 0–100)：**外部**应力斑分布打分（独立功能评好随数据到达，本工程只承接不计算）。
- `ARCHIVE_SCHEMA_VERSION`(模块常量)：只进数据包 manifest，不进样本本体。

### 研判输出层（advisor/，2026-07-02）
- `AdvisoryReport`：讨论稿 §4.2 六项输出的统一装配（`params`/`energy`/`maintenance`/`loading`/`attribution` + `environment`/`optical` 插槽）。
- `SectionStatus{ok, missing}`：缺真值节如实 `cannot_determine`；未标定参考值一律 `is_calibrated=False`。
- 输入载体（schemas/inputs.py，源=架构讨论稿 §2）：`EnvironmentInput{workshop_temp, workshop_humidity_pct, glass_inlet_temp, season_note}`、
  `OpticalFeatureSlots{optical_power_mdpt, bow_height_mm, wave_amplitude_mm}`（光焦度/弓波插槽，未接恒 None）、
  `EquipmentUsage{furnace_id, run_hours, changeover_count, load_frac, source_note}`。
- config 键：`energy.yaml{model,ref_temp_c,kw_per_zone_c,fan_kw_per_speed,base_kw}`、
  `maintenance.yaml{weights.{hours,cycles,load},service_wear_threshold,components[].{name,rated_hours,rated_cycles}}`、
  `loading.yaml{strategy,bed_length_mm,bed_width_mm,min_gap_mm}`、`attribution.yaml{rules[].{signal,op,threshold,issue}}`。

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

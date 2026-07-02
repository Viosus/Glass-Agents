# 模型版本账本（人读）

多头核心/参数头的版本登记表——只有指标与哈希，**无任何配方数值**，可入 git。
机器可读账本在 `models/registry/registry.jsonl`（gitignore，不入库）；
版本产生与晋级流程见 `docs/同步与自我迭代.md`。

状态语义：**候选** = 已登记未过门；**晋级** = 过中心侧第一重门（`tools/model_registry.promote`）；
**激活** = 炉侧本地留存验证再过第二重门（`activate`），真正开始出建议。

| model_id | 登记日期 | 训练样本量 | 门 MAE(param) | weights sha256 | 状态 |
|---|---|---|---|---|---|

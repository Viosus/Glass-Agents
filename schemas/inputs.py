"""五类输入 schema（docs/04 §2.1–2.5）—— 多数被阻塞，留占位不杜撰。

⚠️ `docs/04` 当前不存在，五类输入（2.1–2.5）的精确字段未规格化。按"缺值不杜撰"铁律：
- **已具体定义**：工艺参数输入 = `ProcessParams`（见 schemas/process_params.py）。
- **其余四类**：字段待 `docs/04` → 一律 `TODO(plant)`，**不臆造结构**。拿到 docs/04 后在此补全。

TODO(plant): docs/04 §2.1–2.5 五类输入精确字段：
  - [ ] 2.1 ?（待 docs/04）
  - [ ] 2.2 ?
  - [ ] 2.3 ?
  - [ ] 2.4 ?
  - [ ] 2.5 ?
（哪一节对应"工艺参数输入"也以 docs/04 为准，当前不臆断编号映射。）
"""

from __future__ import annotations

from schemas.process_params import ProcessParams

# 已具体定义的输入：工艺参数（其余四类待 docs/04 规格化）
ProcessParamsInput = ProcessParams

__all__ = ["ProcessParamsInput"]

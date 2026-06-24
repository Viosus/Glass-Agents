"""数据层 schema（pydantic）：工艺参数输入、归档样本 L5、分桶统计。

真值/限值不在此层；校验只管"结构与类型是否合法"。安全/工艺判定仍走 tools/constraints.py。
"""

#!/usr/bin/env python
"""PreToolUse 硬闸门：拦截"测试不过就 git commit"。

规则 > AI：检测到 git commit 命令时先跑 pytest，失败则退出码 2 拦截提交。
由 .claude/settings.json 用 venv 的 python 启动，故 sys.executable 即 venv 解释器，
pytest 必然可用（见 CLAUDE.md / requirements.txt）。
"""

import json
import os
import subprocess
import sys

# 确定性 UTF-8 输出（Claude Code 按 UTF-8 读取 hook 输出；避免 Windows 控制台 GBK 乱码）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool_input = data.get("tool_input") or {}
# Bash 工具与 PowerShell 工具都用 "command" 字段
cmd = tool_input.get("command", "") or ""
if "git commit" not in cmd:
    sys.exit(0)

proj = os.environ.get("CLAUDE_PROJECT_DIR", ".")
r = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/"],
    cwd=proj,
    capture_output=True,
    text=True,
)
if r.returncode != 0:
    sys.stderr.write("提交被拦截：安全/指标测试未通过（规则 > AI）。请先修复再提交。\n")
    sys.stderr.write((r.stdout + r.stderr)[-1500:])
    sys.exit(2)

sys.exit(0)

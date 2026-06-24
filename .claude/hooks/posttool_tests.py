#!/usr/bin/env python
"""PostToolUse 快速反馈：编辑核心代码后跑 pytest（非阻塞）。

仅当被编辑文件落在 tools/ tests/ config/ 时触发；测试失败时把结果作为
additionalContext 回灌，便于及时修复。永远 exit 0（不阻塞编辑）。
"""

import json
import os
import subprocess
import sys

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
path = tool_input.get("file_path", "") or ""
# Windows 路径用反斜杠，归一化后再匹配，避免漏判
norm = path.replace("\\", "/")
if not any(seg in norm for seg in ("/tools/", "/tests/", "/config/")):
    sys.exit(0)

proj = os.environ.get("CLAUDE_PROJECT_DIR", ".")
r = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/"],
    cwd=proj,
    capture_output=True,
    text=True,
)
if r.returncode != 0:
    ctx = "编辑 %s 后核心测试未通过，请先修复：\n%s" % (path, (r.stdout + r.stderr)[-1200:])
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": ctx,
        }
    }))

sys.exit(0)

"""对话壳 REPL 入口：终端里多轮聊（改参→闸门→中文呈现；问工艺；看样片指标）。

用法：
  & "D:\\Glass Agents\\.venv\\Scripts\\python.exe" llm_roles\\run_dialogue.py --no-llm
  & ... run_dialogue.py --no-llm --baseline data\\baseline.json      # 载入基准配方(ProcessParams JSON)
  & ... run_dialogue.py --no-llm --sample data\\archive\\x.json      # 载入样片(ArchiveSample JSON)
--no-llm = 全程确定性（不加载 GGUF，秒开）；去掉则加载本地 Qwen 做润色与兜底意图分类。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 支持直接 `python llm_roles/run_dialogue.py`

from llm_roles.dialogue import DialogueState, respond, state_from_sample  # noqa: E402
from schemas.archive import read_sample  # noqa: E402
from schemas.process_params import ProcessParams  # noqa: E402


def build_state(args: argparse.Namespace) -> DialogueState:
    """按启动参数初始化会话状态（样本优先；其次基准配方；否则空会话）。"""
    if args.sample is not None:
        state = state_from_sample(read_sample(args.sample))
        if args.furnace_id:
            state.furnace_id = args.furnace_id
        return state
    state = DialogueState(furnace_id=args.furnace_id or "unknown")
    if args.baseline is not None:
        state.params = ProcessParams.model_validate_json(Path(args.baseline).read_text(encoding="utf-8"))
    return state


def main() -> int:
    """REPL：读一行→respond→打印；「退出」结束。"""
    ap = argparse.ArgumentParser(description="钢化炉调参对话壳（数值全来自确定性工具，LLM 只做语言壳）")
    ap.add_argument("--no-llm", action="store_true", help="全程确定性，不加载 GGUF")
    ap.add_argument("--furnace-id", default="", help="炉子标识（展示与留痕）")
    ap.add_argument("--baseline", type=Path, default=None, help="基准配方 ProcessParams JSON")
    ap.add_argument("--sample", type=Path, default=None, help="样片 ArchiveSample JSON（含指标/分数）")
    args = ap.parse_args()

    llm = None
    if not args.no_llm:
        from llm_roles._llm import load_llm

        print("加载本地 LLM（GGUF，CPU 可能需要十几秒）…")
        llm = load_llm()

    state = build_state(args)
    llm_mark = "开" if llm is not None else "关"
    print(f"对话开始（炉: {state.furnace_id}；LLM: {llm_mark}）。说「帮助」看用法，「退出」结束。")
    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        state, reply = respond(state, text, llm=llm)
        print(f"助手> {reply}\n")
        if state.turns and state.turns[-1][0] == text and reply.startswith("已结束对话"):
            break
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    sys.exit(main())

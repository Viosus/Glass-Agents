"""本地代码审查器：双通道审查 .py 文件。

通道一·确定性预检（AST + 正则，纯计算，0 模型，0 误报）→ 直接报 ❌：
  - 云端调用：import 黑名单 + 出现即违规的字面子串（API密钥/URL 等）；
  - 中文注释：函数/类缺中文注释或中文 docstring；
  - 命名铁律：函数/变量 snake_case、类 PascalCase、常量 UPPER_SNAKE（CONVENTIONS.md）。
通道二·模型补漏（Qwen2.5 GGUF）→ 标 ⚠️ 需人工确认：
  - 专补正则漏掉的【绕过变种】：动态导入、字符串拼接/混淆、反射、编码绕过等。
规则黑/白名单来自 config/review_rules.yaml（数据驱动，禁硬编码进 tools/）。

用法：
  & "D:\\Glass Agents\\.venv\\Scripts\\python.exe" tools\\review.py <file.py> [more.py ...]
  type code.py | & "...python.exe" tools\\review.py            # 无文件参数则读 stdin
可选：--prompt <模板> --model <gguf> --rules <yaml> --n-ctx 4096 --max-tokens 1024
      --gpu-layers <n>  --no-model（只跑确定性通道，快）

退出码：确定性发现违规(❌)→1；无 ❌（可能有 ⚠️）→0；运行错误→2。
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

import yaml  # 本地库（pyyaml）

# 支持直接 `python tools/review.py`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # Windows 控制台默认 GBK，强制 UTF-8 避免中文/emoji 乱码
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

DEFAULT_PROMPT_PATH = ROOT / "config" / "review_prompt.md"
DEFAULT_MODEL_PATH = ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_RULES_PATH = ROOT / "config" / "review_rules.yaml"

# 命名铁律正则
RE_SNAKE = re.compile(r"^_*[a-z][a-z0-9_]*$")        # snake_case / 私有前缀 / dunder
RE_UPPER_SNAKE = re.compile(r"^_*[A-Z][A-Z0-9_]*$")  # 模块级常量（允许前导下划线的私有常量）
RE_PASCAL = re.compile(r"^_?[A-Z][a-zA-Z0-9]*$")     # 类名
RE_CJK = re.compile(r"[一-鿿]")              # 中文字符


# ============ 工具 ============
def _load_rules(rules_path: Path) -> dict:
    """读取确定性预检规则（云端黑名单、字面子串、受批准大写名）。"""
    return yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}


def _is_approved(name: str, approved: set[str]) -> bool:
    """判断标识符是否为受批准大写记号（CONVENTIONS.md 登记表）。"""
    return name in approved


def _is_cloud_exempt(path: Path | None, rules: dict) -> bool:
    """该文件是否在云端调用豁免路径内（cloud_exempt_paths，前缀匹配、大小写不敏感）。

    仅豁免「云端调用」类规则（import 黑名单 + 联网字面子串）——GlassApp 联网层
    （打分服务器/账号服务/上传器）的产品本职；中文注释与命名铁律照查。
    stdin（无路径）不豁免。
    """
    if path is None:
        return False
    norm = str(path.resolve()).replace("\\", "/").lower()
    for raw in rules.get("cloud_exempt_paths", []) or []:
        entry = str(raw).replace("\\", "/").lower()
        if not entry:
            continue
        # 目录条目（/ 结尾）按前缀匹配；文件条目须精确相等——避免
        # "uploader.py" 顺带豁免 "uploader.py.bak" 一类同前缀文件
        if entry.endswith("/"):
            if norm.startswith(entry):
                return True
        elif norm == entry:
            return True
    return False


# ============ 通道一：确定性预检（AST + 正则）============
def detect_cloud_imports(tree: ast.AST, rules: dict) -> list[tuple[int, str]]:
    """检测 import 黑名单（云端/网络库），按模块根名与点号命名空间匹配。"""
    blacklist = set(rules.get("cloud_import_blacklist", []))
    dotted = list(rules.get("cloud_import_blacklist_dotted", []))
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        for mod in mods:
            root = mod.split(".")[0]
            hit = root in blacklist or any(mod == d or mod.startswith(d + ".") for d in dotted)
            if hit:
                findings.append((node.lineno, f"云端调用 - 导入网络/云库 '{mod}'"))
    return findings


def detect_cloud_strings(lines: list[str], rules: dict) -> list[tuple[int, str]]:
    """按行扫描出现即违规的字面子串（API密钥/URL 等），大小写不敏感。"""
    patterns = [str(p) for p in rules.get("cloud_string_patterns", [])]
    findings: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        for pat in patterns:
            if pat.lower() in low:
                findings.append((i, f"云端调用 - 出现联网字面 '{pat}'"))
    return findings


def _cjk_comment_lines(source: str) -> set[int]:
    """收集所有“含中文的注释”所在行号（tokenize 解析 # 注释）。"""
    cjk_lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and RE_CJK.search(tok.string):
                cjk_lines.add(tok.start[0])
    except Exception:
        pass
    return cjk_lines


def detect_missing_chinese_comments(tree: ast.AST, source: str) -> list[tuple[int, str]]:
    """检测函数/类是否缺中文注释：中文 docstring 或函数体范围内含中文 # 注释即视为有。"""
    cjk_lines = _cjk_comment_lines(source)
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node)
        has_cjk_doc = bool(doc and RE_CJK.search(doc))
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        has_cjk_comment = any(node.lineno - 1 <= ln <= end for ln in cjk_lines)
        if not (has_cjk_doc or has_cjk_comment):
            kind = "类" if isinstance(node, ast.ClassDef) else "函数"
            findings.append((node.lineno, f"缺少中文注释 - {kind} '{node.name}' 无中文注释/docstring"))
    return findings


def detect_naming(tree: ast.AST, rules: dict) -> list[tuple[int, str]]:
    """命名铁律检查：函数 snake_case、类 PascalCase、参数 snake_case、赋值名 snake/UPPER_SNAKE。"""
    approved = set(rules.get("approved_uppercase_names", []))
    findings: list[tuple[int, str]] = []
    seen: set[tuple[str, int]] = set()

    # 模块级赋值名：允许 PascalCase（类型别名/哨兵，如 Grade = Literal[...]）
    module_level: set[str] = set()
    for top in ast.iter_child_nodes(tree):
        if isinstance(top, ast.Assign):
            module_level.update(t.id for t in top.targets if isinstance(t, ast.Name))
        elif isinstance(top, ast.AnnAssign) and isinstance(top.target, ast.Name):
            module_level.add(top.target.id)

    for node in ast.walk(tree):
        # 函数名 → snake_case
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not RE_SNAKE.match(node.name) and not _is_approved(node.name, approved):
                findings.append((node.lineno, f"命名不合规 - 函数 '{node.name}' 应为 snake_case"))
            # 参数 → snake_case（豁免 self/cls 与受批准大写名）
            args = node.args
            all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
            if args.vararg:
                all_args.append(args.vararg)
            if args.kwarg:
                all_args.append(args.kwarg)
            for a in all_args:
                an = a.arg
                if an in ("self", "cls") or RE_SNAKE.match(an) or _is_approved(an, approved):
                    continue
                findings.append((a.lineno, f"命名不合规 - 参数 '{an}' 应为 snake_case"))
        # 类名 → PascalCase
        elif isinstance(node, ast.ClassDef):
            if not RE_PASCAL.match(node.name):
                findings.append((node.lineno, f"命名不合规 - 类 '{node.name}' 应为 PascalCase"))
        # 赋值/绑定目标 → snake_case 或 UPPER_SNAKE（常量）或受批准大写名
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            nm = node.id
            if set(nm) <= {"_"}:  # 占位符 _ / __：Python 惯例的丢弃变量，豁免
                continue
            ok = RE_SNAKE.match(nm) or RE_UPPER_SNAKE.match(nm) or _is_approved(nm, approved)
            if nm in module_level and RE_PASCAL.match(nm):  # 模块级类型别名允许 PascalCase
                ok = True
            if not ok and (nm, node.lineno) not in seen:
                seen.add((nm, node.lineno))
                findings.append((node.lineno, f"命名不合规 - 变量 '{nm}' 应为 snake_case"))
    return findings


def run_deterministic(file_name: str, source: str, rules: dict,
                      cloud_exempt: bool = False) -> list[str]:
    """跑确定性通道，返回排好序的 ❌ 行（语法错误则返回单条解析失败）。

    cloud_exempt=True（文件命中 cloud_exempt_paths）时跳过云端调用两项检查，
    其余规则照查。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"❌ {file_name}: {exc.lineno} - 解析失败 - 语法错误: {exc.msg}"]

    lines = source.splitlines()
    items: list[tuple[int, str]] = []
    if not cloud_exempt:
        items += detect_cloud_imports(tree, rules)
        items += detect_cloud_strings(lines, rules)
    items += detect_missing_chinese_comments(tree, source)
    items += detect_naming(tree, rules)
    items.sort(key=lambda t: t[0])
    return [f"❌ {file_name}: {ln} - {msg}" for ln, msg in items]


# ============ 通道二：模型补漏（Qwen2.5 GGUF）============
def _detect_gpu_layers() -> int:
    """探测是否可用 CUDA：llama-cpp 为 GPU 构建且 torch 报告 CUDA 可用才 offload(-1)，否则 CPU(0)。"""
    try:
        import llama_cpp  # 本地库

        if not llama_cpp.llama_supports_gpu_offload():
            return 0
        import torch  # 本地库

        return -1 if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _make_llm(model_path: Path, n_ctx: int, gpu_layers: int):
    """加载本地 GGUF 模型（仅在需要时导入 llama_cpp，便于无依赖时给清晰报错）。"""
    from llama_cpp import Llama  # 本地库；CPU/GPU 由编译后端决定

    return Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=gpu_layers, verbose=False)


def _number_lines(code: str) -> str:
    """给代码逐行加行号，便于模型按行号定位风险。"""
    rows = code.splitlines()
    width = len(str(len(rows))) if rows else 1
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(rows, start=1))


def _build_model_prompt(template: str, file_name: str, code: str) -> str:
    """构造模型通道提示：复用现有规则模板，并追加“只补漏变种、用规定格式”的侧重说明。"""
    body = f"# 文件名: {file_name}\n{_number_lines(code)}"
    base = template.replace("{code}", body)
    focus = (
        "\n\n【模型通道·只补漏绕过变种】\n"
        "确定性预检已覆盖：字面 import 黑名单、含 API密钥/URL 等字面字符串、命名铁律——请勿重复这些。\n"
        "你只找正则可能漏掉的【绕过变种】：\n"
        "- 动态导入：__import__ / importlib.import_module 加载网络库\n"
        "- 字符串拼接/编码(base64 等)混淆构造 URL 或域名；getattr/反射；exec/eval 联网\n"
        "输出规则：每条风险占一行，必须以 RISK 开头，格式：RISK <行号> <风险简述>\n"
        "没有可疑变种就只输出一行：NONE\n"
        "不要输出代码、不要解释、不要输出其它内容。\n"
    )
    return base + focus


def review_with_model(llm, template: str, file_name: str, code: str, max_tokens: int) -> list[str]:
    """跑模型补漏通道，解析模型输出为 ⚠️ 行（NONE/空 → 无）。"""
    prompt = _build_model_prompt(template, file_name, code)
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = out["choices"][0]["message"]["content"].strip()
    warnings: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        # 只认以 RISK 开头的行，过滤模型回吐的源码/解释；NONE 表示无
        if not line.upper().startswith("RISK"):
            continue
        desc = line[4:].strip().lstrip(":：").strip()
        if desc:
            warnings.append(f"⚠️ {file_name}: {desc}")
        if len(warnings) >= 10:  # 封顶，防模型跑飞刷屏
            break
    return warnings


# ============ 主流程 ============
def _parse_args(argv: list[str]) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="本地双通道代码审查器（确定性 ❌ + 模型 ⚠️）")
    parser.add_argument("files", nargs="*", help="待审查的 .py 文件路径；省略则读 stdin")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH), help="模型通道提示模板")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="本地 GGUF 模型路径")
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH), help="确定性规则 yaml")
    parser.add_argument("--n-ctx", type=int, default=8192, help="上下文长度")
    parser.add_argument("--max-tokens", type=int, default=1024, help="单次审查最大生成 token")
    parser.add_argument("--gpu-layers", type=int, default=None, help="offload 层数；默认自动探测")
    parser.add_argument("--no-model", action="store_true", help="只跑确定性通道（快，跳过 LLM）")
    return parser.parse_args(argv)


def run(argv: list[str]) -> int:
    """主流程：确定性通道(❌) + 可选模型通道(⚠️)，合并报告并返回退出码。"""
    args = _parse_args(argv)
    rules = _load_rules(Path(args.rules))

    # 收集 (文件名, 代码, 云端豁免标志)
    targets: list[tuple[str, str, bool]] = []
    if args.files:
        for f in args.files:
            p = Path(f)
            if not p.exists():
                print(f"❌ 文件不存在：{p}", file=sys.stderr)
                return 2
            targets.append((p.name, p.read_text(encoding="utf-8"), _is_cloud_exempt(p, rules)))
    else:
        targets.append(("<stdin>", sys.stdin.read(), False))

    # 模型通道（按需加载，可 --no-model 跳过）
    llm = None
    template = ""
    if not args.no_model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ 模型文件不存在：{model_path}", file=sys.stderr)
            return 2
        template = Path(args.prompt).read_text(encoding="utf-8")
        gpu_layers = args.gpu_layers if args.gpu_layers is not None else _detect_gpu_layers()
        backend = "GPU(CUDA)" if gpu_layers != 0 else "CPU"
        print(f"[review] 模型={model_path.name} 后端={backend} n_ctx={args.n_ctx}", file=sys.stderr)
        llm = _make_llm(model_path, args.n_ctx, gpu_layers)

    has_violation = False
    for file_name, code, cloud_exempt in targets:
        print(f"\n===== 审查: {file_name} =====")
        hard = run_deterministic(file_name, code, rules, cloud_exempt=cloud_exempt)
        print("[确定性·硬规则]" + ("（云端调用规则已豁免：cloud_exempt_paths）" if cloud_exempt else ""))
        if hard:
            has_violation = True
            for line in hard:
                print(line)
        else:
            print("✅ 确定性通道无违规")

        if llm is not None:
            print("[模型·补漏]")
            try:
                soft = review_with_model(llm, template, file_name, code, args.max_tokens)
                if soft:
                    for line in soft:
                        print(line)
                else:
                    print("✅ 无额外风险")
            except Exception as exc:  # 模型异常(如上下文超限)降级为跳过，不影响确定性结论
                print(f"⚠️(跳过) 模型通道异常，仅确定性结论有效：{exc}")

    return 1 if has_violation else 0


if __name__ == "__main__":
    try:
        sys.exit(run(sys.argv[1:]))
    except Exception as exc:  # 运行错误与“审查发现违规”用不同退出码区分
        print(f"❌ review 运行错误：{exc}", file=sys.stderr)
        sys.exit(2)

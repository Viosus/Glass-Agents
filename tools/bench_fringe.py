"""fringe_scoring 性能基准：逐照片单核计时 + 多进程吞吐（片/秒）。

用法（venv python）：
    python tools/bench_fringe.py <照片目录> [--config yaml] [--workers N] [--rounds N]
- 单核段：逐张顺序打分，报每张耗时与合计片/秒；
- 吞吐段：照片列表 ×rounds 轮铺进 ProcessPoolExecutor（workers 默认=CPU 核数），
  报稳态吞吐——对应产线"每秒 N 片传输"的服务端并行口径。
纯测量工具：不含任何工艺限值；分数本身不在此校验（恒等性由 tests/ 黄金对拍保证）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# 直接以脚本运行（含 spawn 子进程重导 __main__）时把仓库根加进 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_gray(path: str) -> np.ndarray:
    """读灰度图；np.fromfile+imdecode 绕开 cv2.imread 的 Windows 非 ASCII 路径缺陷。"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"bench_fringe: 无法读取 {path}")
    return img


def score_one(args: tuple[str, dict]) -> tuple[int, float]:
    """子进程任务：一张整床照片 → (片数, 秒)。顶层函数保证 Windows spawn 可 pickle。"""
    from fringe_scoring.sheets import score_sheets

    path, cfg = args
    img = _load_gray(path)
    t0 = time.perf_counter()
    res = score_sheets(img, cfg)
    return res.n_sheets, time.perf_counter() - t0


def main() -> None:
    """命令行入口：单核顺序段 + 多进程吞吐段，各报 片/秒。"""
    parser = argparse.ArgumentParser(description="fringe_scoring 性能基准")
    parser.add_argument("photo_dir", help="整床照片目录（*.png/*.PNG/*.jpg）")
    parser.add_argument("--config", default=None, help="配置 yaml（默认仓库 fringe_scoring.yaml）")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="进程数")
    parser.add_argument("--rounds", type=int, default=8, help="吞吐段照片列表重复轮数")
    args = parser.parse_args()

    from fringe_scoring.score import load_config

    cfg = load_config(args.config) if args.config else load_config()
    photos = sorted(
        str(p) for p in Path(args.photo_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )
    if not photos:
        raise SystemExit(f"bench_fringe: {args.photo_dir} 下没有照片")

    print(f"== 单核顺序（{len(photos)} 张） ==")
    total_sheets, total_s = 0, 0.0
    for p in photos:
        n, dt = score_one((p, cfg))
        total_sheets += n
        total_s += dt
        print(f"  {Path(p).name[:40]}: {n} 片 {dt:.2f}s")
    print(f"  合计 {total_sheets} 片 / {total_s:.1f}s = {total_sheets / total_s:.2f} 片/s")

    tasks = [(p, cfg) for p in photos] * args.rounds
    print(f"== 多进程吞吐（workers={args.workers}, {len(tasks)} 张任务） ==")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        sheets = sum(n for n, _ in pool.map(score_one, tasks))
    wall = time.perf_counter() - t0
    print(f"  {sheets} 片 / {wall:.1f}s 墙钟 = {sheets / wall:.2f} 片/s")


if __name__ == "__main__":
    main()

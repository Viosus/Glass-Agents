"""应力斑分布打分 CLI：读图 → 打分 → 打印（可选可视化四联图 / 合成 demo 对照）。

用法（venv python，见 CLAUDE.md 铁律 #1）：
  python fringe_scoring/run_score.py <图.npy|.png|.tif> [更多图...]
  python fringe_scoring/run_score.py <图> --viz-dir out/     # 另存可视化
  python fringe_scoring/run_score.py --demo                  # 合成三图对照演示
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 支持直接 `python fringe_scoring/run_score.py`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fringe_scoring.score import FringeScoreResult, load_config, score_fringe_distribution  # noqa: E402
from fringe_scoring.synth import make_glass_image, warp_into_quad  # noqa: E402

try:  # Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def print_result(name: str, res: FringeScoreResult) -> None:
    """把一张图的打分结果按行打印（主输出 + 诊断量）。"""
    bands = res.border_bands
    centrality = f"{res.centrality:.3f}" if res.centrality is not None else "—(无斑)"
    print(f"\n===== {name} =====")
    print(f"  得分 score_0_100   : {res.score_0_100:.2f}（越高越好）")
    print(f"  原始惩罚 penalty   : {res.penalty_raw:.2f} / 最差 {res.penalty_max:.2f}")
    print(f"  斑面积占比         : {res.fringe_area_frac * 100:.2f}%（占非边框有效区）")
    print(f"  集中度 centrality  : {centrality}（斑深浅加权平均 w，越大越靠中心）")
    print(f"  边框带宽 px(上/下/左/右): {bands.top_px}/{bands.bottom_px}/{bands.left_px}/{bands.right_px}")
    if res.quad_corners_px is not None:
        pts = "; ".join(f"({x:.0f},{y:.0f})" for x, y in res.quad_corners_px)
        print(f"  四边形角点(x,y)    : {pts} → 已透视矫正")
    else:
        print("  四边形角点(x,y)    : —（整图即玻璃，未矫正）")


def save_viz(name: str, image: np.ndarray, res: FringeScoreResult, viz_dir: Path) -> Path:
    """另存可视化四联图：原图 / 深浅强度 s / 斑掩膜+边框带 / 位置权重 w。"""
    import matplotlib

    matplotlib.use("Agg")  # 无窗口环境直接落盘
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # Windows 中文字体
    plt.rcParams["axes.unicode_minus"] = False

    shown = res.warped_image if res.warped_image is not None else image  # 实际被打分的图
    overlay = np.zeros(shown.shape)  # 掩膜合成层：边框带=0.5、斑=1.0
    overlay[res.border_mask_arr] = 0.5
    overlay[res.fringe_mask] = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    title0 = "矫正后图" if res.quad_corners_px is not None else "原图"
    panels = [
        (shown, title0, "gray"),
        (res.intensity_s, "深浅强度 s", "magma"),
        (overlay, "斑(亮)+边框带(灰)", "viridis"),
        (res.weight_map, "位置权重 w", "coolwarm"),
    ]
    for ax, (layer, title, cmap) in zip(axes, panels):
        im = ax.imshow(layer, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{name} · score={res.score_0_100:.2f}")
    fig.tight_layout()

    viz_dir.mkdir(parents=True, exist_ok=True)
    out = viz_dir / f"{Path(name).stem}_score.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def demo_images() -> list[tuple[str, np.ndarray]]:
    """合成四张对照图：干净玻璃 / 中心重斑 / 中心重斑+宽窄不一重边框 / 平行四边形玻璃。"""
    clean = make_glass_image(seed=1)
    center_blob = [(0.5, 0.5, 0.08, 6.0)]
    blob = make_glass_image(blobs=center_blob, seed=1)
    blob_frame = make_glass_image(
        blobs=center_blob, frame_widths_frac=(0.05, 0.12, 0.08, 0.03), frame_amp=40.0, seed=1
    )
    # 大图裁切场景：同样的中心重斑玻璃被贴成平行四边形，四边形外为恒定填充
    quad_corners = np.array([[70.0, 12.0], [360.0, 30.0], [310.0, 246.0], [20.0, 228.0]])
    blob_quad = warp_into_quad(
        make_glass_image(noise_sigma=0.0, blobs=center_blob, seed=1),
        quad_corners, (260, 380), fill_value=30.0, noise_sigma=1.0, seed=1,
    )
    return [
        ("demo1_干净玻璃", clean),
        ("demo2_中心重斑", blob),
        ("demo3_中心重斑+边框", blob_frame),
        ("demo4_平行四边形玻璃(中心重斑)", blob_quad),
    ]


def main(argv: list[str]) -> int:
    """CLI 主流程：解析参数 → 逐图打分打印（可选可视化）。"""
    parser = argparse.ArgumentParser(description="应力斑分布打分（淡且均匀 → 高分）")
    parser.add_argument("files", nargs="*", help="图像路径（.npy 或位图）；配 --demo 可省略")
    parser.add_argument("--config", default=None, help="自定义配置 yaml（默认 config/fringe_scoring.yaml）")
    parser.add_argument("--viz-dir", default=None, help="另存可视化四联图的目录")
    parser.add_argument("--demo", action="store_true", help="跑合成四图对照演示")
    parser.add_argument(
        "--corners", default=None,
        help='显式玻璃角点 "x,y;x,y;x,y;x,y"（跳过自动检测；仅作用于 files 传入的图）',
    )
    parser.add_argument("--no-quad", action="store_true", help="关闭四边形自动检测（按整图即玻璃处理）")
    args = parser.parse_args(argv)

    if not args.files and not args.demo:
        parser.error("请给出图像路径，或用 --demo 跑合成演示")

    cfg = load_config(args.config)
    if args.no_quad and "quad" in cfg:
        cfg["quad"]["auto_detect"] = False

    corners = None
    if args.corners:  # "x,y;x,y;x,y;x,y" → (4,2)
        corners = np.array(
            [[float(v) for v in pt.split(",")] for pt in args.corners.split(";")], dtype=float
        )
        if corners.shape != (4, 2):
            parser.error("--corners 需要 4 个 x,y 点，用分号分隔")

    # (名字, 图像, 是否允许套用 --corners)：显式角点只作用于 files 传入的图
    targets: list[tuple[str, np.ndarray, bool]] = []
    if args.demo:
        targets += [(name, img, False) for name, img in demo_images()]
    if args.files:
        from schemas.image_io import load_array  # 复用统一读图入口（npy=nm，位图=像素强度）

        for f in args.files:
            targets.append((Path(f).name, load_array(f), True))

    for name, image, is_file in targets:
        res = score_fringe_distribution(
            image, config=cfg, quad_corners=corners if is_file else None
        )
        print_result(name, res)
        if args.viz_dir:
            out = save_viz(name, image, res, Path(args.viz_dir))
            print(f"  可视化 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""应力斑分布打分 CLI：读图 → 打分 → 打印（可选可视化四联图 / 合成 demo 对照）。

用法（venv python，见 CLAUDE.md 铁律 #1）：
  python fringe_scoring/run_score.py <图.npy|.png|.tif> [更多图...]
  python fringe_scoring/run_score.py <图> --viz-dir out/     # 另存可视化
  python fringe_scoring/run_score.py <整床照片> --sheets     # 一张照片多块玻璃：切片逐片打分
  python fringe_scoring/run_score.py <整床照片> --sheets --store-dir  # 逐片裁切图+分数落本地调参库
  python fringe_scoring/run_score.py --demo                  # 合成三图对照演示
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 支持直接 `python fringe_scoring/run_score.py`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fringe_scoring.score import FringeScoreResult, load_config, score_fringe_distribution  # noqa: E402
from fringe_scoring.sheets import SheetsScoreResult, score_sheets  # noqa: E402
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


def print_sheets_result(name: str, res: SheetsScoreResult) -> None:
    """整床结果打印：汇总一行 + 每片一到两行（阅读序；配了 indicators 段则加六指标行）。"""
    print(f"\n===== {name}（整床 {res.n_sheets} 片）=====")
    line = f"  整床汇总: score_min={res.score_min:.2f}（最差片） score_mean={res.score_mean:.2f}"
    if res.sheet_indicators is not None:
        line += f" | 六指标总分 min={res.total_min:.2f} mean={res.total_mean:.2f}"
    print(line)
    for i, r in enumerate(res.sheet_results, 1):
        cent = f"{r.centrality:.3f}" if r.centrality is not None else "—(无斑)"
        assert r.quad_corners_px is not None  # 多片模式逐片必带角点
        cx, cy = r.quad_corners_px.mean(axis=0)
        print(
            f"  片{i} @({cx:.0f},{cy:.0f}): score={r.score_0_100:.2f} "
            f"斑占比={r.fringe_area_frac * 100:.1f}% 集中度={cent}"
        )
        if res.sheet_indicators is not None:
            ind = res.sheet_indicators[i - 1]
            print(
                f"      六指标: 总分={ind.total_score:.1f} | "
                f"X0.95={ind.x095:.1f}{ind.x095_unit} 灰度方差={ind.gray_variance:.1f} "
                f"梯度均={ind.gradient_mean:.1f} 梯度方差={ind.gradient_variance:.1f} "
                f"CCP={ind.ccp_value:.3f} 位置评分={ind.position_score:.1f} "
                f"中心集中度={ind.center_concentration:.4f}"
            )


def save_sheets_viz(name: str, image: np.ndarray, res: SheetsScoreResult, viz_dir: Path) -> Path:
    """整床可视化：原图 + 每片四边形描边 + 片号/分数标注。"""
    import matplotlib

    matplotlib.use("Agg")  # 无窗口环境直接落盘
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # Windows 中文字体
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 12 * image.shape[0] / image.shape[1]))
    ax.imshow(image, cmap="gray")
    for i, r in enumerate(res.sheet_results, 1):
        corners = r.quad_corners_px
        assert corners is not None
        ring = np.vstack([corners, corners[:1]])  # 闭合描边
        ax.plot(ring[:, 0], ring[:, 1], color="yellow", linewidth=1.5)
        cx, cy = corners.mean(axis=0)
        ax.text(
            cx, cy, f"#{i}\n{r.score_0_100:.1f}", color="yellow", fontsize=11,
            ha="center", va="center",
            bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"},
        )
    ax.set_title(
        f"{name} · {res.n_sheets} 片 · min={res.score_min:.2f} mean={res.score_mean:.2f}"
    )
    ax.axis("off")
    fig.tight_layout()

    viz_dir.mkdir(parents=True, exist_ok=True)
    out = viz_dir / f"{Path(name).stem}_sheets.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# 本地调参库默认位置（已 gitignore：真实裁切图+分数不入版本库）
DEFAULT_STORE_DIR = ROOT / "fringe_scoring" / "local_store"


def dump_sheets_store(
    name: str, image: np.ndarray, res: SheetsScoreResult, store_dir: Path, cfg: dict
) -> Path:
    """整床结果落盘到本地调参库：每张照片一个子目录。

    内容：逐片矫正裁切图 PNG（文件名带分数，仅供目检——按 0–255 裁剪转 uint8，
    nm 尺度 .npy 输入会饱和）+ 斑分割二值掩膜 sheetNN_mask.png（白=判为应力斑）
    + 并排对照 sheetNN_overlay.png（裁切图｜掩膜）+ scores.json（分数/六指标/角点/
    诊断量/当次 config 快照）。数值真值以 json 为准。
    """
    import cv2

    def write_png(u8: np.ndarray, filename: str) -> None:
        """PNG 落盘（imencode+tofile 而非 imwrite：Windows 中文路径安全）。"""
        ok, buf = cv2.imencode(".png", u8)
        if not ok:
            raise ValueError(f"dump_sheets_store: PNG 编码失败（{filename}）")
        buf.tofile(str(folder / filename))

    folder = store_dir / Path(name).stem
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for i, r in enumerate(res.sheet_results, 1):
        assert r.warped_image is not None and r.quad_corners_px is not None  # 多片模式必有
        u8 = np.clip(np.round(r.warped_image), 0, 255).astype(np.uint8)
        crop_name = f"sheet{i:02d}_score{r.score_0_100:.1f}.png"
        write_png(u8, crop_name)

        # 斑分割可视化：二值掩膜（白=斑）+ 与裁切图并排对照（中间 4px 灰隔条）
        mask_u8 = (r.fringe_mask.astype(np.uint8)) * 255
        mask_name = f"sheet{i:02d}_mask.png"
        write_png(mask_u8, mask_name)
        gap = np.full((u8.shape[0], 4), 128, dtype=np.uint8)
        write_png(np.hstack([u8, gap, mask_u8]), f"sheet{i:02d}_overlay.png")

        bands = r.border_bands
        record = {
            "sheet_index": i,
            "crop_file": crop_name,
            "mask_file": mask_name,
            "score_0_100": r.score_0_100,
            "penalty_raw": r.penalty_raw,
            "penalty_max": r.penalty_max,
            "fringe_area_frac": r.fringe_area_frac,
            "centrality": r.centrality,
            "border_px": [bands.top_px, bands.bottom_px, bands.left_px, bands.right_px],
            "quad_corners_px": r.quad_corners_px.tolist(),
        }
        if res.sheet_indicators is not None:
            ind = res.sheet_indicators[i - 1]
            record["indicators"] = {
                "total_score": ind.total_score,
                "x095": ind.x095,
                "x095_unit": ind.x095_unit,
                "gray_variance": ind.gray_variance,
                "gradient_mean": ind.gradient_mean,
                "gradient_variance": ind.gradient_variance,
                "ccp": ind.ccp_value,
                "ccp_ca": ind.ccp_ca,
                "ccp_cpa": ind.ccp_cpa,
                "ccp_reference_is_plant_calibrated": ind.ccp_reference_is_plant_calibrated,
                "center_concentration": ind.center_concentration,
                "position_score": ind.position_score,
                "uniformity": ind.position_score,  # deprecated：旧消费方兼容，值=position_score
                "position_diagnostics": ind.position_diagnostics,
                "sub_scores": ind.sub_scores,
                "weights": ind.weights,
            }
        records.append(record)
    payload = {
        "source_image": name,
        "image_shape_hw": list(image.shape),
        "n_sheets": res.n_sheets,
        "score_min": res.score_min,
        "score_mean": res.score_mean,
        "total_min": res.total_min if res.sheet_indicators is not None else None,
        "total_mean": res.total_mean if res.sheet_indicators is not None else None,
        "config": cfg,
        "sheets": records,
    }
    with open(folder / "scores.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return folder


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
    parser.add_argument(
        "--sheets", action="store_true",
        help="整床多片模式：一张照片多块玻璃，先切片再逐片打分（与 --corners/--demo 互斥）",
    )
    parser.add_argument(
        "--store-dir", nargs="?", const=str(DEFAULT_STORE_DIR), default=None,
        help="整床模式落盘本地调参库：逐片裁切图+scores.json+总览图"
             "（不带值则用 fringe_scoring/local_store，已 gitignore；仅配 --sheets 用）",
    )
    args = parser.parse_args(argv)

    if not args.files and not args.demo:
        parser.error("请给出图像路径，或用 --demo 跑合成演示")
    if args.sheets and (args.corners or args.demo):
        parser.error("--sheets 与 --corners/--demo 互斥（多片模式逐片角点由检测产出）")
    if args.store_dir and not args.sheets:
        parser.error("--store-dir 仅在 --sheets 整床模式下有效")

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
        if args.sheets:
            sheets_res = score_sheets(image, config=cfg)
            print_sheets_result(name, sheets_res)
            if args.store_dir:
                folder = dump_sheets_store(name, image, sheets_res, Path(args.store_dir), cfg)
                save_sheets_viz(name, image, sheets_res, folder)  # 总览图一并入库便于对照
                print(f"  调参库 → {folder}")
            if args.viz_dir:
                out = save_sheets_viz(name, image, sheets_res, Path(args.viz_dir))
                print(f"  可视化 → {out}")
            continue
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

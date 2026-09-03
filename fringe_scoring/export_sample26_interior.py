"""26 片批「扣除边缘应力斑带后的内部图」导出器（开发壳，不随核心包交付）。

用途（2026-08-05 用户核对，续裁切核对）：把每片矫正图再扣掉边缘应力斑带后的
内部区域导出为 PNG。该域即**位置评分（分布打分）的惩罚域**——免罚边框带机制
为位置评分设计（决策 #13：细圈边部应力物理正常故免罚、超允许宽度的段按缺陷留罚），
带宽 = min(实测带宽, border.normal_band_frac 允许宽度)，与 score.compute_pipeline
的 interior 逐位一致；深浅四项统计域复用同一 interior。实测带宽超允许宽度的片，
超出部分**留在图内**（按缺陷计罚的域，本就该看见），核对 PDF 里同时画实测带内缘
（橙虚线）供对照。

产出（不入库，仅供人工核对）：
- data/derived/sample26_thickness/warped_crops/<文件名>_去边框带.png——内部图；
- 同目录 去边框带核对.pdf——左=矫正图+免罚带内缘（绿）/实测带内缘（橙虚，仅当
  两者不同）；右=内部裁切图；标注四边带宽（免罚|实测，px）。
用法：venv python fringe_scoring/export_sample26_interior.py
"""

from __future__ import annotations

import sys
from pathlib import Path

GLASSAPP = Path(r"D:\GlassApp")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "sample26_thickness" / "warped_crops"

import numpy as np  # noqa: E402
import yaml  # noqa: E402

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _glassapp_api() -> dict:
    """载入 GlassApp 权威实现（防旧副本顶包断言，同 export_sample26_crops）。"""
    if str(GLASSAPP) not in sys.path:
        sys.path.insert(0, str(GLASSAPP))
    cached = sys.modules.get("fringe_scoring")
    if cached is not None and not Path(cached.__file__).resolve().is_relative_to(GLASSAPP):
        raise RuntimeError("fringe_scoring 已解析到本仓旧副本——请以脚本方式直接运行本文件")
    import fringe_scoring
    assert Path(fringe_scoring.__file__).resolve().is_relative_to(GLASSAPP)
    from app.config_store import load_app_config
    from fringe_scoring.score import compute_pipeline
    from fringe_scoring.sheets import detect_sheet_quads_with_meta
    return {"load_app_config": load_app_config, "compute_pipeline": compute_pipeline,
            "detect": detect_sheet_quads_with_meta}


def _load_gray(path: Path) -> np.ndarray:
    """照片 → float 灰度（np.fromfile+imdecode，中文路径安全）。"""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读图失败：{path}")
    return np.asarray(img, dtype=float)


def _save_png(path: Path, arr: np.ndarray) -> None:
    """float 灰度 → uint8 PNG（imencode+tofile，中文路径安全）。"""
    import cv2

    ok, buf = cv2.imencode(".png", np.clip(arr, 0, 255).astype(np.uint8))
    assert ok, f"PNG 编码失败：{path}"
    buf.tofile(str(path))


def _penalty_bands(bands, shape: tuple[int, int], normal_frac: float) -> dict:
    """免罚带宽 = min(实测带宽, 允许宽度)——与 score.compute_pipeline 同式复算。"""
    rows, cols = shape
    allow_r, allow_c = int(round(normal_frac * rows)), int(round(normal_frac * cols))
    return {"top": min(bands.top_px, allow_r), "bottom": min(bands.bottom_px, allow_r),
            "left": min(bands.left_px, allow_c), "right": min(bands.right_px, allow_c)}


def _rect(ax, top: int, bottom: int, left: int, right: int, shape, **kw) -> None:
    """在矫正图坐标系画内缘矩形（带宽向内偏移后的边界）。"""
    rows, cols = shape
    xs = [left - 0.5, cols - right - 0.5, cols - right - 0.5, left - 0.5, left - 0.5]
    ys = [top - 0.5, top - 0.5, rows - bottom - 0.5, rows - bottom - 0.5, top - 0.5]
    ax.plot(xs, ys, **kw)


def main() -> int:
    """逐张扣带导出 + 核对 PDF。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    api = _glassapp_api()
    cfg = api["load_app_config"]()
    normal_frac = float(cfg["border"]["normal_band_frac"])
    man = yaml.safe_load((ROOT / "data" / "derived" / "sample26_thickness" /
                          "manifest.yaml").read_text(encoding="utf-8"))
    photo_dir = ROOT / man["photo_dir"]
    OUT.mkdir(parents=True, exist_ok=True)

    rows_out, capped = [], []
    for e in man["photos"]:
        arr = _load_gray(photo_dir / e["file"])
        if e.get("manual_quad"):
            quad = np.asarray(e["manual_quad"], dtype=float)
        else:
            quads, _ = api["detect"](arr, cfg)
            assert len(quads) == 1, f"{e['file']}: 自动检出 {len(quads)} 片≠1，与报告口径不符"
            quad = quads[0]
        pipe = api["compute_pipeline"](arr, config=cfg, quad_corners=quad)
        h, w = pipe.arr.shape
        b = pipe.bands
        pb = _penalty_bands(b, pipe.arr.shape, normal_frac)
        inner = pipe.arr[pb["top"]: h - pb["bottom"], pb["left"]: w - pb["right"]]
        out_png = OUT / f"{Path(e['file']).stem}_去边框带.png"
        _save_png(out_png, inner)
        meas = {"top": b.top_px, "bottom": b.bottom_px, "left": b.left_px, "right": b.right_px}
        is_capped = any(meas[k] > pb[k] for k in pb)
        if is_capped:
            capped.append(e["file"])
        rows_out.append({"e": e, "warp": pipe.arr, "pb": pb, "meas": meas,
                         "inner": inner, "capped": is_capped})
        print(f"{e['file']:10s} {e['sample_id']:5s} 免罚带宽 上{pb['top']} 下{pb['bottom']} "
              f"左{pb['left']} 右{pb['right']}px"
              + (f"（实测 上{meas['top']} 下{meas['bottom']} 左{meas['left']} "
                 f"右{meas['right']}——超允许宽，超出部分留在图内计罚）" if is_capped else "")
              + f" → 内部 {inner.shape[1]}×{inner.shape[0]}px")

    pdf_out = OUT / "去边框带核对.pdf"
    with PdfPages(pdf_out) as pdf:
        for p0 in range(0, len(rows_out), 4):
            fig, axes = plt.subplots(4, 2, figsize=(8.27, 11.69), dpi=110)
            for k in range(4):
                ax_l, ax_r = axes[k]
                if p0 + k >= len(rows_out):
                    ax_l.set_axis_off(), ax_r.set_axis_off()
                    continue
                r = rows_out[p0 + k]
                e, pb, meas = r["e"], r["pb"], r["meas"]
                ax_l.imshow(r["warp"], cmap="gray", vmin=0, vmax=255)
                _rect(ax_l, pb["top"], pb["bottom"], pb["left"], pb["right"],
                      r["warp"].shape, color="#30c060", lw=1.2)
                if r["capped"]:  # 实测带内缘（虚线）——两框之间的环按缺陷计罚
                    _rect(ax_l, meas["top"], meas["bottom"], meas["left"], meas["right"],
                          r["warp"].shape, color="#e08020", lw=1.2, ls="--")
                ax_l.set_title(
                    f"{e['file']} · {e['sample_id']}　带宽(免罚|实测)px 上{pb['top']}|{meas['top']}"
                    f" 下{pb['bottom']}|{meas['bottom']} 左{pb['left']}|{meas['left']}"
                    f" 右{pb['right']}|{meas['right']}", fontsize=7)
                ax_r.imshow(r["inner"], cmap="gray", vmin=0, vmax=255)
                ax_r.set_title(f"扣免罚边框带内部 {r['inner'].shape[1]}×"
                               f"{r['inner'].shape[0]}px（位置评分惩罚域；深浅四项同域）",
                               fontsize=7.5)
                for ax in (ax_l, ax_r):
                    ax.set_xticks([]), ax.set_yticks([])
            fig.suptitle(
                f"26 片批去边框带核对（{p0 + 1}–{min(p0 + 4, len(rows_out))} / {len(rows_out)}）"
                "　左=矫正图+免罚带内缘（绿实线；橙虚线=实测带内缘，仅超允许宽的片有）"
                "　右=内部图", fontsize=8.5)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n内部图 {len(rows_out)} 张 + 核对 PDF → {OUT}")
    print(f"实测带宽超允许宽（{normal_frac:.0%}）的片 {len(capped)} 张："
          f"{'、'.join(capped) if capped else '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

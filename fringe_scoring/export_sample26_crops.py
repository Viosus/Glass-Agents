"""26 片批「实际被打分的裁切图」导出器（开发壳，不随核心包交付）。

用途（2026-08-05 用户核对裁切）：把报告（v1/v3/v4 同一打分口径）里每张照片
实际进入六指标计算的**透视矫正后单片图**导出为 PNG，供人工核对裁切正确性。
口径与 make_sample26_assets._score_photo 完全一致：manifest.manual_quad 优先，
否则自动检测（恰 1 片才裁）；矫正 = compute_pipeline 的 warp（含角点排序）。

产出（不入库，仅供人工核对）：
- data/derived/sample26_thickness/warped_crops/<文件名>_裁切.png——矫正后单片图；
- 同目录 裁切核对.pdf——每片一格：原图+检出框（红=自动/绿=人工） | 裁切图，
  含矫正尺寸与规格对照，7 页速览。
用法：venv python fringe_scoring/export_sample26_crops.py
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
    """载入 GlassApp 权威实现（防旧副本顶包断言，同 make_sample26_assets）。"""
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


def main() -> int:
    """逐张裁切导出 + 7 页核对 PDF。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    api = _glassapp_api()
    cfg = api["load_app_config"]()
    man = yaml.safe_load((ROOT / "data" / "derived" / "sample26_thickness" /
                          "manifest.yaml").read_text(encoding="utf-8"))
    photo_dir = ROOT / man["photo_dir"]
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for e in man["photos"]:
        arr = _load_gray(photo_dir / e["file"])
        if e.get("manual_quad"):
            quad, prov = np.asarray(e["manual_quad"], dtype=float), "manual"
        else:
            quads, _ = api["detect"](arr, cfg)
            assert len(quads) == 1, f"{e['file']}: 自动检出 {len(quads)} 片≠1，与报告口径不符"
            quad, prov = quads[0], "auto"
        pipe = api["compute_pipeline"](arr, config=cfg, quad_corners=quad)
        out_png = OUT / f"{Path(e['file']).stem}_裁切.png"
        _save_png(out_png, pipe.arr)
        rows.append({"e": e, "arr": arr, "quad": quad, "prov": prov, "warp": pipe.arr})
        wh, ww = pipe.arr.shape
        print(f"{e['file']:10s} {e['sample_id']:5s} 角点{prov:6s} → 裁切 {ww}×{wh}px "
              f"(规格 {e['spec_mm'][0]}×{e['spec_mm'][1]}mm) → {out_png.name}")

    pdf_out = OUT / "裁切核对.pdf"
    with PdfPages(pdf_out) as pdf:
        for p0 in range(0, len(rows), 4):
            fig, axes = plt.subplots(4, 2, figsize=(8.27, 11.69), dpi=110)
            for k in range(4):
                ax_l, ax_r = axes[k]
                if p0 + k >= len(rows):
                    ax_l.set_axis_off(), ax_r.set_axis_off()
                    continue
                r = rows[p0 + k]
                e, quad = r["e"], r["quad"]
                ax_l.imshow(r["arr"], cmap="gray", vmin=0, vmax=255)
                qq = np.vstack([quad, quad[:1]])
                ax_l.plot(qq[:, 0], qq[:, 1], "-",
                          color="#30c060" if r["prov"] == "manual" else "#e04040", lw=1.4)
                ax_l.set_title(f"{e['file']} · {e['sample_id']}（角点 {r['prov']}）",
                               fontsize=8)
                wh, ww = r["warp"].shape
                ax_r.imshow(r["warp"], cmap="gray", vmin=0, vmax=255)
                ax_r.set_title(f"裁切 {ww}×{wh}px ｜ 规格 {e['spec_mm'][0]}×"
                               f"{e['spec_mm'][1]}mm（1px/mm 口径）", fontsize=8)
                for ax in (ax_l, ax_r):
                    ax.set_xticks([]), ax.set_yticks([])
            fig.suptitle(f"26 片批裁切核对（{p0 + 1}–{min(p0 + 4, len(rows))} / "
                         f"{len(rows)}）　左=原图+检出框（红=自动/绿=人工）　右=实际被打分的矫正图",
                         fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"\n裁切图 {len(rows)} 张 + 核对 PDF → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""26 片批 GlassApp 纯自动检测普查资产生成器（开发壳，不随核心包交付）。

回答「不带人工定角、直接用 GlassApp 测，哪些照片测不出来、为什么」：
口径 = 桌面端真实调用路径（app.config_store.load_app_config + fringe_scoring.sheets
的 detect_sheet_quads_with_meta，纯自动；worker.py 对 ValueError 弹窗拒绝打分）。
manifest 的 manual_quad 在本普查中**不使用**，只作对照标注。

产出（入库）：data/derived/sample26_thickness/auto_census/
- auto_census.json：逐张检出片数 / ValueError 文案 / 阈值阶段统计（失败与错检张）；
- diag_<stem>.png：失败/错检/曾人工定角张的四联诊断图（原图/阈值前景/闭运算/角点对比）。
用法：venv python fringe_scoring/make_sample26_census_assets.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

GLASSAPP = Path(r"D:\GlassApp")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "sample26_thickness" / "auto_census"
DATE = "2026-08-05"

import numpy as np  # noqa: E402
import yaml  # noqa: E402

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _glassapp_api() -> dict:
    """载入 GlassApp 权威实现（防混用断言同 make_sample26_assets：本仓旧副本不得顶包）。"""
    if str(GLASSAPP) not in sys.path:
        sys.path.insert(0, str(GLASSAPP))
    cached = sys.modules.get("fringe_scoring")
    if cached is not None and not Path(cached.__file__).resolve().is_relative_to(GLASSAPP):
        raise RuntimeError("fringe_scoring 已解析到本仓旧副本——请以脚本方式直接运行本文件")
    import fringe_scoring
    assert Path(fringe_scoring.__file__).resolve().is_relative_to(GLASSAPP)
    from app.config_store import load_app_config
    from fringe_scoring.segment import robust_scale
    from fringe_scoring.sheets import detect_sheet_quads_with_meta
    return {"load_app_config": load_app_config, "robust_scale": robust_scale,
            "detect": detect_sheet_quads_with_meta}


def _load_gray(path: Path) -> np.ndarray:
    """照片 → float 灰度（imdecode 走 np.fromfile，中文路径安全）。"""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读图失败：{path}")
    return np.asarray(img, dtype=float)


def _stage_stats(arr: np.ndarray, sheets_cfg: dict, robust_scale) -> dict:
    """复算 detect_sheet_quads_with_meta 的阈值/连通域阶段统计（仅诊断披露，非第二实现）。"""
    import cv2

    rows, cols = arr.shape
    trim = max(1, int(round(float(sheets_cfg["edge_trim_frac"]) * min(rows, cols))))
    inner = arr[trim: rows - trim, trim: cols - trim]
    bg_level = float(np.percentile(inner, 100.0 * float(sheets_cfg["bg_quantile"])))
    bg_scale = float(robust_scale(inner[inner <= bg_level]))
    p_bright = float(np.percentile(inner, 100.0 * float(sheets_cfg["bright_ref_quantile"])))
    thr_z = float(sheets_cfg["fg_min_z"]) * bg_scale
    thr_rel = float(sheets_cfg["fg_min_rel"]) * (p_bright - bg_level)
    threshold = bg_level + max(thr_z, thr_rel)
    fg = inner > threshold
    close_px = max(3, int(round(float(sheets_cfg["close_frac"]) * min(rows, cols))))
    close_px += (close_px + 1) % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px))
    closed = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(sheets_cfg["min_area_frac"]) * inner.shape[0] * inner.shape[1]
    min_side = float(sheets_cfg["min_side_frac"]) * min(inner.shape)
    comps = []
    for c in contours:
        a = float(cv2.contourArea(c))
        (w, h) = cv2.minAreaRect(c)[1]
        comps.append({"area": round(a, 1), "min_side": round(min(w, h), 1),
                      "pass_area": a >= min_area, "pass_side": min(w, h) >= min_side})
    comps.sort(key=lambda d: -d["area"])
    return {"trim": trim, "bg_level": round(bg_level, 2), "bg_scale": round(bg_scale, 3),
            "p_bright": round(p_bright, 2), "thr_z_term": round(thr_z, 2),
            "thr_rel_term": round(thr_rel, 2), "threshold": round(threshold, 2),
            "fg_frac": round(float(fg.mean()), 4),
            "min_area": round(min_area, 1), "min_side": round(min_side, 1),
            "n_contours": len(comps), "components_top8": comps[:8],
            "_inner": inner, "_fg": fg, "_closed": closed}


def _diag_png(name: str, st: dict, quads, manual_quad) -> str:
    """四联诊断图：原图 / 阈值前景 / 闭运算 / 自动角点(红) vs 人工角点(绿虚)。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=110)
    titles = ["原图（裁边后）", f"阈值前景 thr={st['threshold']:.1f}",
              "闭运算后", "自动角点(红) vs 人工(绿虚)"]
    axes[0].imshow(st["_inner"], cmap="gray", vmin=0, vmax=255)
    axes[1].imshow(st["_fg"], cmap="gray")
    axes[2].imshow(st["_closed"], cmap="gray")
    axes[3].imshow(st["_inner"], cmap="gray", vmin=0, vmax=255)
    trim = st["trim"]
    for q in quads or []:
        qq = np.vstack([q, q[:1]]) - trim  # 回到裁边坐标系叠画
        axes[3].plot(qq[:, 0], qq[:, 1], "-", color="#e04040", lw=1.6)
    if manual_quad:
        m = np.asarray(manual_quad, dtype=float) - trim
        m = np.vstack([m, m[:1]])
        axes[3].plot(m[:, 0], m[:, 1], "--", color="#30c060", lw=1.6)
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(name, fontsize=11)
    fig.tight_layout()
    out = OUT / f"diag_{Path(name).stem}.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def main() -> int:
    """普查全批：纯自动检测 → 逐张结果 + 失败/错检四联诊断图 → 落盘。"""
    api = _glassapp_api()
    cfg = api["load_app_config"]()
    cfg_sha = hashlib.sha256(
        json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    man = yaml.safe_load((ROOT / "data" / "derived" / "sample26_thickness" /
                          "manifest.yaml").read_text(encoding="utf-8"))
    photo_dir = ROOT / man["photo_dir"]
    OUT.mkdir(parents=True, exist_ok=True)

    photos = []
    for entry in man["photos"]:
        name = entry["file"]
        arr = _load_gray(photo_dir / name)
        rec: dict = {"file": name, "sample_id": entry["sample_id"], "type": entry["type"],
                     "had_manual_quad": bool(entry.get("manual_quad"))}
        try:
            quads, metas = api["detect"](arr, cfg)
            rec["outcome"] = "detected"
            rec["n_sheets"] = len(quads)
            rec["quads"] = [np.asarray(q).round(1).tolist() for q in quads]
            rec["edge_completed"] = [bool(m.get("edge_completed")) for m in metas]
        except ValueError as e:
            quads = None
            rec["outcome"] = "error"
            rec["error"] = str(e)
        # 可测性判定：报错 / 片数≠1（本批每张恰一片）都算「纯自动测不出/错检」
        rec["auto_ok"] = rec["outcome"] == "detected" and rec.get("n_sheets") == 1
        if not rec["auto_ok"] or rec["had_manual_quad"]:
            st = _stage_stats(arr, cfg["sheets"], api["robust_scale"])
            rec["diag_png"] = _diag_png(name, st, quads, entry.get("manual_quad"))
            rec["stages"] = {k: v for k, v in st.items() if not k.startswith("_")}
        photos.append(rec)
        tag = (f"{rec['n_sheets']} 片" if rec["outcome"] == "detected"
               else f"报错：{rec['error'][:60]}")
        print(f"{name:10s} {entry['sample_id']:5s} → {tag}"
              f"{' [曾人工定角]' if rec['had_manual_quad'] else ''}")

    census = {
        "meta": {"date": DATE, "config_sha256_16": cfg_sha,
                 "script": "fringe_scoring/make_sample26_census_assets.py",
                 "口径": "GlassApp 桌面端同款纯自动路径（score_sheets 检测段，无 manual_quad）；"
                        "worker.py 对 ValueError 弹窗拒绝打分"},
        "n_photos": len(photos),
        "n_auto_ok": sum(1 for r in photos if r["auto_ok"]),
        "n_error": sum(1 for r in photos if r["outcome"] == "error"),
        "n_miscount": sum(1 for r in photos if r["outcome"] == "detected"
                          and r["n_sheets"] != 1),
        "photos": photos,
    }
    (OUT / "auto_census.json").write_text(
        json.dumps(census, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n合计：纯自动可测 {census['n_auto_ok']}/{census['n_photos']}，"
          f"报错 {census['n_error']}，错检片数 {census['n_miscount']} → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

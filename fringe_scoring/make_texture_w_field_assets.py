"""W_w 产线大样本验证资产生成器（开发壳，不随核心包交付）。

数据源：E 盘产线归档 stress_glass.png（暗场偏光线扫全幅，高 1728/3000 px、
宽 821~10280 px；共约 3.8 万张，覆盖 2025-10 ~ 2026-05 的 166 个日期）。**无人工标注**——本脚本验证的不是"与人评的相关性"
（那由 115 片标注集承担），而是 W_w 在真实产线规模上的：
1. 数学保证：W ∈ [0,1) 无一例外（两个理论上界 (Ng−1)² 与 (2(Ng−1))⁴/12 的实测确认）；
2. 稳健性：检出成功率、异常/退化计数、单片耗时；
3. 分布形态：W 的分位数与直方图——为未来分级划线提供真实分布依据；
4. 口径体检（跨月漂移）：按日期分层统计 W 与成像特征（饱和占比/动态范围/尺寸），
   识别"曝光/分辨率是否变更过"这类会污染标定的非仿射漂移；
5. 性质在真实图上的复核：仿射不变（×a+b 逐位同值）、确定性（重复计算一致）。

抽样：按"日期目录"分层等距抽样（覆盖全时间跨度，避免只取某几天），默认 500 张。

⚠️ 预处理口径（**原生分辨率**，2026-07-30 二次修正）：用户确认 **mm/px = 1**——
原图 1 像素即 1 毫米（高 1728/3000 px ↔ 玻璃跨幅 1.73/3.00 m，宽 821~10280 px ↔
长度 0.82~10.28 m，与建筑玻璃实际尺寸吻合）。因此**任何降采样都会破坏标准要求的
"GLCM 步距 d=1 ≡ 真实 1 mm"**：
- 早期长边归一（MAX_SIDE=2000）→ 每张等效 mm/px 都不同（213 种）；
- 其后定高归一（h=1000）→ 1 px 实为 3 mm（h=3000 组）/ 1.73 mm（h=1728 组）。
两者都不是标准口径。故默认 `--preprocess native`：**原图直接喂流水线，不缩放**，
此时 mm_per_px=1.0 走恒等映射正是正确行为。`height_norm` 仅保留作快速抽查/口径对照。
（注意：两个传感器高度并非 mm/px 不同，而是玻璃跨幅不同——都是 1 px/mm。）
处理链：（可选缩放）→ sheets 检出+透视矫正 → 主片喂 tools.metrics.texture_w。

用法：venv python fringe_scoring/make_texture_w_field_assets.py [--n 500]
      [--preprocess native|height_norm] [--target-h 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fringe_scoring.sheets import score_sheets  # noqa: E402
from tools import metrics  # noqa: E402

FIELD_ROOTS = (Path(r"E:\zhuogaoData"), Path(r"E:\0201-0228"))
CFG_PATH = ROOT / "config" / "fringe_scoring.yaml"
OUT = ROOT / "data" / "derived" / "texture_w_field"
TARGET_H = 1000       # 定高归一目标高度（仅 height_norm 对照口径用）
# 矫正后宽度上限（纯算力护栏）：原生口径下最长片可达 10280 px，取 10600 留余量——
# 取值必须大于实际最大宽度，否则截断会静默改变统计域（触发时在 values.json 记录）
MAX_WARP_W = 10600

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _load_cfg() -> dict:
    """读取本仓库生效配置（仅供 sheets 检出与矫正；W_w 本身不读任何配置常数）。"""
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _collect_by_date() -> dict[str, list[Path]]:
    """扫描产线归档 → {日期目录名: [stress_glass.png, ...]}（按日期分层的抽样底册）。"""
    by_date: dict[str, list[Path]] = {}
    for root in FIELD_ROOTS:
        if not root.exists():
            continue
        for date_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            shots = sorted(date_dir.glob("*/stress_glass.png"))
            if shots:
                by_date.setdefault(date_dir.name, []).extend(shots)
    return by_date


def _stratified_sample(by_date: dict[str, list[Path]], n: int) -> list[Path]:
    """按日期分层等距抽样：各日按配额均匀取（等距而非随机，保证覆盖当日全时段）。"""
    dates = sorted(by_date)
    if not dates:
        return []
    per = max(1, n // len(dates))
    picked: list[Path] = []
    for d in dates:
        shots = by_date[d]
        k = min(per, len(shots))
        idx = np.linspace(0, len(shots) - 1, k).round().astype(int)
        picked.extend(shots[i] for i in dict.fromkeys(idx.tolist()))
    if len(picked) > n:  # 等距下采样到目标数量，保持日期覆盖均衡
        keep = np.linspace(0, len(picked) - 1, n).round().astype(int)
        picked = [picked[i] for i in dict.fromkeys(keep.tolist())]
    return picked


def _load_gray(path: Path, preprocess: str,
               target_h: int) -> tuple[np.ndarray, tuple[int, int], float]:
    """产线原图 → 灰度 float。返回 (图, 原始 shape, 实际缩放因子)。

    - ``native``（默认、标准口径）：不缩放。mm/px=1 时原图 1 px 即 1 mm，
      GLCM 步距 d=1 直接对应真实 1 mm，无需也不得重采样；
    - ``height_norm``（对照口径）：按高度缩放到 target_h。**会破坏 1 mm 对应**
      （缩放后 1 px = 原高/target_h 毫米），仅用于快速抽查与口径对照。
    """
    raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"读图失败：{path}")
    orig = (int(raw.shape[0]), int(raw.shape[1]))
    if preprocess == "native":
        return np.asarray(raw, dtype=float), orig, 1.0
    h, w = raw.shape
    scale = target_h / float(h)
    new_w = max(2, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    raw = cv2.resize(raw, (new_w, target_h), interpolation=interp)
    return np.asarray(raw, dtype=float), orig, float(scale)


def _capture_meta(path: Path) -> dict:
    """由采集目录名解析成像参数（…_0mm_400us_f4_255 = 厚度/曝光/光圈/增益）。"""
    parts = path.parent.name.split("_")
    return {
        "dir": path.parent.name,
        "date": path.parent.parent.name,
        "exposure": parts[5] if len(parts) > 5 else "?",
        "aperture": parts[6] if len(parts) > 6 else "?",
        "gain": parts[7] if len(parts) > 7 else "?",
    }


def _quantiles(x: np.ndarray) -> dict:
    """分位数摘要（分级划线用的分布依据）。"""
    qs = (0, 1, 5, 25, 50, 75, 95, 99, 100)
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs}


def main(argv: list[str] | None = None) -> int:
    """分层抽样 → 批量跑 W_w → 落盘 values.json 并打印摘要。"""
    ap = argparse.ArgumentParser(description="W_w 产线大样本验证")
    ap.add_argument("--n", type=int, default=500, help="目标样本数（默认 500）")
    ap.add_argument("--preprocess", choices=("native", "height_norm"), default="native",
                    help="native=原图不缩放（标准口径，mm/px=1 时 d=1≡1mm）；"
                         "height_norm=定高归一（对照用，会破坏 1mm 对应）")
    ap.add_argument("--target-h", type=int, default=TARGET_H,
                    help="height_norm 的目标高度（native 下忽略）")
    args = ap.parse_args(argv)

    by_date = _collect_by_date()
    if not by_date:
        print(f"[跳过] 未找到产线归档：{[str(r) for r in FIELD_ROOTS]}")
        return 0
    total = sum(len(v) for v in by_date.values())
    picked = _stratified_sample(by_date, args.n)
    print(f"底册 {total} 张 / {len(by_date)} 个日期 → 分层抽样 {len(picked)} 张\n")

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()
    recs, failures = [], []
    t_start = time.time()
    for i, p in enumerate(picked, 1):
        meta = _capture_meta(p)
        t0 = time.time()
        try:
            arr, orig, scale = _load_gray(p, args.preprocess, args.target_h)
            inside_raw = arr[arr > 0]
            res = score_sheets(arr, cfg)
            warped = np.asarray(res.sheet_results[0].warped_image, dtype=float)
            truncated = bool(warped.shape[1] > MAX_WARP_W)
            if truncated:  # 算力护栏（逐片记录，不静默）
                warped = warped[:, :MAX_WARP_W]
            r = metrics.texture_w(warped, mm_per_px=1.0)
            recs.append({
                **meta, "orig_h": orig[0], "orig_w": orig[1], "scale": round(scale, 4),
                "sensor_group": int(orig[0]), "n_sheets": int(res.n_sheets),
                "warp_h": int(warped.shape[0]), "warp_w": int(warped.shape[1]),
                "texture_w": round(float(r.value), 5),
                "ca_w": round(float(r.ca_w), 4),
                "cpa_w": round(float(r.cpa_w), 3),
                "dyn_range": round(float(r.dynamic_range_nm), 1),
                "degenerate": bool(r.degenerate), "truncated": truncated,
                # 成像特征（口径体检用）
                "sat_frac": round(float(np.mean(arr >= 250)), 6),
                "median_gray": round(float(np.median(inside_raw)) if inside_raw.size else 0.0, 2),
                "sec": round(time.time() - t0, 2),
            })
        except Exception as e:  # 检出失败/异常一律记录，不静默跳过
            failures.append({**meta, "error": f"{type(e).__name__}: {e}"})
        if i % 50 == 0 or i == len(picked):
            ok = len(recs)
            print(f"  [{i}/{len(picked)}] 成功 {ok} 失败 {len(failures)} "
                  f"累计 {time.time() - t_start:.0f}s")

    if not recs:
        print("[跳过] 无成功样本")
        return 0

    w = np.array([r["texture_w"] for r in recs])
    assert float(w.max()) < 1.0, f"理论上界失守：W_max={w.max()}"

    # 按月分层统计（口径体检：跨月漂移）。**按月不按日**——日粒度每桶仅 ~3 张，
    # 极值桶由 n=1 的单片主导，会把抽样噪声误读成设备漂移（2026-07-30 修正）
    by_m: dict[str, list[float]] = {}
    for r in recs:
        by_m.setdefault(r["date"][:7], []).append(r["texture_w"])
    month_stats = [{"month": m, "n": len(v), "w_mean": round(float(np.mean(v)), 4),
                    "w_med": round(float(np.median(v)), 4)}
                   for m, v in sorted(by_m.items()) if len(v) >= 5]
    m_meds = [x["w_med"] for x in month_stats]

    # 传感器分组统计（线扫两种配置：高 3000 / 1728，物理跨幅可能不同）
    by_g: dict[int, list[float]] = {}
    for r in recs:
        by_g.setdefault(r["sensor_group"], []).append(r["texture_w"])
    group_stats = [{"sensor_h": g, "n": len(v), "w_mean": round(float(np.mean(v)), 4),
                    "w_med": round(float(np.median(v)), 4)} for g, v in sorted(by_g.items())]

    # 成像口径一致性（非仿射漂移探针）
    sizes = {f"{r['orig_h']}x{r['orig_w']}" for r in recs}
    expo = {r["exposure"] for r in recs}
    sat = np.array([r["sat_frac"] for r in recs])

    # 尺度污染探针：定高归一后 W 是否仍随缩放因子/幅面系统性变化（应≈0）
    def _sp(x, y) -> float | None:
        """Spearman 秩相关（argsort 名次法）；任一侧为常量 → None（相关性无定义）。

        原生口径下 scale 恒为 1.0，此时返回 None 而非 NaN——避免把"无定义"
        写成一个看似有值的数字。
        """
        ax, ay = np.asarray(x, float), np.asarray(y, float)
        if ax.std() == 0 or ay.std() == 0:
            return None
        rx = np.argsort(np.argsort(ax)).astype(float)
        ry = np.argsort(np.argsort(ay)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    def _sp3(x, y) -> float | None:
        """_sp 的三位小数版（None 透传）。"""
        v = _sp(x, y)
        return None if v is None else round(v, 3)

    scale_arr = np.array([r["scale"] for r in recs])
    warp_area = np.array([r["warp_h"] * r["warp_w"] for r in recs])
    contamination = {
        "spearman_w_vs_scale": _sp3(w, scale_arr),   # native 下 scale 恒 1.0 → None
        "spearman_w_vs_warp_area": _sp3(w, warp_area),
        # ⚠️ 结构性混淆声明（2026-07-30 对抗复核）：定高归一后 scale 只有两个取值、
        # 与传感器组一一映射，而两个传感器组的样本在时间上完全分离 → 该探针无法区分
        # 尺度效应/传感器效应/月份效应，不能据此声称"无尺度污染"
        "confounded": True,
        "scale_unique_values": sorted({round(s, 4) for s in scale_arr.tolist()}),
        "sensor_group_by_month": {
            str(g): sorted({r["date"][:7] for r in recs if r["sensor_group"] == g})
            for g in sorted({r["sensor_group"] for r in recs})
        },
    }

    # 跨月趋势的显著性检验（校验指出：不能把显著趋势写成"真实工艺波动"）
    day_ord = np.array([
        int(r["date"][:4]) * 372 + int(r["date"][5:7]) * 31 + int(r["date"][8:10])
        for r in recs], dtype=float)
    rho_t = _sp(w, day_ord)
    n_t = len(w)
    t_stat = (rho_t * np.sqrt((n_t - 2) / max(1e-12, 1 - rho_t ** 2))
              if rho_t is not None else float("nan"))
    trend = {
        "spearman_w_vs_date": round(rho_t, 3) if rho_t is not None else None, "n": n_t,
        "t_stat": round(float(t_stat), 2),
        "significant_at_0p01": bool(abs(t_stat) > 2.58),
        "note": "本批无质量标签、无月度重复扫描参考样 → 无法区分工艺波动与缓慢口径漂移；"
                "后者对固定分级线风险更大，须在标定样本集上专项排除",
    }

    # 失败细分（校验指出：不能笼统归因"透视矫正"）
    fail_kinds: dict[str, int] = {}
    for f in failures:
        key = f["error"].split(":")[1].strip().split()[0] if ":" in f["error"] else "unknown"
        fail_kinds[key] = fail_kinds.get(key, 0) + 1

    # 选片口径披露（校验指出：474 片实为每图取阅读序第一片）
    nsheets = np.array([r["n_sheets"] for r in recs])
    warp_h_arr = np.array([r["warp_h"] for r in recs])
    selection = {
        "rule": "每张图取阅读序第一片（主片），非整图全部片",
        "n_images_with_multiple_sheets": int((nsheets >= 2).sum()),
        "max_sheets_in_one_image": int(nsheets.max()),
        "n_warp_height_below_200px": int((warp_h_arr < 200).sum()),
        "min_warp_shape": [int(warp_h_arr.min()),
                           int(min(r["warp_w"] for r in recs))],
    }

    # 真实图上的性质复核（取前 5 张成功样本）
    prop = {"affine_invariant": True, "deterministic": True, "checked": 0}
    for p in picked[:40]:
        if prop["checked"] >= 5:
            break
        try:
            arr, _, _ = _load_gray(p, args.preprocess, args.target_h)
            wp = np.asarray(score_sheets(arr, cfg).sheet_results[0].warped_image, dtype=float)
            wp = wp[:, :MAX_WARP_W]
        except Exception:
            continue
        v0 = metrics.texture_w(wp, mm_per_px=1.0).value
        v_aff = metrics.texture_w(wp * 1.7 + 13.0, mm_per_px=1.0).value
        v_rep = metrics.texture_w(wp, mm_per_px=1.0).value
        if abs(v_aff - v0) > 1e-9 * max(v0, 1e-9):
            prop["affine_invariant"] = False
        if v_rep.hex() != v0.hex():
            prop["deterministic"] = False
        prop["checked"] += 1

    values = {
        "source": [str(r) for r in FIELD_ROOTS], "n_pool": total,
        "n_dates": len(by_date), "n_sampled": len(picked),
        "n_scored": len(recs), "n_failed": len(failures),
        "success_rate": round(100.0 * len(recs) / max(len(picked), 1), 1),
        "w_quantiles": _quantiles(w),
        "w_mean": round(float(w.mean()), 4), "w_std": round(float(w.std()), 4),
        "w_max_lt_1": bool(w.max() < 1.0),
        "n_degenerate": int(sum(r["degenerate"] for r in recs)),
        "month_stats": month_stats,
        "month_drift": {"w_med_min": round(min(m_meds), 4),
                        "w_med_max": round(max(m_meds), 4),
                        "spread": round(max(m_meds) - min(m_meds), 4)},
        "sensor_group_stats": group_stats,
        "scale_contamination": contamination,
        "date_trend": trend,
        "failure_kinds": fail_kinds,
        "sheet_selection": selection,
        "capture_consistency": {
            "orig_sizes": sorted(sizes), "n_sizes": len(sizes),
            "exposures": sorted(expo),
            "sat_frac_max": round(float(sat.max()), 6),
            "sat_frac_mean": round(float(sat.mean()), 6),
        },
        "properties_on_real_images": prop,
        "sec_per_image": round(float(np.mean([r["sec"] for r in recs])), 2),
        "failures": failures[:50],
        "all": recs,
        "preprocess": args.preprocess,
        "target_h": args.target_h if args.preprocess == "height_norm" else None,
        "n_truncated": int(sum(r["truncated"] for r in recs)),
        "_口径说明": (
            "暗场产线原图（线扫）。"
            + ("预处理=native：不缩放；用户确认 mm/px=1，故原图 1px 即 1mm，"
               "GLCM 步距 d=1 直接对应真实 1mm，mm_per_px=1.0 恒等映射即标准口径。"
               if args.preprocess == "native" else
               f"预处理=height_norm（对照口径）：定高至 {args.target_h}px，"
               "会破坏 1mm 对应，不可作标准结论。")
            + "无人工标注——本批验证数学保证/稳健性/分布形态，"
              "与人评的相关性由 115 片标注集承担"),
    }
    (OUT / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")

    q = values["w_quantiles"]
    print(f"\n预处理口径: {args.preprocess}"
          + (f"（原图不缩放，d=1≡1mm 标准口径）" if args.preprocess == "native"
             else f"（定高 {args.target_h}px，对照口径）")
          + f"　超宽截断 {values['n_truncated']} 片")
    print(f"成功 {len(recs)}/{len(picked)}（{values['success_rate']}%）失败 {len(failures)}")
    print(f"W 分布: min={q['p0']} p5={q['p5']} p25={q['p25']} 中位={q['p50']} "
          f"p75={q['p75']} p95={q['p95']} max={q['p100']}")
    print(f"  均值 {values['w_mean']} ± {values['w_std']}；**全体 < 1: {values['w_max_lt_1']}** ✓")
    md = values["month_drift"]
    print(f"跨月漂移（n≥5 的月，取中位）: {md['w_med_min']} ~ {md['w_med_max']}"
          f"（跨度 {md['spread']}，共 {len(month_stats)} 个月）")
    print(f"传感器分组: " + "; ".join(
        f"h={g['sensor_h']} n={g['n']} 中位={g['w_med']}" for g in group_stats))
    _sc = contamination["spearman_w_vs_scale"]
    print(f"尺度探针: W~缩放 {'无定义(native 下 scale 恒 1)' if _sc is None else f'{_sc:+.3f}'}  "
          f"W~矫正面积 {contamination['spearman_w_vs_warp_area']:+.3f}"
          f"  ⚠️传感器组与月份共线={contamination['confounded']}")
    print(f"日期趋势: ρ={trend['spearman_w_vs_date']:+.3f} t={trend['t_stat']} "
          f"0.01 显著={trend['significant_at_0p01']}（工艺波动 vs 口径漂移无法区分）")
    print(f"失败细分: {fail_kinds}")
    print(f"选片口径: {selection['rule']}；多片图 {selection['n_images_with_multiple_sheets']} 张"
          f"（最多 {selection['max_sheets_in_one_image']} 片）；"
          f"矫正高<200px {selection['n_warp_height_below_200px']} 片")
    cc = values["capture_consistency"]
    print(f"成像一致性: 原图高 {sorted({r['orig_h'] for r in recs})}（线扫，宽随长度变）；"
          f"曝光 {cc['exposures']}；饱和占比 max={cc['sat_frac_max']:.4%}")
    print(f"真实图性质: 仿射不变={prop['affine_invariant']} 确定性={prop['deterministic']} "
          f"（复核 {prop['checked']} 张）")
    print(f"平均耗时 {values['sec_per_image']}s/张")
    print(f"资产 → {OUT / 'values.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

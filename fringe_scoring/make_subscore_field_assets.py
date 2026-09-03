"""产线四项深浅类指标分布普查资产生成器（开发壳，不随核心包交付）。

目的（2026-08-05 用户拍板任务 1）：四项子分（x095/灰度方差/梯度均值/梯度方差）
的 refs 重定标需要产线分布，而 texture_w_field 资产只算了 W_w——本脚本在同一份
480 张已打分样本清单（data/derived/texture_w_field/values.json 的 all[]）上补算
四项原始值，产出分位表与「候选 refs × 映射机制」的 0 分率/100 分率模拟。

口径：
- 图像与主片选择与 field 资产同源：E 盘原图 → 检出 → **阅读序第一片**
  （sheet_results[0] 同位）；本脚本用 GlassApp 权威实现 + app_config.yaml
  （产品口径；close_frac 取当前 config 值，运行时间点应在 close_frac 拍板落地后）。
- 指标 = compute_sheet_indicators 权威实现的 raw_values（含 W_w 顺带，供与
  field 资产交叉核对），不做第二套公式。
- 模拟：线性 = indicators._sub_score 同式；对数 = 26 片批 v2 已验证的
  make_sample26_rescore_pdf._log_sub 同式（本脚本内复制并注明出处）。

产出：data/derived/subscore_field/values.json
用法：venv python fringe_scoring/make_subscore_field_assets.py [--limit N]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

GLASSAPP = Path(r"D:\GlassApp")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "subscore_field"
FIELD_ROOTS = (Path(r"E:\zhuogaoData"), Path(r"E:\0201-0228"))
DATE = "2026-08-05"
FOUR = ("x095", "gray_variance", "gradient_mean", "gradient_variance")
QS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)
# 候选 worst 网格（26 片批候选 + 上下各留一档；best 沿用现值——26 片验证口径）
CAND_BEST = {"x095": 20.0, "gray_variance": 50.0, "gradient_mean": 5.0,
             "gradient_variance": 30.0}
CAND_WORST = {"x095": (210.0, 250.0), "gray_variance": (4000.0, 6000.0, 8000.0),
              "gradient_mean": (32.0, 64.0, 80.0),
              "gradient_variance": (1100.0, 8000.0, 12000.0)}

import numpy as np  # noqa: E402

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _glassapp_api() -> dict:
    """载入 GlassApp 权威实现（防旧副本顶包断言）。"""
    if str(GLASSAPP) not in sys.path:
        sys.path.insert(0, str(GLASSAPP))
    cached = sys.modules.get("fringe_scoring")
    if cached is not None and not Path(cached.__file__).resolve().is_relative_to(GLASSAPP):
        raise RuntimeError("fringe_scoring 已解析到本仓旧副本——请以脚本方式直接运行本文件")
    import fringe_scoring
    assert Path(fringe_scoring.__file__).resolve().is_relative_to(GLASSAPP)
    from app.config_store import load_app_config
    from fringe_scoring.indicators import compute_sheet_indicators
    from fringe_scoring.score import compute_pipeline, score_from_pipeline
    from fringe_scoring.sheets import detect_sheet_quads_with_meta
    return {"load_app_config": load_app_config,
            "compute_sheet_indicators": compute_sheet_indicators,
            "compute_pipeline": compute_pipeline,
            "score_from_pipeline": score_from_pipeline,
            "detect": detect_sheet_quads_with_meta}


def _load_gray(path: Path) -> np.ndarray:
    """照片 → float 灰度（np.fromfile+imdecode，中文路径安全）。"""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读图失败：{path}")
    return np.asarray(img, dtype=float)


def _field_path(date: str, dir_name: str) -> Path | None:
    """产线记录 (date, dir) → stress_glass.png 路径（两根都试）。"""
    for root in FIELD_ROOTS:
        p = root / date / dir_name / "stress_glass.png"
        if p.exists():
            return p
    return None


def _log_sub(value: float, best: float, worst: float) -> float:
    """对数域子分（与 make_sample26_rescore_pdf._log_sub 同式，26 片批已验证）。"""
    if value <= 0.0:
        return 100.0
    s = 100.0 * (math.log(worst) - math.log(value)) / (math.log(worst) - math.log(best))
    return float(np.clip(s, 0.0, 100.0))


def _lin_sub(value: float, best: float, worst: float) -> float:
    """线性子分（indicators._sub_score 非溢出键同式）。"""
    s = 100.0 * (worst - value) / (worst - best)
    return float(np.clip(s, 0.0, 100.0))


def _simulate(vals: np.ndarray, best: float, worst: float, mech: str) -> dict:
    """一组原始值 × 一套 refs × 机制 → 0 分率 / 100 分率 / 子分五数概括。"""
    f = _log_sub if mech == "log" else _lin_sub
    subs = np.array([f(v, best, worst) for v in vals])
    return {"mech": mech, "best": best, "worst": worst,
            "zero_rate": round(float((subs == 0.0).mean()), 4),
            "full_rate": round(float((subs == 100.0).mean()), 4),
            "sub_p5": round(float(np.quantile(subs, 0.05)), 1),
            "sub_p50": round(float(np.quantile(subs, 0.50)), 1),
            "sub_p95": round(float(np.quantile(subs, 0.95)), 1)}


def main() -> int:
    """480 样本四项普查 → 分位表 + 候选模拟 → 落盘。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（调试用；0=全量）")
    args = ap.parse_args()

    api = _glassapp_api()
    cfg = api["load_app_config"]()
    close_frac = float(cfg["sheets"]["close_frac"])
    field = json.loads((ROOT / "data" / "derived" / "texture_w_field" /
                        "values.json").read_text(encoding="utf-8"))
    entries = field["all"][: args.limit] if args.limit else field["all"]
    OUT.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    t0 = time.time()
    for i, s in enumerate(entries, 1):
        p = _field_path(s["date"], s["dir"])
        if p is None:
            failures.append({"date": s["date"], "dir": s["dir"], "error": "missing_on_disk"})
            continue
        try:
            arr = _load_gray(p)
            quads, _ = api["detect"](arr, cfg)
            pipe = api["compute_pipeline"](arr, config=cfg, quad_corners=quads[0])
            r = api["score_from_pipeline"](pipe, cfg)
            cal = copy.deepcopy(cfg)
            cal["indicators"]["calibration"]["mm_per_px"] = 1.0  # 线扫原生口径
            ind = api["compute_sheet_indicators"](pipe.arr, ~r.border_mask_arr, cal,
                                                  pipeline=pipe)
            rows.append({
                "date": s["date"], "dir": s["dir"], "n_sheets": len(quads),
                "warp_h": int(pipe.arr.shape[0]), "warp_w": int(pipe.arr.shape[1]),
                "x095": round(ind.x095, 2),
                "gray_variance": round(ind.gray_variance, 2),
                "gradient_mean": round(ind.gradient_mean, 3),
                "gradient_variance": round(ind.gradient_variance, 2),
                "texture_w": round(ind.texture_w, 5),
                "texture_w_field_ref": s.get("texture_w"),  # field 资产旧值（交叉核对）
            })
        except Exception as e:  # 检出失败等一律记录，不静默跳过
            failures.append({"date": s["date"], "dir": s["dir"],
                             "error": f"{type(e).__name__}: {str(e)[:90]}"})
        if i % 60 == 0 or i == len(entries):
            print(f"  [{i}/{len(entries)}] 成功 {len(rows)} 失败 {len(failures)} "
                  f"累计 {time.time() - t0:.0f}s", flush=True)

    if not rows:
        print("无成功样本，终止（E 盘是否在线？）")
        return 1

    # 分位表 + 候选模拟 + gradient_mean 机制判据（p99/p5 跨度）
    quantiles, sims, spans = {}, {}, {}
    for key in FOUR:
        vals = np.array([r[key] for r in rows], dtype=float)
        quantiles[key] = {f"p{int(q * 100)}": round(float(np.quantile(vals, q)), 3)
                          for q in QS}
        spans[key] = round(float(np.quantile(vals, 0.99) / max(np.quantile(vals, 0.05),
                                                               1e-9)), 2)
        sims[key] = []
        for worst in CAND_WORST[key]:
            for mech in (("lin",) if key == "x095" else ("lin", "log")):
                sims[key].append(_simulate(vals, CAND_BEST[key], worst, mech))

    # W_w 交叉核对（同片同参时应几乎逐位一致；close_frac 或截断差异会体现在此）
    diffs = [abs(r["texture_w"] - r["texture_w_field_ref"]) for r in rows
             if r.get("texture_w_field_ref") is not None]
    cross = {"n": len(diffs), "max_abs_diff": round(max(diffs), 6) if diffs else None,
             "median_abs_diff": round(float(np.median(diffs)), 6) if diffs else None}

    values = {
        "meta": {"date": DATE, "script": "fringe_scoring/make_subscore_field_assets.py",
                 "close_frac": close_frac,
                 "主片": "阅读序第一片（与 texture_w_field 同位）",
                 "评估域": "扣免罚边框带内部（x095/方差/梯度四项口径）",
                 "候选best沿用现值": CAND_BEST},
        "n_listed": len(entries), "n_scored": len(rows), "n_failed": len(failures),
        "quantiles": quantiles, "p99_over_p5": spans, "simulations": sims,
        "texture_w_cross_check": cross,
        "failures": failures, "all": rows,
    }
    (OUT / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nclose_frac={close_frac}  成功 {len(rows)}/{len(entries)}")
    for key in FOUR:
        q = quantiles[key]
        print(f"{key:18s} p5={q['p5']:>9} p50={q['p50']:>9} p95={q['p95']:>9} "
              f"p99={q['p99']:>9} max={q['p100']:>9}  p99/p5={spans[key]}")
    print(f"W_w 交叉核对：中位差 {cross['median_abs_diff']}，最大差 {cross['max_abs_diff']}")
    print(f"资产 → {OUT / 'values.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

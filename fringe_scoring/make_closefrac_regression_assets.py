"""close_frac 0.005→0.01 回归资产生成器（开发壳，不随核心包交付）。

背景（2026-08-05 用户拍板「回归通过即落地」）：16#（Low-E 中空）型失败机理 =
边缘亮圈碎段在 close_frac=0.005（3px 核）下连不成闭环 → 弧段凸包四边形拟合退化。
0.01 已在 26 片批实验救回（IoU 0.97）且 8# 未钢化保持正确拒测；本脚本在三个
数据集上做 0.005 vs 0.01 全量双跑，产出采纳判定所需的全部证据。

三数据集与采纳标准（不达标不落地）：
A. 产线 14 张失败照（texture_w_field values.json failures[]）：
   7 张 quad_from_hull 同签名救回 ≥5（目标全部）；其余不恶化；无新增失败。
B. 产线 480 张已打分样本（all[]）：逐张 n_sheets 零变化（366 张多片防桥接）；
   全部片角点 L∞ 漂移 ≤2px（超限出四联诊断图人工裁决，目标 0 张）。
   角点相同 ⇒ 矫正图相同 ⇒ 下游指标逐位相同（确定性管线，无需重打分）。
C. 26 片批 28 张：16-1/16-2 救回且与人工角点 IoU ≥0.97；8#-1/8#-2 保持拒测
   （不得检出假片）；其余张角点零漂移。

产出：data/derived/closefrac_regression/values.json + 漂移/救回张四联诊断图。
用法：venv python fringe_scoring/make_closefrac_regression_assets.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

GLASSAPP = Path(r"D:\GlassApp")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "closefrac_regression"
FIELD_ROOTS = (Path(r"E:\zhuogaoData"), Path(r"E:\0201-0228"))
DATE = "2026-08-05"
CLOSE_A, CLOSE_B = 0.005, 0.01   # 现行 vs 候选
DRIFT_TOL_PX = 2.0               # 角点 L∞ 漂移容差

import numpy as np  # noqa: E402
import yaml  # noqa: E402

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _glassapp_api() -> dict:
    """载入 GlassApp 权威实现（防旧副本顶包断言，同 make_sample26_census_assets）。"""
    if str(GLASSAPP) not in sys.path:
        sys.path.insert(0, str(GLASSAPP))
    cached = sys.modules.get("fringe_scoring")
    if cached is not None and not Path(cached.__file__).resolve().is_relative_to(GLASSAPP):
        raise RuntimeError("fringe_scoring 已解析到本仓旧副本——请以脚本方式直接运行本文件")
    import fringe_scoring
    assert Path(fringe_scoring.__file__).resolve().is_relative_to(GLASSAPP)
    from app.config_store import load_app_config
    from fringe_scoring.sheets import detect_sheet_quads_with_meta
    return {"load_app_config": load_app_config, "detect": detect_sheet_quads_with_meta}


def _load_gray(path: Path) -> np.ndarray:
    """照片 → float 灰度（np.fromfile+imdecode，中文路径安全）。"""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读图失败：{path}")
    return np.asarray(img, dtype=float)


def _field_path(date: str, dir_name: str) -> Path | None:
    """产线记录 (date, dir) → E 盘 stress_glass.png 路径（两根都试，缺盘/缺图返回 None）。"""
    for root in FIELD_ROOTS:
        p = root / date / dir_name / "stress_glass.png"
        if p.exists():
            return p
    return None


def _detect(api: dict, arr: np.ndarray, cfg: dict, close_frac: float) -> dict:
    """一次检测 → {ok, n, quads|error}。quads 为 (n,4,2) 列表（原图坐标）。"""
    c = copy.deepcopy(cfg)
    c["sheets"]["close_frac"] = close_frac
    try:
        quads, _metas = api["detect"](arr, c)
        return {"ok": True, "n": len(quads),
                "quads": [np.asarray(q, dtype=float) for q in quads]}
    except ValueError as e:
        return {"ok": False, "n": 0, "error": str(e)}


def _quad_drift(qa: list, qb: list) -> float:
    """两组片（同数）按阅读序逐片对齐后的全部角点 L∞ 漂移最大值（px）。"""
    worst = 0.0
    for a, b in zip(qa, qb):
        worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def _quad_iou(q1: np.ndarray, q2) -> float:
    """两凸四边形 IoU（cv2.intersectConvexConvex；退化按 0 计）。"""
    import cv2

    a = np.asarray(q1, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(q2, dtype=np.float32).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    ua = float(cv2.contourArea(a.reshape(-1, 1, 2)))
    ub = float(cv2.contourArea(b.reshape(-1, 1, 2)))
    union = ua + ub - inter
    return float(inter / union) if union > 0 else 0.0


def _diag_png(tag: str, arr: np.ndarray, ra: dict, rb: dict) -> str:
    """双联诊断图：0.005 检出（红）vs 0.01 检出（绿）叠画原图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), dpi=100)
    for ax, r, color, name in ((axes[0], ra, "#e04040", f"close={CLOSE_A}"),
                               (axes[1], rb, "#30c060", f"close={CLOSE_B}")):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        for q in r.get("quads", []):
            qq = np.vstack([q, q[:1]])
            ax.plot(qq[:, 0], qq[:, 1], "-", color=color, lw=1.4)
        n_or_err = f"{r['n']} 片" if r["ok"] else "报错"
        ax.set_title(f"{name} → {n_or_err}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(tag, fontsize=10)
    fig.tight_layout()
    out = OUT / f"diag_{tag.replace('/', '_').replace(chr(92), '_')}.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def _run_pair(api: dict, cfg: dict, arr: np.ndarray) -> tuple[dict, dict]:
    """同一张图两参数各跑一次。"""
    return _detect(api, arr, cfg, CLOSE_A), _detect(api, arr, cfg, CLOSE_B)


def run_failures(api: dict, cfg: dict, field: dict) -> dict:
    """数据集 A：14 张产线失败照双跑。"""
    rows, n_rescued_quad, n_new_fail = [], 0, 0
    for f in field["failures"]:
        p = _field_path(f["date"], f["dir"])
        row = {"date": f["date"], "dir": f["dir"], "old_error": f["error"][:80]}
        if p is None:
            row["status"] = "missing_on_disk"
            rows.append(row)
            continue
        ra, rb = _run_pair(api, cfg, _load_gray(p))
        was_quad_fail = "quad_from_hull" in f["error"]
        row.update({
            "close_a": ra["n"] if ra["ok"] else f"err:{ra['error'][:60]}",
            "close_b": rb["n"] if rb["ok"] else f"err:{rb['error'][:60]}",
            "signature": "quad_from_hull" if was_quad_fail else "no_foreground",
            "rescued": (not ra["ok"]) and rb["ok"],
        })
        if row["rescued"] and was_quad_fail:
            n_rescued_quad += 1
        if ra["ok"] and not rb["ok"]:
            n_new_fail += 1
        row["diag_png"] = _diag_png(f"A_{f['date']}_{f['dir'][:15]}",
                                    _load_gray(p), ra, rb)
        rows.append(row)
    return {"rows": rows, "n_rescued_quad_from_hull": n_rescued_quad,
            "n_new_failures": n_new_fail}


def run_samples(api: dict, cfg: dict, field: dict) -> dict:
    """数据集 B：480 张已打分样本双跑（片数一致性 + 角点漂移）。"""
    n_checked = n_missing = n_count_diff = n_drift = 0
    multi_checked = 0
    worst_drift = 0.0
    bad_rows = []
    t0 = time.time()
    for i, s in enumerate(field["all"]):
        p = _field_path(s["date"], s["dir"])
        if p is None:
            n_missing += 1
            continue
        arr = _load_gray(p)
        ra, rb = _run_pair(api, cfg, arr)
        n_checked += 1
        if s.get("n_sheets", 1) > 1:
            multi_checked += 1
        row = {"date": s["date"], "dir": s["dir"],
               "a": ra["n"] if ra["ok"] else "err", "b": rb["n"] if rb["ok"] else "err"}
        if (ra["ok"], ra["n"]) != (rb["ok"], rb["n"]):
            n_count_diff += 1
            row["diag_png"] = _diag_png(f"B_{s['date']}_{s['dir'][:15]}", arr, ra, rb)
            bad_rows.append(row)
        elif ra["ok"]:
            drift = _quad_drift(ra["quads"], rb["quads"])
            worst_drift = max(worst_drift, drift)
            if drift > DRIFT_TOL_PX:
                n_drift += 1
                row["drift_px"] = round(drift, 2)
                row["diag_png"] = _diag_png(f"B_{s['date']}_{s['dir'][:15]}", arr, ra, rb)
                bad_rows.append(row)
        if (i + 1) % 60 == 0:
            print(f"  B 进度 {i + 1}/{len(field['all'])}"
                  f"（{(time.time() - t0) / (i + 1):.2f}s/张）", flush=True)
    return {"n_listed": len(field["all"]), "n_checked": n_checked, "n_missing": n_missing,
            "n_multi_checked": multi_checked, "n_count_diff": n_count_diff,
            "n_drift_gt_tol": n_drift, "worst_drift_px": round(worst_drift, 3),
            "drift_tol_px": DRIFT_TOL_PX, "bad_rows": bad_rows}


def run_sample26(api: dict, cfg: dict) -> dict:
    """数据集 C：26 片批 28 张双跑（16# 救回 IoU、8# 拒测语义、其余零漂移）。"""
    man = yaml.safe_load((ROOT / "data" / "derived" / "sample26_thickness" /
                          "manifest.yaml").read_text(encoding="utf-8"))
    photo_dir = ROOT / man["photo_dir"]
    rows = []
    for e in man["photos"]:
        arr = _load_gray(photo_dir / e["file"])
        ra, rb = _run_pair(api, cfg, arr)
        row = {"file": e["file"], "sample_id": e["sample_id"],
               "a": ra["n"] if ra["ok"] else f"err:{ra['error'][:40]}",
               "b": rb["n"] if rb["ok"] else f"err:{rb['error'][:40]}"}
        if e["sample_id"] == "16#" and rb["ok"] and rb["n"] == 1 and e.get("manual_quad"):
            row["iou_vs_manual"] = round(_quad_iou(rb["quads"][0], e["manual_quad"]), 4)
        if ra["ok"] and rb["ok"] and ra["n"] == rb["n"]:
            row["drift_px"] = round(_quad_drift(ra["quads"], rb["quads"]), 2)
        if (not ra["ok"]) != (not rb["ok"]) or ra.get("n") != rb.get("n"):
            row["diag_png"] = _diag_png(f"C_{Path(e['file']).stem}", arr, ra, rb)
        rows.append(row)
    return {"rows": rows}


def _verdict(a: dict, b: dict, c: dict) -> dict:
    """三数据集 → 逐条采纳标准判定（缺盘照片如实计入 caveat，不静默放行）。"""
    c16 = [r for r in c["rows"] if r["sample_id"] == "16#"]
    c8 = [r for r in c["rows"] if r["sample_id"].startswith("8#")]
    c_other = [r for r in c["rows"] if r["sample_id"] != "16#"
               and not r["sample_id"].startswith("8#")]
    checks = {
        "A_rescued_quad_ge5": a["n_rescued_quad_from_hull"] >= 5,
        "A_no_new_failures": a["n_new_failures"] == 0,
        "B_zero_count_change": b["n_count_diff"] == 0,
        "B_zero_drift_over_tol": b["n_drift_gt_tol"] == 0,
        "C_16_rescued_iou": all(isinstance(r.get("iou_vs_manual"), float)
                                and r["iou_vs_manual"] >= 0.97 for r in c16) and len(c16) == 2,
        "C_8_still_rejected": all(str(r["b"]).startswith("err") for r in c8),
        "C_others_zero_drift": all(r.get("drift_px", 0.0) == 0.0 for r in c_other
                                   if "drift_px" in r),
    }
    return {"checks": checks, "pass": all(checks.values()),
            "caveats": ([f"B 缺盘 {b['n_missing']} 张未复核"] if b["n_missing"] else [])}


def main() -> int:
    """三数据集双跑 → 采纳判定 → 落盘。"""
    api = _glassapp_api()
    cfg = api["load_app_config"]()
    assert float(cfg["sheets"]["close_frac"]) == CLOSE_A, \
        "config 现值已非 0.005——回归基线口径变了，先人工核对"
    field = json.loads((ROOT / "data" / "derived" / "texture_w_field" /
                        "values.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"A：{len(field['failures'])} 张失败照双跑 …", flush=True)
    a = run_failures(api, cfg, field)
    print(f"  quad_from_hull 救回 {a['n_rescued_quad_from_hull']}/7，"
          f"新增失败 {a['n_new_failures']}", flush=True)

    print(f"B：{len(field['all'])} 张样本双跑 …", flush=True)
    b = run_samples(api, cfg, field)
    print(f"  可核 {b['n_checked']}（多片 {b['n_multi_checked']}），缺盘 {b['n_missing']}；"
          f"片数变化 {b['n_count_diff']}，漂移>tol {b['n_drift_gt_tol']}"
          f"（最大 {b['worst_drift_px']}px）", flush=True)

    print("C：26 片批 28 张双跑 …", flush=True)
    c = run_sample26(api, cfg)

    verdict = _verdict(a, b, c)
    values = {
        "meta": {"date": DATE, "script": "fringe_scoring/make_closefrac_regression_assets.py",
                 "close_a": CLOSE_A, "close_b": CLOSE_B, "drift_tol_px": DRIFT_TOL_PX,
                 "确定性说明": "角点相同 ⇒ 矫正图相同 ⇒ 下游指标逐位相同（管线无随机性），"
                              "故 B 不重打分，只对拍片数与角点",
                 "fallback_不做的原因": "碎段凸包并集重拟合须动三端 cv2 精确移植面四文件；"
                                       "0.01 已救回同签名失败且距假片阈值 0.02 有 2 倍裕量"},
        "A_failures": a, "B_samples": b, "C_sample26": c, "verdict": verdict,
    }
    (OUT / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n判定：{'PASS' if verdict['pass'] else 'FAIL'}  {verdict['checks']}")
    if verdict["caveats"]:
        print(f"注意：{verdict['caveats']}")
    print(f"资产 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

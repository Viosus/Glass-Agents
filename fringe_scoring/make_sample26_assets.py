"""26 块带厚度标注样片「三合一」资产生成器（开发壳，不随核心包交付）。

三件事（2026-08-04 到样，规格表见 manifest）：
① 六指标全量 + 普白钢化按厚度分档统计 → refs_by_thickness 初标草案（只出报告与
   可粘贴 yaml 片段，GlassApp config 零改动——用户 2026-08-04 拍板）；
② 未钢化对照（8#-1/8#-2）验指标零点 + ε 条款候选窗口（不写 config，TODO(plant)）；
③ 已知物理尺寸（610×610 / 510×360 mm）反解逐张 mm/px（首次非产线空间标定）。

口径要点：
- 一遍打分流：检出角点 → compute_pipeline 一次 → 由矫正尺寸反解 mm/px →
  逐片 deepcopy(cfg) 注入 calibration.{mm_per_px, thickness_mm} → 六指标。
  mm_per_px 只被 compute_sheet_indicators 的 W_w 链路消费，pipeline 无需重跑；
  每片另跑 mm_per_px=None 的原生对照口径（只多一次 GLCM）。
- 角点三分支：manifest.manual_quad > 自动检测 > detection_failed（不出值，等人工定角）。
- 几何 QA：检出片数 / 两轴 mm/px 相对差（aniso）/ mm/px 合理域；容差在 manifest.qa。
- 防污染硬断言：进分档统计必须 core + qa 非 fail + provenance∈{auto,manual} + 单片；
  scored + excluded_from_stats + failed = 照片总数守恒。
- ⚠️ 本批照片为 ~650×630 导出缩图（非相机原始分辨率），mm/px≈1.0 是导出缩放的
  产物；「重采样 1px/mm」在此近似恒等，原始光学分辨率已丢——报告须披露。

算法调用 D:\\GlassApp\\fringe_scoring 权威实现（v1.12 texture_w 版）；本仓自带的
fringe_scoring 是 ccp 旧版，启动断言防混用。照片不入库；本脚本产物入库。
用法：venv python fringe_scoring/make_sample26_assets.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

GLASSAPP = Path(r"D:\GlassApp")
ROOT = Path(__file__).resolve().parents[1]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


def _glassapp_api() -> dict:
    """运行期载入 GlassApp 权威实现（v1.12 texture_w 版）并做版本护栏断言。

    不放模块顶层：pytest 从本仓收集时 `fringe_scoring` 包名会先解析到本仓
    v1.10 旧副本（ccp/无 texture_w），顶层 import 必撞缓存假阳性；单测只测
    本文件的纯函数，不需要 GlassApp。
    """
    if str(GLASSAPP) not in sys.path:
        sys.path.insert(0, str(GLASSAPP))
    cached = sys.modules.get("fringe_scoring")
    if cached is not None and not Path(cached.__file__).resolve().is_relative_to(GLASSAPP):
        raise RuntimeError(
            f"fringe_scoring 已被解析到 {cached.__file__}（本仓旧副本）——"
            "请以脚本方式直接运行本文件，勿在已 import 本仓包的进程里调用")
    import fringe_scoring
    assert Path(fringe_scoring.__file__).resolve().is_relative_to(GLASSAPP), (
        f"fringe_scoring 解析到 {fringe_scoring.__file__}，应为 GlassApp 权威实现")
    from app.config_store import load_app_config
    from fringe_scoring.indicators import compute_sheet_indicators
    from fringe_scoring.score import compute_pipeline, score_from_pipeline
    from fringe_scoring.sheets import detect_sheet_quads_with_meta
    return {"load_app_config": load_app_config,
            "compute_sheet_indicators": compute_sheet_indicators,
            "compute_pipeline": compute_pipeline,
            "score_from_pipeline": score_from_pipeline,
            "detect_sheet_quads_with_meta": detect_sheet_quads_with_meta}

DATE = "2026-08-04"
MANIFEST = ROOT / "data" / "derived" / "sample26_thickness" / "manifest.yaml"
OUT = MANIFEST.parent
# refs_by_thickness 档内可调的五键（与 GlassApp config_page.BAND_KEYS 一致；
# position_score 不入厚度档，恒继承默认 refs）
BAND_REF_KEYS = ("x095", "gray_variance", "gradient_mean", "gradient_variance", "texture_w")
# 分档统计出草案的厚度（n≥2）；n=1 厚度只出观测值不出 refs（见 _refs_draft）
DRAFT_BANDS_MM = (4, 5, 6, 8)

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _load_manifest() -> dict:
    """读名册并校验：登记必在盘、盘上图必登记（规格表.jpeg 除外）、文件名唯一。"""
    man = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    photo_dir = ROOT / man["photo_dir"]
    files = [p["file"] for p in man["photos"]]
    assert len(files) == len(set(files)), "manifest 有重复文件名"
    missing = [f for f in files if not (photo_dir / f).exists()]
    assert not missing, f"manifest 登记但盘上缺图：{missing}"
    on_disk = {p.name for p in photo_dir.glob("*.png")}
    unlisted = on_disk - set(files)
    assert not unlisted, f"盘上有图未登记：{sorted(unlisted)}"
    man["_photo_dir"] = photo_dir
    return man


def _load_gray(path: Path) -> np.ndarray:
    """照片 → float 灰度（imdecode 走 np.fromfile，中文路径安全）。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读图失败：{path}")
    return np.asarray(img, dtype=float)


def _mm_per_px(spec_mm: list, warp_w: int, warp_h: int) -> dict:
    """已知物理尺寸 + 矫正像素尺寸 → mm/px（长边配长边；正方形规格免配准）。

    两轴各得一个估计，相对差 aniso = |m_long/m_short − 1|——它同时就是
    「矫正长宽比 vs 规格长宽比」的偏差，是几何 QA 的主判据。
    """
    spec_long, spec_short = sorted((float(spec_mm[0]), float(spec_mm[1])), reverse=True)
    warp_long, warp_short = sorted((int(warp_w), int(warp_h)), reverse=True)
    m_long = spec_long / warp_long
    m_short = spec_short / warp_short
    return {
        "long": round(m_long, 5), "short": round(m_short, 5),
        "mean": round((m_long + m_short) / 2.0, 5),
        "aniso": round(abs(m_long / m_short - 1.0), 5),
    }


def _qa_check(n_sheets: int, expected: int, mppx: dict | None,
              provenance: str, qa_cfg: dict) -> dict:
    """几何 QA 分级：fail=不出指标/不进统计；warn=出值进统计带标记。"""
    reasons = []
    level = "pass"
    if provenance == "detection_failed":
        return {"level": "fail", "reasons": ["检测失败（前景为空）"]}
    if n_sheets != expected:
        reasons.append(f"检出片数 {n_sheets}≠{expected}")
        level = "fail"
    if mppx is not None:
        if mppx["aniso"] > qa_cfg["aniso_fail"]:
            reasons.append(f"两轴 mm/px 相对差 {mppx['aniso']:.1%} > {qa_cfg['aniso_fail']:.0%}")
            level = "fail"
        elif mppx["aniso"] > qa_cfg["aniso_warn"]:
            reasons.append(f"两轴 mm/px 相对差 {mppx['aniso']:.1%} > {qa_cfg['aniso_warn']:.0%}")
            level = "warn" if level == "pass" else level
        if not qa_cfg["mm_per_px_lo"] <= mppx["mean"] <= qa_cfg["mm_per_px_hi"]:
            reasons.append(f"mm/px={mppx['mean']} 出合理域 "
                           f"[{qa_cfg['mm_per_px_lo']}, {qa_cfg['mm_per_px_hi']}]")
            level = "fail"
    if provenance == "fullframe":
        reasons.append("整图口径（未检出玻璃边界，含背景污染）")
        level = "warn" if level == "pass" else level
    return {"level": level, "reasons": reasons}


def _ind_payload(ind) -> dict:
    """SheetIndicators → 落盘字段（展示位数四舍五入；W_w 保 4 位——报数精度要求）。"""
    d = ind.position_diagnostics
    return {
        "x095": round(ind.x095, 1), "x095_unit": ind.x095_unit,
        "gray_variance": round(ind.gray_variance, 1),
        "gradient_mean": round(ind.gradient_mean, 2),
        "gradient_variance": round(ind.gradient_variance, 1),
        "texture_w": round(ind.texture_w, 4),
        "texture_w_ca_w": round(ind.texture_w_ca_w, 4),
        "texture_w_cpa_w": round(ind.texture_w_cpa_w, 1),
        "texture_w_dynamic_range": round(ind.texture_w_dynamic_range, 2),
        "texture_w_degenerate": bool(ind.texture_w_degenerate),
        "position_score": round(ind.position_score, 1),
        "center_concentration": round(ind.center_concentration, 4),
        "weighted_coverage": round(float(d.get("weighted_coverage", float("nan"))), 3),
        "binding_branch": d.get("binding_branch"),
        "sub_scores": {k: round(v, 1) for k, v in ind.sub_scores.items()},
        "total_score": round(ind.total_score, 2),
        "verification": ind.verification,
    }


def _score_photo(entry: dict, base_cfg: dict, qa_cfg: dict, photo_dir: Path,
                 api: dict) -> dict:
    """一张照片的完整记录：角点三分支 → 一遍流 → mm/px → QA → 双口径指标。"""
    rec: dict = {k: entry.get(k) for k in
                 ("file", "sample_id", "category", "type", "thickness_mm", "spec_mm", "notes")}
    arr = _load_gray(photo_dir / entry["file"])
    rec["image_px"] = [int(arr.shape[1]), int(arr.shape[0])]

    # ── 角点三分支 ──
    if entry.get("manual_quad"):
        quad = np.asarray(entry["manual_quad"], dtype=float)
        provenance, n_sheets = "manual", 1
    else:
        try:
            quads, _metas = api["detect_sheet_quads_with_meta"](arr, base_cfg)
            n_sheets = len(quads)
            quad = quads[0] if n_sheets == 1 else None
            provenance = "auto"
        except ValueError as e:
            rec["detection"] = {"provenance": "detection_failed", "n_sheets": 0,
                                "error": str(e)[:160]}
            rec["mm_per_px"] = None
            rec["qa"] = _qa_check(0, entry["expected_sheets"], None, "detection_failed", qa_cfg)
            rec["indicators"] = None
            return rec
    rec["detection"] = {"provenance": provenance, "n_sheets": n_sheets}

    if quad is None:  # 片数不符：只记检测信息，指标空缺等人工定角
        rec["mm_per_px"] = None
        rec["qa"] = _qa_check(n_sheets, entry["expected_sheets"], None, provenance, qa_cfg)
        rec["indicators"] = None
        return rec

    # ── 一遍流：pipeline 只算一次 ──
    pipe = api["compute_pipeline"](arr, config=base_cfg, quad_corners=quad)
    warp_h, warp_w = pipe.arr.shape
    rec["warp_px"] = [int(warp_w), int(warp_h)]
    mppx = _mm_per_px(entry["spec_mm"], warp_w, warp_h)
    rec["mm_per_px"] = mppx
    rec["qa"] = _qa_check(n_sheets, entry["expected_sheets"], mppx, provenance, qa_cfg)
    if rec["qa"]["level"] == "fail":
        rec["indicators"] = None
        return rec

    cfg_cal = copy.deepcopy(base_cfg)
    calib = cfg_cal["indicators"]["calibration"]
    calib["mm_per_px"] = mppx["mean"]
    calib["thickness_mm"] = entry.get("thickness_mm")
    r = api["score_from_pipeline"](pipe, cfg_cal)
    ind = api["compute_sheet_indicators"](pipe.arr, ~r.border_mask_arr, cfg_cal, pipeline=pipe)
    rec["indicators"] = _ind_payload(ind)

    # 原生对照口径（不重采样）：只为量化 mm/px 注入对 W_w 的实际影响
    cfg_nat = copy.deepcopy(base_cfg)
    cfg_nat["indicators"]["calibration"]["mm_per_px"] = None
    ind_nat = api["compute_sheet_indicators"](pipe.arr, ~r.border_mask_arr, cfg_nat, pipeline=pipe)
    rec["indicators_native"] = {
        "texture_w": round(ind_nat.texture_w, 4),
        "texture_w_dynamic_range": round(ind_nat.texture_w_dynamic_range, 2),
    }
    # 内缩 5% 的动态范围（诊断量：剔除边缘辉光/矫正边界对 max−min 的污染）
    mh, mw = max(1, int(0.05 * warp_h)), max(1, int(0.05 * warp_w))
    inset = pipe.arr[mh:warp_h - mh, mw:warp_w - mw]
    rec["dynamic_range_inset5"] = round(float(np.max(inset) - np.min(inset)), 2)
    return rec


def _stats_eligible(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """分档统计的入围过滤 + 防污染硬断言；返回 (入围, 排除清单)。"""
    eligible, excluded = [], []
    for r in records:
        why = []
        if r["category"] != "core":
            why.append(f"category={r['category']}")
        if r["indicators"] is None:
            why.append("无指标值")
        if r["qa"]["level"] == "fail":
            why.append("QA fail")
        if r["detection"]["provenance"] not in ("auto", "manual"):
            why.append(f"provenance={r['detection']['provenance']}")
        elif r["detection"]["n_sheets"] != 1:
            why.append(f"n_sheets={r['detection']['n_sheets']}")
        if why:
            excluded.append({"sample_id": r["sample_id"], "file": r["file"], "reasons": why})
        else:
            eligible.append(r)
    for r in eligible:  # 双保险：入围者违反任一不变量即崩，绝不静默污染统计
        assert r["category"] == "core" and r["indicators"] is not None
        assert r["qa"]["level"] != "fail" and r["detection"]["n_sheets"] == 1
        assert r["detection"]["provenance"] in ("auto", "manual")
    return eligible, excluded


def _band_stats(eligible: list[dict]) -> list[dict]:
    """普白钢化按公称厚度分档：n、成员、六指标 mean/min/max。"""
    bands = []
    for t in sorted({r["thickness_mm"] for r in eligible}):
        members = [r for r in eligible if r["thickness_mm"] == t]
        row: dict = {"thickness_mm": t, "n": len(members),
                     "members": [m["sample_id"] for m in members]}
        for key in (*BAND_REF_KEYS, "position_score", "total_score"):
            vals = np.array([m["indicators"][key] for m in members], dtype=float)
            row[key] = {"mean": round(float(vals.mean()), 4),
                        "min": round(float(vals.min()), 4),
                        "max": round(float(vals.max()), 4)}
        bands.append(row)
    return bands


def _refs_draft(bands: list[dict]) -> dict:
    """refs_by_thickness 初标草案：n≥2 档取批内 min/max，n=1 档只留 TODO 注释。

    方向：五键全部「越小越好」→ best=min、worst=max；断言 best<worst（相等即
    映射退化，宁缺勿滥）。草案仅供报告粘贴，不写 config。
    """
    draft = []
    for b in bands:
        if b["thickness_mm"] not in DRAFT_BANDS_MM:
            continue
        assert b["n"] >= 2, f"{b['thickness_mm']}mm 档 n={b['n']}，不足以出 refs"
        refs = {}
        for key in BAND_REF_KEYS:
            best, worst = b[key]["min"], b[key]["max"]
            assert best < worst, f"{b['thickness_mm']}mm 档 {key} best==worst，无法定标"
            refs[key] = {"best": best, "worst": worst}
        draft.append({"max_thickness_mm": float(b["thickness_mm"]),
                      "fail_line": None, "refs": refs})
    return {"refs_by_thickness": draft}


def _avg_ranks(v) -> np.ndarray:
    """并列取平均秩（同 make_texture_w_assets._avg_ranks，跨仓 import 撞包名故复制）。"""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    pos = np.empty(v.size, dtype=float)
    pos[order] = np.arange(v.size, dtype=float)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=pos) / cnt)[inv]


def _spearman(x, y) -> float:
    """Spearman 秩相关，并列取平均秩。"""
    return float(np.corrcoef(_avg_ranks(x), _avg_ranks(y))[0, 1])


def _scatter_png(records: list[dict], eligible: list[dict]) -> str:
    """W_w vs 厚度散点（灰阶+形状区分四组，单轴，网格克制；dataviz 口径）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)

    core = [(r["thickness_mm"], r["indicators"]["texture_w"]) for r in eligible]
    n1 = [(r["thickness_mm"], r["indicators"]["texture_w"]) for r in records
          if r["category"] == "special" and r["indicators"] and r["thickness_mm"]
          and r["thickness_mm"] >= 10]
    spec = [(r["thickness_mm"], r["indicators"]["texture_w"], r["sample_id"]) for r in records
            if r["category"] == "special" and r["indicators"]
            and r["thickness_mm"] is not None and r["thickness_mm"] < 10]
    ctrl = [(r["thickness_mm"], r["indicators"]["texture_w"]) for r in records
            if r["category"] == "control" and r["indicators"]]

    ax.scatter([t for t, _ in core], [w for _, w in core], s=42, c="#333333",
               marker="o", label="普白钢化（分档统计）", zorder=3)
    # 档均值连线（趋势参考，仅 n≥2 档）
    by_t: dict = {}
    for t, w in core:
        by_t.setdefault(t, []).append(w)
    ts = sorted(t for t, ws in by_t.items() if len(ws) >= 2)
    ax.plot(ts, [float(np.mean(by_t[t])) for t in ts], c="#333333", lw=2,
            alpha=0.65, label="档均值（n≥2）", zorder=2)
    if n1:
        ax.scatter([t for t, _ in n1], [w for _, w in n1], s=52, facecolors="none",
                   edgecolors="#333333", marker="s", label="特殊工艺 n=1（均质/高应力）",
                   zorder=3)
    if spec:
        ax.scatter([t for t, _, _ in spec], [w for _, w, _ in spec], s=46, c="#8a8a8a",
                   marker="^", label="特种片（超白/Low-E/压花）", zorder=3)
        for t, w, sid in spec:
            ax.annotate(sid, (t, w), textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color="#6b6b6b")
    if ctrl:
        ax.scatter([t for t, _ in ctrl], [w for _, w in ctrl], s=46, c="#b0b0b0",
                   marker="D", label="未钢化对照", zorder=3)
    ax.set_xlabel("公称厚度（mm）")
    ax.set_ylabel("纹理指数 W_w（越大越差）")
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = OUT / "w_by_thickness.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    """全批打分 → QA 摘要 → 统计/草案/图 → 落盘。"""
    api = _glassapp_api()
    man = _load_manifest()
    photo_dir = man["_photo_dir"]
    qa_cfg = man["qa"]
    cfg = api["load_app_config"]()
    cfg_sha = hashlib.sha256(
        json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    records = []
    for entry in man["photos"]:
        rec = _score_photo(entry, cfg, qa_cfg, photo_dir, api)
        records.append(rec)
        ind = rec["indicators"]
        tag = (f"W_w={ind['texture_w']:.4f} 总分={ind['total_score']:6.2f}"
               if ind else f"无指标（{rec['qa']['reasons']}）")
        print(f"{rec['file']:10s} {rec['sample_id']:5s} [{rec['category']:7s}] "
              f"qa={rec['qa']['level']:4s} {tag}")

    scored = [r for r in records if r["indicators"] is not None]
    failed = [r for r in records if r["detection"].get("provenance") == "detection_failed"
              or r["indicators"] is None]
    eligible, excluded = _stats_eligible(records)
    assert len(scored) + len([r for r in records if r["indicators"] is None]) == len(records)

    bands = _band_stats(eligible)
    draft = _refs_draft(bands)
    tw = [(r["thickness_mm"], r["indicators"]["texture_w"]) for r in eligible]
    sp = _spearman([t for t, _ in tw], [w for _, w in tw]) if len(tw) >= 3 else None

    # ε 候选窗口：对照片 DR 上界 vs 钢化片 DR 下界（双口径）
    ctrl = [r for r in records if r["category"] == "control" and r["indicators"]]
    temp = [r for r in scored if r["category"] != "control"]
    epsilon = {
        "note": "官方判据（整片 max−min）实测无区分力——片上标签纸/边缘辉光把对照片 DR "
                "拉到与钢化片同域；内缩 5% 口径才分离。ε 落地须 (a) 无标签样本重测或 "
                "(b) 判据改内缩域（工程决策），且本批为导出缩图非产线原生分辨率——"
                "ε 定值仍 TODO(plant)",
        "controls_dr": [r["indicators"]["texture_w_dynamic_range"] for r in ctrl],
        "controls_dr_inset5": [r.get("dynamic_range_inset5") for r in ctrl],
        "tempered_dr_min": (min(r["indicators"]["texture_w_dynamic_range"] for r in temp)
                            if temp else None),
        "tempered_dr_inset5_min": (min(r["dynamic_range_inset5"] for r in temp
                                       if r.get("dynamic_range_inset5") is not None)
                                   if temp else None),
        "window_official": ([max(r["indicators"]["texture_w_dynamic_range"] for r in ctrl),
                             min(r["indicators"]["texture_w_dynamic_range"] for r in temp)]
                            if ctrl and temp else None),
        "window_inset5": ([max(r["dynamic_range_inset5"] for r in ctrl),
                           min(r["dynamic_range_inset5"] for r in temp
                               if r.get("dynamic_range_inset5") is not None)]
                          if ctrl and temp else None),
    }

    # 双照重复性（1#/16#：同一样片两张照片）
    repeat = {}
    by_sid: dict = {}
    for r in scored:
        by_sid.setdefault(r["sample_id"], []).append(r)
    for sid, rs in by_sid.items():
        if len(rs) == 2:
            repeat[sid] = {
                "d_texture_w": round(abs(rs[0]["indicators"]["texture_w"]
                                         - rs[1]["indicators"]["texture_w"]), 4),
                "d_mm_per_px": (round(abs(rs[0]["mm_per_px"]["mean"]
                                          - rs[1]["mm_per_px"]["mean"]), 5)
                                if rs[0]["mm_per_px"] and rs[1]["mm_per_px"] else None),
                "d_total": round(abs(rs[0]["indicators"]["total_score"]
                                     - rs[1]["indicators"]["total_score"]), 2),
            }

    # 第三方 L2（Softsolution 线扫）交叉对照：nm 域分位 vs 本报告灰度域指标。
    # 同一样片两拍时取拍间均值；13#（L2 测量失败）自然缺席。
    l2 = {e["sample_id"]: e for e in man.get("l2_reference", []) if e["q95_nm"] is not None}
    l2_rows = []
    for sid, ref in l2.items():
        rs = [r for r in scored if r["sample_id"] == sid]
        if not rs:
            continue
        l2_rows.append({
            "sample_id": sid, "match": ref["match"],
            "q95_nm": ref["q95_nm"], "q98_nm": ref["q98_nm"],
            "x095_gray": round(float(np.mean([r["indicators"]["x095"] for r in rs])), 1),
            "texture_w": round(float(np.mean([r["indicators"]["texture_w"] for r in rs])), 4),
        })
    l2_cross = {"rows": l2_rows}
    if len(l2_rows) >= 3:
        q95 = [r["q95_nm"] for r in l2_rows]
        l2_cross["spearman_x095_vs_q95"] = round(
            _spearman([r["x095_gray"] for r in l2_rows], q95), 4)
        l2_cross["spearman_texture_w_vs_q95"] = round(
            _spearman([r["texture_w"] for r in l2_rows], q95), 4)
        same = [r for r in l2_rows if r["match"] == "same_sheet"]
        if len(same) >= 3:
            l2_cross["spearman_x095_vs_q95_same_sheet_only"] = round(
                _spearman([r["x095_gray"] for r in same], [r["q95_nm"] for r in same]), 4)
        l2_cross["note"] = (
            "L2 评估域=扣角 75mm/扣边 25mm、nm 域；本报告=扣免罚边框带、灰度域，"
            "两次成像相隔一月——只作秩一致性对照，不构成 gray→nm 标定")

    values = {
        "meta": {"date": DATE, "config_sha256_16": cfg_sha,
                 "script": "fringe_scoring/make_sample26_assets.py",
                 "评估域": "整片矫正图（W_w）/ 扣免罚边框带内部（其余五项）",
                 "口径": "逐片 mm/px 注入重采样 1px/mm；照片为导出缩图，见 ⚠️ 披露"},
        "n_photos": len(records), "n_scored": len(scored),
        "n_no_indicators": len(records) - len(scored),
        "photos": records,
        "bands": bands,
        "eligible_n": len(eligible),
        "excluded_from_stats": excluded,
        "spearman_w_vs_thickness": (round(sp, 4) if sp is not None else None),
        "epsilon": epsilon,
        "repeatability": repeat,
        "l2_cross": l2_cross,
        "refs_draft": draft,
    }
    values["scatter"] = _scatter_png(records, eligible)
    (OUT / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=1), encoding="utf-8")

    draft_text = (
        "# refs_by_thickness 初标草案（26 片带厚度批，2026-08-04）\n"
        "# ⚠️ 批内极值初值（best=档内最好、worst=档内最差），非好/差样片人工标定；\n"
        "#    每档 n=2~5；发档使用前须同厚度好/废片人工复标。\n"
        "# 粘贴位置：GlassApp config app_config.yaml 的 app.plans.<方案名>.refs_by_thickness\n"
        "#（顶层 indicators.refs_by_thickness 会被 apply_active_plan 覆盖，别粘那里）。\n"
        "# 选档口径：max_thickness_mm ≥ 实测厚度 中取最小档；厚度超出全部档（如本批\n"
        "# 10/12/15mm）会静默回退默认 refs——不是回退最大档。\n"
        "# n=1 厚度不出档（单点无区间）：10mm(7# 均质)、12mm(14# 高应力防火)、\n"
        "# 15mm(15#)——TODO(plant) 待补样后定。\n"
        + yaml.safe_dump(draft, allow_unicode=True, sort_keys=False)
    )
    (OUT / "refs_by_thickness_draft.yaml").write_text(draft_text, encoding="utf-8")

    n_fail = len([r for r in records if r["qa"]["level"] == "fail"])
    print(f"\n照片 {len(records)}：出值 {len(scored)}、无指标 {len(records) - len(scored)}"
          f"（QA fail {n_fail}）；分档入围 {len(eligible)}、排除 {len(excluded)}")
    for b in bands:
        print(f"  {b['thickness_mm']:>4}mm n={b['n']} {','.join(b['members'])}"
              f"  W_w [{b['texture_w']['min']:.4f}, {b['texture_w']['max']:.4f}]"
              f" 均值 {b['texture_w']['mean']:.4f}")
    if sp is not None:
        print(f"Spearman(W_w, 厚度) = {sp:+.4f}（并列平均秩，n={len(tw)}）")
    if epsilon["window_official"]:
        wo, wi = epsilon["window_official"], epsilon["window_inset5"]
        sep_o = "分离" if wo[0] < wo[1] else "无区分力（倒置）"
        sep_i = "分离" if wi[0] < wi[1] else "无区分力（倒置）"
        print(f"ε 官方口径（整片 DR）窗口 ({wo[0]}, {wo[1]}) → {sep_o}")
        print(f"ε 内缩5% 口径窗口 ({wi[0]}, {wi[1]}) → {sep_i}")
    if "spearman_x095_vs_q95" in l2_cross:
        print(f"L2 对照（n={len(l2_rows)}）：x095(灰度) vs q95(nm) "
              f"ρ={l2_cross['spearman_x095_vs_q95']:+.4f}；"
              f"W_w vs q95 ρ={l2_cross['spearman_texture_w_vs_q95']:+.4f}")
    print(f"资产 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

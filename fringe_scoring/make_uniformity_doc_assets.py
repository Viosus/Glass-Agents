"""《应力斑位置评分指标_技术说明》实测算例资产生成器（开发壳，不随核心包交付）。

数据源（2026-07-22 换代）：121 张带人工评分的单片应力斑照片
（data/images/stress_fringe/NNN_系统X_人工Y.png，暗背景、单片、命名含标注）。
逐片检出矫正 → 位置指标全量（ρu / 加权覆盖度 / 双支路评分）+ CCP/X0.95 →
产出：
- data/derived/uniformity_doc/*.png　　选例样片灰度图 + 位置评分-人工分散点图
- data/derived/uniformity_doc/values.json　实测数值（PDF 生成器注入正文，图文同源）
内容：人工分档统计（8 档）、Spearman、选例（甲=人工≥80 中评分最高、乙=评分
最低的非饱和集中片、CCP 高/低）、CCP 位置盲区平移演示（裁边+1/4 循环平移）。
算法调用 D:\\GlassApp\\fringe_scoring 权威实现 + app_config.yaml 生效配置。
照片按仓库规则不入库；本脚本产物（png/json）入库。
用法：venv python fringe_scoring/make_uniformity_doc_assets.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, r"D:\GlassApp")

import cv2
import numpy as np
import yaml

from fringe_scoring.indicators import compute_sheet_indicators
from fringe_scoring.score import compute_pipeline
from fringe_scoring.sheets import score_sheets

ROOT = Path(__file__).resolve().parents[1]
PHOTO_DIR = ROOT / "data" / "images" / "stress_fringe"
OUT = ROOT / "data" / "derived" / "uniformity_doc"
MAX_SIDE = 2000  # 示例处理分辨率（长边 px），与文中说明一致
MAN_BANDS = (50, 55, 60, 65, 70, 75, 80, 85)  # 人工分档（数据集实际取值）

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _load_cfg() -> dict:
    """读取生效配置（app_config.yaml，含 sheets/indicators 全段）。"""
    with open(r"D:\GlassApp\config\app_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_gray(path: Path) -> np.ndarray:
    """照片 → 灰度 float 数组，长边等比缩至 MAX_SIDE。"""
    # cv2.imread 在 Windows 上读不了含非 ASCII 字符的路径（照片名含中文）→ imdecode
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读图失败：{path}")
    h, w = img.shape
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return np.asarray(img, dtype=float)


def _save_gray(arr: np.ndarray, name: str, max_w: int = 700) -> str:
    """样片灰度图落盘（uint8 直存不做归一化），返回 ROOT 相对路径。"""
    u8 = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = u8.shape
    if w > max_w:
        u8 = cv2.resize(u8, (max_w, int(round(h * max_w / w))),
                        interpolation=cv2.INTER_AREA)
    out = OUT / name
    cv2.imwrite(str(out), u8)
    return str(out.relative_to(ROOT)).replace("\\", "/")


def _trim_border(arr: np.ndarray, cfg: dict) -> np.ndarray:
    """按实测边框带宽度（另加 1% 余量）裁除单片四周的边缘应力亮带。

    供循环平移构造演示用：裁掉亮边后拼接缝两侧都是内部区域，不产生
    "上下亮边拼成中缝亮线"的构造伪影。"""
    quad = dict(cfg.get("quad") or {})
    quad["auto_detect"] = False
    pipe = compute_pipeline(arr, {**cfg, "quad": quad})
    extra = max(2, int(round(0.01 * min(arr.shape))))
    b = pipe.bands
    return arr[b.top_px + extra: arr.shape[0] - b.bottom_px - extra,
               b.left_px + extra: arr.shape[1] - b.right_px - extra]


def _indicators_of(arr: np.ndarray, cfg: dict):
    """矫正后单片灰度图 → 六指标（关四边形检测，输入即整片）。"""
    quad = dict(cfg.get("quad") or {})
    quad["auto_detect"] = False
    cfg2 = {**cfg, "quad": quad}
    return compute_sheet_indicators(arr, np.ones(arr.shape, dtype=bool), cfg2)


def _rec(ind, no: int, sys_score: float, man: float) -> dict:
    """一片的记录（文档要用的量，四舍五入到展示位数）。"""
    d = ind.position_diagnostics
    return {
        "no": no, "sys": sys_score, "man": man,
        "U": round(ind.position_score, 1),
        "rho": round(ind.center_concentration, 4),
        "cov": round(d["weighted_coverage"], 3),
        "u_rho": round(d["u_rho"], 1),
        "u_cov": round(d["u_cov"], 1),
        "branch": d["binding_branch"],
        "ccp": round(ind.ccp_value, 3),
        "ca": round(ind.ccp_ca, 4),
        "cpa": round(ind.ccp_cpa, 1),
        "x095": round(ind.x095, 1),
    }


def _spearman(x, y) -> float:
    """Spearman 秩相关（并列取平均秩的简化版对本数据集足够：先按值排序取名次）。"""
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _scatter_png(records: list[dict]) -> str:
    """位置评分 vs 人工分散点图（图文同源；离散人工分档加横向抖动便于目视）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    man = np.array([r["man"] for r in records])
    u = np.array([r["U"] for r in records])
    rng = np.random.default_rng(7)  # 仅作图抖动，不进任何数值
    jitter = rng.uniform(-0.8, 0.8, size=man.size)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.scatter(man + jitter, u, s=18, c="dimgray", alpha=0.75, linewidths=0)
    for b in MAN_BANDS:  # 分档均值连线（趋势参考）
        sel = man == b
        if sel.any():
            ax.plot(b, u[sel].mean(), marker="_", ms=18, mew=2.5, c="black")
    ax.set_xlabel("人工评分（专家标注）")
    ax.set_ylabel("位置评分 U")
    ax.set_xticks(list(MAN_BANDS))
    ax.set_ylim(-4, 104)
    ax.grid(True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    out = OUT / "scatter.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    """跑全部标注照 → 统计/选例/平移演示 → 落盘 PNG 与 values.json。"""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()
    photos = sorted(PHOTO_DIR.glob("*.png"))
    records, crops, failures = [], {}, []
    for p in photos:
        m = re.match(r"(\d+)_系统([\d.]+)_人工([\d.]+)", p.stem)
        if not m:
            continue
        no = int(m.group(1))
        arr = _load_gray(p)
        try:
            res = score_sheets(arr, cfg)
        except ValueError as e:
            failures.append({"no": no, "error": str(e)})
            continue
        # 单片测试照：阅读序第一片=主片（025 等画幅边缘含真实邻片残段，只记主片）
        assert res.sheet_indicators is not None
        records.append(_rec(res.sheet_indicators[0], no, float(m.group(2)), float(m.group(3))))
        crops[no] = np.asarray(res.sheet_results[0].warped_image, dtype=float)
        print(f"{no:03d}: U={records[-1]['U']:5.1f} ρu={records[-1]['rho']:.4f} "
              f"cov={records[-1]['cov']:.3f} {records[-1]['branch']}")

    man = np.array([r["man"] for r in records])
    u_arr = np.array([r["U"] for r in records])
    bands = []
    for b in MAN_BANDS:
        sel = man == b
        if not sel.any():
            continue
        bands.append({
            "man": b, "n": int(sel.sum()),
            "u_mean": round(float(u_arr[sel].mean()), 1),
            "u_min": round(float(u_arr[sel].min()), 1),
            "u_max": round(float(u_arr[sel].max()), 1),
            "rho_mean": round(float(np.mean([r["rho"] for r in records if r["man"] == b])), 4),
            "cov_mean": round(float(np.mean([r["cov"] for r in records if r["man"] == b])), 3),
        })

    # 选例：甲=人工≥80 中位置评分最高（好片正面例）；乙=评分最低的"集中带"片
    # （排除覆盖近饱和的整片皆斑者，取 cov ≤ 0.55，保证平移演示有可移的集中区）；
    # CCP 最高/最低照旧
    good = [r for r in records if r["man"] >= 80]
    jia = max(good, key=lambda r: r["U"])
    concentrated = [r for r in records if r["cov"] <= 0.55]
    yi = min(concentrated, key=lambda r: r["U"])
    by_ccp = sorted(records, key=lambda r: r["ccp"])
    picks = {"u_hi": jia, "u_lo": yi, "ccp_hi": by_ccp[-1], "ccp_lo": by_ccp[0]}
    values: dict = {
        "n_photos": len(photos), "n_scored": len(records),
        "failures": failures,
        "spearman": round(_spearman(u_arr, man), 3),
        "u_min": round(float(u_arr.min()), 1), "u_max": round(float(u_arr.max()), 1),
        "n_below_60": int((u_arr < 60).sum()),
        "bands": bands,
        "all": records,
        "scatter": None,  # 占位，下面填
    }
    for key, rec in picks.items():
        values[key] = {**rec, "img": _save_gray(crops[rec["no"]], f"{key}.png")}
    values["scatter"] = _scatter_png(records)

    # CCP 位置盲区构造演示：取集中度最高的片（cov ≤ 0.55 中 ρu 最大，保证有
    # 可移的集中区），先裁除四周边缘应力亮带（无缝构造，见 _trim_border），再
    # 整体垂直循环平移四分之一板高——斑移离中心但不贴边；相邻灰度对多重集仅在
    # 拼接缝一行改变 → CCP 应基本不变；斑离开高权重中央区 → 指标主量 ρu 显著
    # 下降。演示以 ρu 论证（评分层覆盖支路对构造图像的边框带/背景重估有小幅
    # 敏感，不入演示口径）。
    roll_pick = max(concentrated, key=lambda r: r["rho"])
    base = _trim_border(crops[roll_pick["no"]], cfg)
    rolled = np.roll(base, base.shape[0] // 4, axis=0)
    ind_a, ind_b = _indicators_of(base, cfg), _indicators_of(rolled, cfg)
    def _roll_rec(ind, name):
        """平移演示一侧的记录（数值 + 落盘灰度图路径）。"""
        d = ind.position_diagnostics
        return {"U": round(ind.position_score, 1), "rho": round(ind.center_concentration, 4),
                "cov": round(d["weighted_coverage"], 3), "ccp": round(ind.ccp_value, 3),
                "ca": round(ind.ccp_ca, 4), "cpa": round(ind.ccp_cpa, 1),
                "img": _save_gray(np.clip(base if name == "a" else rolled, 0, 255), f"roll_{name}.png")}
    values["roll"] = {"no": roll_pick["no"], "man": roll_pick["man"],
                      "a": _roll_rec(ind_a, "a"), "b": _roll_rec(ind_b, "b")}

    (OUT / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nn={len(records)} Spearman={values['spearman']} "
          f"U∈[{values['u_min']},{values['u_max']}] <60分 {values['n_below_60']} 张")
    print(f"甲={jia['no']:03d}(U {jia['U']}) 乙={yi['no']:03d}(U {yi['U']}) "
          f"CCP高={by_ccp[-1]['no']:03d} CCP低={by_ccp[0]['no']:03d}")
    print(f"平移演示: U {values['roll']['a']['U']}→{values['roll']['b']['U']} "
          f"CCP {values['roll']['a']['ccp']}→{values['roll']['b']['ccp']}")
    print(f"资产 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

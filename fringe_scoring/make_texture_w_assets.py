"""W_w（位置加权聚类突出指数，零标定常数）验证资产生成器（开发壳，不随核心包交付）。

在 115 片带双标注（系统分+人工分）的产线照片上验证 tools/metrics.texture_w：
1. 相关性：Spearman(W_w, 人工分/系统分)，**并列取平均秩**（2026-08-04 起；
   此前形式对照的 vs_man 为 argsort 口径写死值，已作废）。形式对照 + 两组配对
   bootstrap（B=2000）每次运行真复算落盘，数值见 values.json form_comparison；
2. 值域断言：全体 W ∈ [0,1)（理论上界 (2(Ng−1))⁴/12 的数学保证）；
3. 分档均值表 + 单调性计数；
4. 合成性质复核（经同一实现）：均匀退化 / 中心vs角落方向性 / iid 噪声钝感。

口径声明：**原生分辨率**（不缩放；mm/px=1 → d=1≡1mm 标准口径，2026-07-30 修正）。
幅度仍为 8-bit 强度域（gray→nm 标定仍缺），W_w 因仿射不变而不依赖该标定；
结论限"相关性与排序"层面。
用法：venv python fringe_scoring/make_texture_w_assets.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fringe_scoring.sheets import score_sheets  # noqa: E402
from tools import metrics  # noqa: E402

PHOTO_DIR = ROOT / "data" / "images" / "stress_fringe"
CFG_PATH = ROOT / "config" / "fringe_scoring.yaml"
OUT = ROOT / "data" / "derived" / "texture_w_doc"

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _load_cfg() -> dict:
    """读取本仓库生效配置（仅用于整床/单片检出与矫正，评分口径与 texture_w 无关）。"""
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_gray(path: Path) -> np.ndarray:
    """照片 → 灰度 float，**原生分辨率不缩放**（imdecode 兼容中文路径）。

    2026-07-30 修正：本批与产线归档同一台线扫相机（高 1728/3000、宽可变），
    mm/px=1。此前长边缩至 2000 会让 78% 的样片被缩放且每张尺度不同，破坏标准要求的
    "GLCM 步距 d=1 ≡ 真实 1 mm"。故不再降采样。
    """
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读图失败：{path}")
    return np.asarray(img, dtype=float)


def _avg_ranks(v) -> np.ndarray:
    """并列取**平均秩**（scipy rankdata 'average' 口径，纯 numpy）。"""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    pos = np.empty(v.size, dtype=float)
    pos[order] = np.arange(v.size, dtype=float)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=pos) / cnt)[inv]


def _spearman(x, y) -> float:
    """Spearman 秩相关，**并列取平均秩**（2026-08-04 更正）。

    原 argsort 名次法对并列值按枚举顺序强行分先后；人工分 5 分一档、99.1%
    样本在并列组内，据其报出的形式对照 vs_man 三值（-0.888/-0.930/-0.877）
    已作废——本文件所有秩相关（含偏相关、bootstrap）一律并列修正口径。
    """
    return float(np.corrcoef(_avg_ranks(x), _avg_ranks(y))[0, 1])


def _form_comparison(recs: list[dict]) -> dict:
    """形式对照表 + 配对 bootstrap 显著性（全部并列平均秩口径，运行即复算）。

    配对 bootstrap（B=2000，固定种子）：对样片重采样，Δ|ρ| = |ρ_A|−|ρ_B| 的
    2.5/97.5 百分位为 CI95。注意 Δ 的方差取决于两估计量的相关性——dual 与
    cpa_only 嵌套（前者含后者）故 CI 窄；ca 与 cpa 相关性低故 CI 宽，
    点差更大也可能不显著，这不是矛盾。
    """
    man = np.array([r["man"] for r in recs], float)
    sysv = np.array([r["sys"] for r in recs], float)
    forms = {
        "dual_bound_combined": np.array([r["texture_w"] for r in recs], float),
        "ca_only": np.array([r["ca_w"] for r in recs], float),
        "cpa_only": np.array([r["cpa_w"] for r in recs], float),
        "cpa_unweighted_no_position": np.array([r["cpa_unw"] for r in recs], float),
    }
    out: dict = {
        "note": "同一批片、**原生分辨率**同一预处理，仅换特征组合形式；"
                "ρ 为 Spearman(指标, 人工分/系统分)，**并列取平均秩**，"
                "负号=方向正确（指标越大越差）。"
                "⚠️ 见 saturation_entanglement：本批排序受 8-bit 裁剪纠缠，属临时结论",
        "preprocess": "native",
        "rank_method": "average(ties)",
    }
    for k, v in forms.items():
        out[k] = {"vs_man": round(_spearman(v, man), 4),
                  "vs_sys": round(_spearman(v, sysv), 4)}
    out["dual_bound_combined"]["note"] = "本实现采用的形式"
    out["cpa_unweighted_no_position"]["note"] = "去掉位置加权后的判别力落差=位置加权的贡献"

    n = len(recs)
    rng = np.random.default_rng(20260730)
    B = 2000

    def _boot(va, vb):
        """Δ|ρ|(va,man)−(vb,man) 的点估计与 CI95。"""
        pt = abs(_spearman(va, man)) - abs(_spearman(vb, man))
        d = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, n)
            d[b] = abs(_spearman(va[idx], man[idx])) - abs(_spearman(vb[idx], man[idx]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta_abs_rho": round(float(pt), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "significant": bool(lo > 0 or hi < 0), "B": B}

    out["significance_vs_cpa_only"] = _boot(forms["dual_bound_combined"], forms["cpa_only"])
    out["significance_ca_vs_cpa_only"] = _boot(forms["ca_only"], forms["cpa_only"])
    out["ca_only_caveat"] = (
        "仅Ca 的点估计最高，但其与仅 CPa 的差异经配对 bootstrap 不显著"
        "（见 significance_ca_vs_cpa_only），且在**无裁剪的合成条纹图**上会把真实"
        "重斑片判得比完美片更好（顺序反转）——原生 1mm 步距远低于应力斑的"
        "几十毫米尺度，Ca 在该尺度上量的是噪声与裁剪边缘，故不予采用")
    return out


def _synthetic_properties() -> dict:
    """合成性质复核（同一实现口径）：均匀退化 / 方向性 / iid 噪声钝感。"""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:300, 0:400].astype(float)
    # 均匀退化：iid 噪声场，加权 vs 未加权 CP 指数
    homog = rng.normal(100.0, 20.0, size=(300, 400))
    v_w = metrics.texture_w(homog, mm_per_px=1.0).value
    ca_p, cpa_p = metrics._glcm_ca_cpa(homog, ng=8)
    v_plain = 0.5 * ((ca_p / metrics._ca_sup(8)) ** 0.5
                     + (cpa_p / metrics._cp_sup(8)) ** 0.25)
    # 方向性：同幅度斑放中心 vs 放角落
    def blob(cy, cx):
        """单高斯斑合成图：σ=30px、峰值 60、背景 100。"""
        return 100.0 + 60.0 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 30.0 ** 2)))
    v_center = metrics.texture_w(blob(150, 200), mm_per_px=1.0).value
    v_corner = metrics.texture_w(blob(30, 40), mm_per_px=1.0).value
    # iid 噪声钝感：近平坦+微噪
    flat = 100.0 + rng.normal(0.0, 0.5, size=(300, 400))
    v_noise = metrics.texture_w(flat, mm_per_px=1.0).value
    return {
        "degeneracy": {"v_weighted": round(v_w, 5), "v_plain": round(v_plain, 5),
                       "rel": round(abs(v_w - v_plain) / v_plain, 6)},
        "direction": {"center": round(v_center, 4), "corner": round(v_corner, 4),
                      "center_over_corner": round(v_center / max(v_corner, 1e-12), 3)},
        "iid_noise": round(v_noise, 4),
    }


def main() -> int:
    """跑 115 片 + 合成性质 → 落盘 values.json 并打印摘要。"""
    if not PHOTO_DIR.exists() or not any(PHOTO_DIR.glob("*.png")):
        print(f"[跳过] 未找到标注照片：{PHOTO_DIR}\\*.png（真实图不入库，需本地放置）")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()

    recs, failures = [], []
    for p in sorted(PHOTO_DIR.glob("*.png")):
        m = re.match(r"(\d+)_系统([\d.]+)_人工([\d.]+)", p.stem)
        if not m:
            continue
        no = int(m.group(1))
        try:
            res = score_sheets(_load_gray(p), cfg)
        except ValueError as e:
            failures.append({"no": no, "error": str(e)})
            continue
        warped = np.asarray(res.sheet_results[0].warped_image, dtype=float)  # 主片
        r = metrics.texture_w(warped, mm_per_px=1.0)
        # 未加权对照（形式对照表第 4 行）：同一矫正图、同一 min/max 量化，仅去位置权重
        ca_u, cpa_u = metrics._glcm_ca_cpa(warped)
        recs.append({"no": no, "sys": float(m.group(2)), "man": float(m.group(3)),
                     "texture_w": round(float(r.value), 5),
                     "ca_w": round(float(r.ca_w), 4),
                     "cpa_w": round(float(r.cpa_w), 2),
                     "ca_unw": round(float(ca_u), 4),
                     "cpa_unw": round(float(cpa_u), 2),
                     # 饱和占比：8-bit 采集的高光裁剪程度（非仿射污染的强度探针）
                     "sat_frac": round(float(np.mean(warped >= 250)), 5),
                     "degenerate": bool(r.degenerate)})

    if not recs:
        print("[跳过] 无可解析记录")
        return 0

    man = np.array([r["man"] for r in recs])
    sysv = np.array([r["sys"] for r in recs])
    w = np.array([r["texture_w"] for r in recs])
    assert float(w.max()) < 1.0, "理论上界失守：出现 W ≥ 1"

    sp_man, sp_sys = _spearman(w, man), _spearman(w, sysv)
    bands = []
    for b in sorted(set(man)):
        sel = man == b
        bands.append({"man": b, "n": int(sel.sum()),
                      "texture_w_mean": round(float(w[sel].mean()), 4)})
    means = [x["texture_w_mean"] for x in bands]
    inversions = sum(1 for i in range(len(means) - 1) if means[i] < means[i + 1])

    # 饱和裁剪纠缠诊断（2026-07-30）：本批为 8-bit 强度采集，重斑片高光被削平。
    # 该裁剪是非仿射污染，且与两个纹理分量高度共线——故所有相关性结论都须声明
    # "受裁剪纠缠"，在 nm 标定（无裁剪）数据上必须重新确立形式排序。
    def _partial(x, y, z) -> float:
        """秩域偏相关：x、y 各对 z 回归后取残差相关。"""
        R = _avg_ranks  # 并列取平均秩（与 _spearman 同口径）
        rx, ry, rz = R(x), R(y), R(z)
        Z = np.column_stack([rz, np.ones_like(rz)])
        ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
        return float(np.corrcoef(ex, ey)[0, 1])

    sat = np.array([r["sat_frac"] for r in recs])
    ca_arr = np.array([r["ca_w"] for r in recs])
    cpa_arr = np.array([r["cpa_w"] for r in recs])
    saturation = {
        "sat_frac_median": round(float(np.median(sat)), 5),
        "sat_frac_max": round(float(sat.max()), 5),
        "spearman_ca_vs_sat": round(_spearman(ca_arr, sat), 3),
        "spearman_cpa_vs_sat": round(_spearman(cpa_arr, sat), 3),
        "spearman_man_vs_sat": round(_spearman(man, sat), 3),
        "partial_ca_vs_man_given_sat": round(_partial(ca_arr, man, sat), 3),
        "partial_cpa_vs_man_given_sat": round(_partial(cpa_arr, man, sat), 3),
        "partial_w_vs_man_given_sat": round(_partial(w, man, sat), 3),
        "note": "两个纹理分量与饱和占比高度共线；人工分本身也与饱和强相关"
                "（越饱和判越差）。故本批的判别力与形式排序均受 8-bit 裁剪纠缠，"
                "属临时结论——须在 nm 标定（无裁剪）样本集上重新确立",
    }

    values = {
        "n_photos": len(list(PHOTO_DIR.glob("*.png"))), "n_scored": len(recs),
        "failures": failures,
        "saturation_entanglement": saturation,
        "spearman_texture_w_vs_man": round(sp_man, 3),
        "spearman_texture_w_vs_sys": round(sp_sys, 3),
        # 形式对照：本次运行**真复算**（2026-08-04 起；此前为 2026-07-30 argsort
        # 口径的写死值，已作废）。Spearman 对严格单调变换不变 → 各单分量形式直接
        # 用其原始量（ca_w/cpa_w/cpa_unw）计算，无需再套 √ 或 ⁴√。
        "form_comparison": _form_comparison(recs),
        "w_min": round(float(w.min()), 5), "w_max": round(float(w.max()), 5),
        "n_degenerate": int(sum(r["degenerate"] for r in recs)),
        "bands": bands, "band_inversions": inversions,
        "synthetic": _synthetic_properties(),
        "all": recs,
        "preprocess": "native",
        "_口径说明": "原生分辨率不缩放（mm/px=1 → d=1≡1mm 标准口径）；幅度为 8-bit 强度域（gray→nm 未标定，W_w 仿射不变故不依赖）；结论限相关性与排序层面",
    }
    (OUT / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"n={len(recs)}/{values['n_photos']}  failures={len(failures)}")
    fc = values["form_comparison"]
    print(f"Spearman(W_w, 人工分) = {sp_man:+.3f}   （仅CPa {fc['cpa_only']['vs_man']:+.4f}）")
    print(f"Spearman(W_w, 系统分) = {sp_sys:+.3f}   （仅CPa {fc['cpa_only']['vs_sys']:+.4f}）")
    s1, s2 = fc["significance_vs_cpa_only"], fc["significance_ca_vs_cpa_only"]
    print(f"双分量 vs 仅CPa: Δ|ρ|={s1['delta_abs_rho']:+.4f} CI{s1['ci95']} 显著={s1['significant']}")
    print(f"仅Ca  vs 仅CPa: Δ|ρ|={s2['delta_abs_rho']:+.4f} CI{s2['ci95']} 显著={s2['significant']}")
    print(f"W ∈ [{values['w_min']}, {values['w_max']}]  全体 <1 ✓   degenerate={values['n_degenerate']}")
    print(f"分档数 {len(bands)}，倒挂 {inversions} 处；分档均值 {means}")
    s = values["synthetic"]
    print(f"合成复核: 退化 rel={s['degeneracy']['rel']:.4%}  "
          f"方向性 中心/角落={s['direction']['center_over_corner']}  "
          f"iid噪声 W={s['iid_noise']}")
    sa = values["saturation_entanglement"]
    print(f"饱和裁缠: 占比中位 {sa['sat_frac_median']:.2%} 最大 {sa['sat_frac_max']:.2%}；"
          f"Ca~饱和 {sa['spearman_ca_vs_sat']:+.3f} CPa~饱和 {sa['spearman_cpa_vs_sat']:+.3f} "
          f"人工~饱和 {sa['spearman_man_vs_sat']:+.3f}")
    print(f"  控制饱和后偏相关: Ca {sa['partial_ca_vs_man_given_sat']:+.3f}  "
          f"CPa {sa['partial_cpa_vs_man_given_sat']:+.3f}  W {sa['partial_w_vs_man_given_sat']:+.3f}"
          f"  ← 结论受裁剪纠缠，须在 nm 标定数据上重定")
    print(f"资产 → {OUT / 'values.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""《应力斑均匀度指标_技术说明》实测算例资产生成器（开发壳，不随核心包交付）。

跑真实整床应力斑照片（data/images/stress_fringe/*.PNG）→ 逐片六指标 →
挑选对比样片（均匀度高/低、CCP 高/低）+ CCP 位置盲区构造演示（同一样片
图像整体循环平移：相邻灰度对统计几乎不变 → CCP 基本不动；斑位置改变 →
均匀度显著变化）。产出：
- data/derived/uniformity_doc/*.png　　嵌入 PDF 的样片灰度图
- data/derived/uniformity_doc/values.json　实测数值（PDF 生成器注入正文，图文同源）
算法调用 D:\\GlassApp\\fringe_scoring 权威实现 + app_config.yaml 生效配置。
用法：venv python fringe_scoring/make_uniformity_doc_assets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\GlassApp")

import cv2
import numpy as np
import yaml

from fringe_scoring.indicators import compute_sheet_indicators
from fringe_scoring.score import compute_pipeline, score_from_pipeline
from fringe_scoring.sheets import score_sheets

ROOT = Path(__file__).resolve().parents[1]
PHOTO_DIR = ROOT / "data" / "images" / "stress_fringe"
OUT = ROOT / "data" / "derived" / "uniformity_doc"
MAX_SIDE = 2000  # 示例处理分辨率（长边 px），与文中说明一致

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
    pipe = compute_pipeline(arr, cfg2)
    res = score_from_pipeline(pipe, cfg2)
    return compute_sheet_indicators(arr, ~res.border_mask_arr, cfg2, pipeline=pipe)


def _rec(ind, photo: str, sheet_no: int, bed_no: int) -> dict:
    """一片的记录（文档要用的量，四舍五入到展示位数）。"""
    return {
        "photo": photo, "sheet": sheet_no, "bed": bed_no,
        "U": round(ind.uniformity, 1),
        "ccp": round(ind.ccp_value, 3),
        "ca": round(ind.ccp_ca, 4),
        "cpa": round(ind.ccp_cpa, 1),
        "x095": round(ind.x095, 1),
    }


def main() -> int:
    """跑全部整床照 → 选例 → 落盘 PNG 与 values.json。"""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()
    photos = sorted(PHOTO_DIR.glob("*.PNG"))
    if not photos:
        raise ValueError(f"未找到整床照片：{PHOTO_DIR}")

    records, crops = [], []  # 全部片的指标记录与矫正图（跨照片统一编号）
    bed_rel = None
    for pi, photo in enumerate(photos):
        arr = _load_gray(photo)
        result = score_sheets(arr, cfg)
        assert result.sheet_indicators is not None
        if pi == 0:  # 首张整床照缩略图（示例场景说明用）
            bed_rel = _save_gray(arr, "bed.png", max_w=1400)
        for si, (r, ind) in enumerate(zip(result.sheet_results, result.sheet_indicators)):
            records.append(_rec(ind, photo.name, si + 1, pi + 1))
            crops.append(np.asarray(r.warped_image, dtype=float))
        print(f"{photo.name}: {result.n_sheets} 片")

    by_u = sorted(range(len(records)), key=lambda i: records[i]["U"])
    by_ccp = sorted(range(len(records)), key=lambda i: records[i]["ccp"])
    picks = {
        "u_lo": by_u[0], "u_hi": by_u[-1],        # 均匀度最低/最高
        "ccp_hi": by_ccp[-1], "ccp_lo": by_ccp[0],  # CCP 最差/最好
    }
    values: dict = {"bed": bed_rel, "n_photos": len(photos), "n_sheets": len(records),
                    "all": records}  # 全批逐片记录（打分样例总表用）
    for key, idx in picks.items():
        values[key] = {**records[idx], "img": _save_gray(crops[idx], f"{key}.png")}

    # CCP 位置盲区构造演示：取均匀度最低片（斑最集中），先裁除四周边缘应力亮带
    # （无缝构造，见 _trim_border），再整体垂直循环平移四分之一板高——斑移离中心
    # 但不贴边（贴边会牵动边框带识别、间接改变 CCP 的评估区域，污染对比）；
    # 相邻灰度对多重集仅在拼接缝一行改变 → CCP 应基本不变；斑离开高权重区 → U 应上升
    base = _trim_border(crops[picks["u_lo"]], cfg)
    rolled = np.roll(base, base.shape[0] // 4, axis=0)
    ind_a, ind_b = _indicators_of(base, cfg), _indicators_of(rolled, cfg)
    values["roll"] = {
        "a": {"U": round(ind_a.uniformity, 1), "ccp": round(ind_a.ccp_value, 3),
              "ca": round(ind_a.ccp_ca, 4), "cpa": round(ind_a.ccp_cpa, 1),
              "img": _save_gray(base, "roll_a.png")},
        "b": {"U": round(ind_b.uniformity, 1), "ccp": round(ind_b.ccp_value, 3),
              "ca": round(ind_b.ccp_ca, 4), "cpa": round(ind_b.ccp_cpa, 1),
              "img": _save_gray(rolled, "roll_b.png")},
    }

    (OUT / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(values, ensure_ascii=False, indent=2))
    print(f"资产 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

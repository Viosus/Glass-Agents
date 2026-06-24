"""应力斑图像加载与引用：把 data/images/stress_fringe/ 下的图片接到 schemas.ImageRef。

职责（未被阻塞部分）：读文件 → ndarray + sha256 + 尺寸 → ImageRef。
⚠️ 像素值 → 标定光程差(nm) 的换算依赖扫描仪标定（TODO(plant)，见现场拍板「取数/扫描仪」）；
本模块不臆造 nm 标定。约定：.npy 存"已标定光程差(nm)"；位图(png/tif/jpg)读出的是像素强度，非 nm。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from schemas.archive import ImageRef

_REPO_ROOT = Path(__file__).resolve().parent.parent
STRESS_IMAGE_DIR = _REPO_ROOT / "data" / "images" / "stress_fringe"


def sha256_of_file(path: Path | str) -> str:
    """文件内容 sha256（与 ImageRef.sha256 对齐）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_array(path: Path | str) -> np.ndarray:
    """读图为二维 float ndarray。

    .npy → 约定存已标定光程差(nm)，直接返回；
    位图(png/tif/jpg…) → 转灰度返回像素强度（非 nm，标定 TODO(plant)）。
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.asarray(np.load(p), dtype=float)

    from skimage.color import rgb2gray
    from skimage.io import imread

    img = imread(p)
    if img.ndim == 3:
        img = rgb2gray(img[..., :3])
    return np.asarray(img, dtype=float)


def make_image_ref(path: Path | str, mm_per_px: float) -> ImageRef:
    """为一张图生成 ImageRef（路径+sha256+尺寸+尺度），与结构化数据分离引用。

    path 若在仓库内，则以仓库根的相对 posix 路径入库，便于跨机引用。
    """
    p = Path(path).resolve()
    arr = load_array(p)
    height_px, width_px = int(arr.shape[0]), int(arr.shape[1])
    try:
        rel = p.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        rel = str(p)
    return ImageRef(
        path=rel,
        sha256=sha256_of_file(p),
        width_px=width_px,
        height_px=height_px,
        mm_per_px=mm_per_px,
    )

"""schemas/image_io.py 单测：加载、哈希、生成 ImageRef。"""

import numpy as np

from schemas.archive import ImageRef
from schemas.image_io import load_array, make_image_ref, sha256_of_file


def test_sha256_and_load_npy(tmp_path):
    arr = np.arange(12, dtype=float).reshape(3, 4)
    p = tmp_path / "ret.npy"
    np.save(p, arr)
    loaded = load_array(p)
    assert loaded.shape == (3, 4)
    assert np.allclose(loaded, arr)
    assert len(sha256_of_file(p)) == 64


def test_make_image_ref_from_npy(tmp_path):
    arr = np.zeros((10, 20), dtype=float)   # rows=height=10, cols=width=20
    p = tmp_path / "img.npy"
    np.save(p, arr)
    ref = make_image_ref(p, mm_per_px=0.5)
    assert isinstance(ref, ImageRef)
    assert ref.width_px == 20 and ref.height_px == 10
    assert ref.mm_per_px == 0.5
    assert len(ref.sha256) == 64

# 应力斑图像 · stress_fringe

存放应力斑 / 光程差图像。**图像本体放这里；元数据/引用进结构化数据层**（见下「引用/链接」）。

## 命名与格式约定
- 文件名：`<sample_id>.<ext>`，与 [schemas/archive.py](../../../schemas/archive.py) 的 `ArchiveSample.sample_id` 对应。
- 格式：
  - `.npy` —— 约定存**已标定光程差(nm)** 的二维数组（可直接喂 `tools/metrics`）。
  - `.png / .tif / .jpg` —— 位图（像素强度，**非 nm**）。像素→nm 标定依赖扫描仪，待现场标定（`TODO(plant)`，见现场拍板「取数/扫描仪」）。
- 单位：尺度 `mm_per_px`（随图记录，进 `ImageRef`）；光程差 `nm`。

## 引用 / 链接到代码
- **引用模型**：[schemas/archive.py](../../../schemas/archive.py) 的 `ImageRef`（路径 + sha256 + 尺寸 + `mm_per_px`），与结构化数据分离。
- **加载/生成引用**：[schemas/image_io.py](../../../schemas/image_io.py)
  - `load_array(path)` → 二维 float ndarray
  - `make_image_ref(path, mm_per_px)` → `ImageRef`
  - `sha256_of_file(path)` → 内容哈希
- **指标计算**：[tools/metrics.py](../../../tools/metrics.py)（X0.95 / IsoT / CCP，仅在掩膜 M 内统计）。

## 用法（把图放进来后）
```python
from schemas.image_io import make_image_ref
ref = make_image_ref("data/images/stress_fringe/0001.npy", mm_per_px=0.1)
# ref 挂到 ArchiveSample.stress_image；图像数组喂 tools.metrics（需为已标定 nm）
```

## 注意
- 大量/大图建议后续启用 git-lfs；当前默认随仓库跟踪。
- 位图像素**不是** nm；换算标定未到位前不要当真值喂判级（规则 > AI）。

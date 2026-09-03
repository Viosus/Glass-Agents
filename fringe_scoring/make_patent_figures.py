"""专利交底书附图生成器（开发壳，不随核心包交付）。

为《专利交底书_钢化炉逆向诊断_草稿.md》生成 6 张说明书附图（正式黑白线框，
matplotlib，SimHei，dpi=300）：
  图1 系统整体框图          图2 方法总流程图（摘要附图）
  图3 多头预测模型结构      图4 三态诊断判定流程
  图5 安全校验交互时序      图6 数据闭环流程

口径（2026-08-19 改稿）：应力斑指标计算属外部检测环节，本申请只把缺陷指标当
输入数据——特征提取子流程图与掩膜/位置权重示意图已删除，模块框相应改名。
约定：纯黑白线框（无灰度渐变、无照片），区域区分用 hatch；模块框带附图标记
数字（10/20/…、S1/S2/…）与交底书正文引用一致；不读任何生产数据。
用法：venv python fringe_scoring/make_patent_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

OUT = ROOT / "data" / "derived" / "patent_figures"

try:  # Windows 控制台默认 GBK，强制 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

_LW = 1.2          # 统一线宽
_FS = 9.5          # 框内文字字号
_FS_NUM = 8.5      # 附图标记字号


# --------------------------------------------------------------------------- #
# 绘图基元
# --------------------------------------------------------------------------- #
def _new_ax(width_in: float, height_in: float, xlim, ylim):
    """新建无坐标轴画布（附图统一入口）。"""
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def _box(ax, cx, cy, w, h, text, num=None, dashed=False, fs=_FS):
    """矩形框（中心定位）+ 居中多行文字 + 可选附图标记（框右上角外侧）。"""
    ax.add_patch(Rectangle(
        (cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor="black",
        linewidth=_LW, linestyle="--" if dashed else "-",
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="black")
    if num is not None:
        ax.text(cx + w / 2 + 0.06, cy + h / 2, str(num),
                ha="left", va="top", fontsize=_FS_NUM, color="black")


def _diamond(ax, cx, cy, w, h, text, fs=_FS):
    """菱形判定框（中心定位）。"""
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor="black", linewidth=_LW))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="black")


def _arrow(ax, x1, y1, x2, y2, label=None, label_dx=0.08, label_dy=0.0,
           dashed=False, fs=_FS):
    """实线/虚线箭头 + 可选标签（标签放在箭头中点偏移处）。"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(
        arrowstyle="-|>", linewidth=_LW, color="black",
        linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0,
    ))
    if label is not None:
        ax.text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy, label,
                ha="left", va="center", fontsize=fs, color="black")


def _save(fig, name: str) -> Path:
    """保存 PNG（dpi=300，白底，紧边界）并返回落盘路径。"""
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 图 1：系统整体框图
# --------------------------------------------------------------------------- #
def fig1_system() -> Path:
    """图 1：系统整体框图（缺陷指标为外部检测环节输入，主链五模块 + 闭环侧路）。"""
    fig, ax = _new_ax(6.3, 7.2, (0, 10), (0, 13.2))

    # 外部环节（虚线，不在保护范围内）
    _box(ax, 3.4, 12.4, 4.4, 1.1, "应力检测装置及检测软件\n（应力斑图像 → 缺陷指标）", dashed=True, fs=8.8)
    _box(ax, 8.1, 12.4, 3.4, 1.1, "炉控系统 / 生产执行系统\n（工艺参数、工况数据）", dashed=True, fs=8.8)

    # 主链五模块（右移留出闭环侧路）
    cx, w = 5.7, 6.0
    _box(ax, cx, 10.7, w, 0.95, "数据输入接口模块\n（缺陷指标 / 工艺参数 / 基准配方 / 工况）", num=10, fs=8.8)
    _box(ax, cx, 8.95, w, 1.15,
         "特征向量构造模块\n（固定契约·缺失项填 0 并附存在性标志位）", num=20, fs=8.6)
    _box(ax, cx, 7.1, w, 1.15,
         "多头预测模块\n（共享编码器＋参数残差/质量/能耗/归因四头）", num=30, fs=8.6)
    _box(ax, cx, 5.25, w, 1.15,
         "状态判定模块\n（质量残差×补偿形态联合判据→三态）", num=40, fs=8.6)
    _box(ax, cx, 3.4, w, 1.15,
         "安全校验与输出模块\n（硬约束校验；不通过则拒绝并返回违规项）", num=50, fs=8.6)

    # 数据闭环（左侧侧路）
    _box(ax, 1.25, 5.25, 2.3, 2.0,
         "数据闭环模块\n（标注回灌·\n数据门·评测门·\n版本回滚）", num=60, fs=8.6)

    # 输出（外部，虚线）
    _box(ax, cx, 1.35, w, 1.0, "操作终端\n（调参建议 / 维保检查清单 / 违规项提示）", dashed=True, fs=8.6)

    _arrow(ax, 3.4, 11.85, 4.7, 11.18)
    _arrow(ax, 8.1, 11.85, 6.7, 11.18)
    for y1, y2 in ((10.22, 9.53), (8.37, 7.68), (6.52, 5.83), (4.67, 3.98), (2.82, 1.85)):
        _arrow(ax, cx, y1, cx, y2)
    # 闭环：50 左边 → 60 底部；60 顶部 → 30 左边
    ax.plot([2.7, 1.25], [3.4, 3.4], color="black", linewidth=_LW)
    _arrow(ax, 1.25, 3.4, 1.25, 4.25)
    ax.plot([1.25, 1.25], [6.25, 7.1], color="black", linewidth=_LW)
    _arrow(ax, 1.25, 7.1, 2.7, 7.1)

    ax.text(5.0, 0.35, "图 1  系统整体框图", ha="center", va="center", fontsize=10.5)
    return _save(fig, "fig1_system.png")


# --------------------------------------------------------------------------- #
# 图 2：方法总流程图（摘要附图）
# --------------------------------------------------------------------------- #
def fig2_flow() -> Path:
    """图 2：方法总流程图（S1–S5 + 校验分支；摘要附图）。"""
    fig, ax = _new_ax(6.3, 7.6, (0, 10), (0, 14.2))

    _box(ax, 5.0, 13.3, 7.2, 0.95, "S1  获取缺陷特征参数集与当前工艺参数、基准配方\n（缺陷指标由外部应力检测环节测得）")
    _box(ax, 5.0, 11.6, 7.2, 0.95, "S2  构造固定契约的输入特征向量\n（缺失项填 0 并附存在性标志位）")
    _box(ax, 5.0, 9.9, 7.2, 0.95, "S3  特征向量输入多头预测模型\n（输出参数残差、质量预测值、归因指向）")
    _box(ax, 5.0, 8.2, 7.2, 0.95, "S4  质量残差 × 参数补偿形态联合判据\n（判定：健康 / 带病已补偿 / 带病未补偿）")
    _box(ax, 5.0, 6.5, 7.2, 0.95, "S5  按判定状态与参数残差生成调参建议\n（建议参数送硬约束安全校验）")
    _diamond(ax, 5.0, 4.35, 3.4, 1.7, "校验\n是否通过？")
    _box(ax, 2.45, 1.9, 4.1, 1.1, "输出调参建议\n（附预测质量与根因候选排序）")
    _box(ax, 7.55, 1.9, 4.1, 1.1, "拒绝下发\n（原样返回违规项列表）")

    _arrow(ax, 5.0, 12.82, 5.0, 12.08)
    _arrow(ax, 5.0, 11.12, 5.0, 10.38)
    _arrow(ax, 5.0, 9.42, 5.0, 8.68)
    _arrow(ax, 5.0, 7.72, 5.0, 6.98)
    _arrow(ax, 5.0, 6.02, 5.0, 5.2)
    _arrow(ax, 3.3, 4.35, 2.45, 4.35)
    _arrow(ax, 2.45, 4.35, 2.45, 2.45)
    ax.text(2.6, 3.5, "是", fontsize=_FS)
    _arrow(ax, 6.7, 4.35, 7.55, 4.35)
    _arrow(ax, 7.55, 4.35, 7.55, 2.45)
    ax.text(7.7, 3.5, "否", fontsize=_FS)

    ax.text(5.0, 0.75, "图 2  方法总流程图", ha="center", va="center", fontsize=10.5)
    return _save(fig, "fig2_flow.png")


# --------------------------------------------------------------------------- #
# 图 3：多头预测模型结构
# --------------------------------------------------------------------------- #
def fig3_model() -> Path:
    """图 3：多头预测模型结构（共享编码器 + 四头 + 残差叠加）。"""
    fig, ax = _new_ax(6.3, 5.6, (0, 10), (0, 10.4))

    _box(ax, 5.0, 9.6, 7.4, 0.9, "输入特征向量（26 维：缺陷指标＋基准配方＋规格＋工况，\n缺失项填 0 并附存在性标志位）", fs=9)
    _box(ax, 5.0, 8.0, 5.2, 0.75, "全连接层（64 维）＋ ReLU")
    _box(ax, 5.0, 6.85, 5.2, 0.75, "全连接层（64 维）＋ ReLU")
    ax.text(7.85, 7.42, "共享编码器", ha="left", va="center", fontsize=9)
    ax.add_patch(Rectangle((2.1, 6.35), 5.8, 2.1, fill=False, edgecolor="black",
                           linewidth=0.9, linestyle="--"))

    heads = [
        ("参数残差头\nΔ（6 维）", 1.55), ("质量预测头\n（1 维）", 3.85),
        ("能耗预测头\n（1 维）", 6.15), ("归因头\n（4 维）", 8.45),
    ]
    for text, cx in heads:
        _box(ax, cx, 4.6, 2.05, 1.05, text, fs=9)
        _arrow(ax, 5.0, 6.35, cx, 5.13)

    _box(ax, 1.55, 2.5, 2.9, 1.05, "建议参数 =\n基准配方 + Δ", fs=9)
    _box(ax, 6.15, 2.5, 5.6, 1.05, "状态判定模块（质量残差 / 归因指向）", fs=9)

    _arrow(ax, 5.0, 9.15, 5.0, 8.38)
    _arrow(ax, 5.0, 7.63, 5.0, 7.23)
    _arrow(ax, 1.55, 4.08, 1.55, 3.03)
    _arrow(ax, 3.85, 4.08, 4.6, 3.03)
    _arrow(ax, 6.15, 4.08, 6.15, 3.03)
    _arrow(ax, 8.45, 4.08, 7.7, 3.03)
    _arrow(ax, 1.55, 1.98, 1.55, 1.15)
    ax.text(1.7, 1.55, "送安全校验", fontsize=8.5, ha="left")

    ax.text(5.0, 0.5, "图 3  多头预测模型结构", ha="center", va="center", fontsize=10.5)
    return _save(fig, "fig3_model.png")


# --------------------------------------------------------------------------- #
# 图 4：三态诊断判定流程
# --------------------------------------------------------------------------- #
def fig4_tristate() -> Path:
    """图 4：三态诊断判定流程（质量残差 × 补偿签名二级判定）。"""
    fig, ax = _new_ax(6.3, 5.2, (0, 10), (0, 9.6))

    _box(ax, 5.0, 8.9, 7.6, 0.85, "输入：质量残差（实测质量 - 预测质量）＋ 参数补偿形态")
    _diamond(ax, 5.0, 7.0, 3.9, 1.6, "质量残差\n是否正常？")
    _diamond(ax, 3.1, 4.35, 3.9, 1.6, "是否存在\n补偿签名？")
    _box(ax, 8.15, 4.35, 3.1, 1.15, "态③ 带病未补偿\n（异常告警＋根因\n候选排序）", fs=9)
    _box(ax, 1.55, 1.55, 2.8, 1.05, "态① 健康", fs=9.5)
    _box(ax, 5.35, 1.55, 3.4, 1.2, "态② 带病已补偿\n（不报故障；列入下次\n周保养优先检查清单）", fs=8.5)

    _arrow(ax, 5.0, 8.47, 5.0, 7.8)
    _arrow(ax, 6.95, 7.0, 8.15, 7.0)
    _arrow(ax, 8.15, 7.0, 8.15, 4.93)
    ax.text(8.3, 6.0, "否", fontsize=_FS)
    _arrow(ax, 5.0, 6.2, 3.1, 5.15)
    ax.text(3.75, 5.85, "是", fontsize=_FS)
    _arrow(ax, 1.15, 4.35, 1.15, 2.08)
    ax.text(0.85, 3.3, "否", fontsize=_FS)
    _arrow(ax, 3.1, 3.55, 4.3, 2.15)
    ax.text(3.85, 3.05, "是", fontsize=_FS)

    ax.text(5.0, 0.5, "图 4  三态诊断判定流程", ha="center", va="center", fontsize=10.5)
    return _save(fig, "fig4_tristate.png")


# --------------------------------------------------------------------------- #
# 图 5：安全校验交互时序
# --------------------------------------------------------------------------- #
def fig5_sequence() -> Path:
    """图 5：安全校验交互时序（四生命线 + 二选一分支）。"""
    fig, ax = _new_ax(6.3, 5.8, (0, 10), (0, 10.8))

    lanes = [("多头预测模块", 1.3), ("参数解码器", 3.8), ("安全校验器", 6.3), ("操作终端", 8.8)]
    for name, x in lanes:
        _box(ax, x, 10.1, 2.1, 0.75, name, fs=9)
        ax.plot([x, x], [9.7, 1.3], color="black", linewidth=0.9, linestyle="--")

    _arrow(ax, 1.3, 9.0, 3.8, 9.0, label="① 参数残差 Δ", label_dx=-1.9, label_dy=0.25, fs=8.5)
    _arrow(ax, 3.8, 8.0, 6.3, 8.0, label="② 完整参数集\n（基准配方＋Δ）", label_dx=-2.2, label_dy=0.4, fs=8.5)
    ax.text(6.85, 7.2, "③ 逐项校验：梯度温控 /\n厚度-时长 / 对流配比 /\n炸板风险（阈值缺失\n一律判不通过）",
            ha="left", va="center", fontsize=8)
    ax.add_patch(Rectangle((6.05, 6.35), 0.5, 1.7, fill=False, edgecolor="black", linewidth=_LW))

    # alt 分支框
    ax.add_patch(Rectangle((0.5, 1.7), 9.2, 4.2, fill=False, edgecolor="black",
                           linewidth=0.9, linestyle="--"))
    ax.text(0.65, 5.62, "〔二选一〕", ha="left", va="center", fontsize=8)
    _arrow(ax, 6.3, 4.9, 8.8, 4.9, label="④a 通过：调参建议＋预测质量", label_dx=-2.4, label_dy=0.25, fs=8.5)
    ax.plot([0.5, 9.7], [3.9, 3.9], color="black", linewidth=0.7, linestyle=":")
    _arrow(ax, 6.3, 3.1, 8.8, 3.1, label="④b 不通过：拒绝下发，\n原样返回违规项列表", label_dx=-2.4, label_dy=0.45, fs=8.5)
    ax.text(6.45, 2.3, "（不自动放宽、不自动修正）", ha="left", va="center", fontsize=8)

    ax.text(5.0, 0.6, "图 5  安全校验交互时序", ha="center", va="center", fontsize=10.5)
    return _save(fig, "fig5_sequence.png")


# --------------------------------------------------------------------------- #
# 图 6：数据闭环流程
# --------------------------------------------------------------------------- #
def fig6_loop() -> Path:
    """图 6：数据闭环流程（数据门 → 训练 → 评测门 → 发布/回滚 + 回灌）。"""
    fig, ax = _new_ax(6.3, 6.4, (0, 10), (0, 11.8))

    _box(ax, 5.0, 11.05, 7.0, 0.9, "生产反馈数据（标注表：配方、最终参数、\n判级、归因说明、出炉实测质量与能耗）")
    _box(ax, 5.0, 9.5, 7.0, 0.8, "导入校验（逐行模式校验，脏行拒收）")
    _box(ax, 5.0, 8.1, 7.0, 0.8, "样本库（按厚度 × 品类 × 质量模式分桶）")
    _diamond(ax, 5.0, 6.45, 3.6, 1.5, "样本量达到\n训练门槛？")
    _box(ax, 5.0, 4.55, 7.0, 0.8, "模型训练（时间有序切分，禁止随机打乱）")
    _diamond(ax, 5.0, 2.8, 4.6, 1.6, "评测门：拟合优度不回归\n且解码参数全部过校验？")
    _box(ax, 1.85, 0.85, 3.1, 0.9, "发布新版本\n（双重版本校验）", fs=9)
    _box(ax, 8.15, 0.85, 3.1, 0.9, "不发布，沿用\n当前版本（回滚）", fs=9)

    _arrow(ax, 5.0, 10.6, 5.0, 9.9)
    _arrow(ax, 5.0, 9.1, 5.0, 8.5)
    _arrow(ax, 5.0, 7.7, 5.0, 7.2)
    _arrow(ax, 5.0, 5.7, 5.0, 4.95)
    ax.text(5.2, 5.32, "是", fontsize=_FS)
    _arrow(ax, 5.0, 4.15, 5.0, 3.6)
    _arrow(ax, 2.7, 2.8, 1.85, 2.8)
    _arrow(ax, 1.85, 2.8, 1.85, 1.3)
    ax.text(2.0, 2.1, "是", fontsize=_FS)
    _arrow(ax, 7.3, 2.8, 8.15, 2.8)
    _arrow(ax, 8.15, 2.8, 8.15, 1.3)
    ax.text(8.3, 2.1, "否", fontsize=_FS)
    # 等待更多样本：门 否 → 返回样本库
    _arrow(ax, 6.8, 6.45, 9.3, 6.45)
    ax.plot([9.3, 9.3], [6.45, 8.1], color="black", linewidth=_LW)
    _arrow(ax, 9.3, 8.1, 8.5, 8.1)
    ax.text(7.0, 6.7, "否（继续积累）", fontsize=8.5)
    # 部署回灌：发布 → 顶部
    ax.plot([0.3, 0.1], [0.85, 0.85], color="black", linewidth=_LW)
    ax.plot([0.1, 0.1], [0.85, 11.05], color="black", linewidth=_LW)
    _arrow(ax, 0.1, 11.05, 1.5, 11.05)
    ax.text(0.32, 6.4, "部署推理，持续回灌", fontsize=8.5, rotation=90,
            ha="center", va="center")

    ax.text(5.0, 0.1, "图 6  数据闭环流程", ha="center", va="center", fontsize=10.5)
    return _save(fig, "fig6_loop.png")


def main() -> None:
    """全量生成 6 张附图并打印落盘清单。"""
    OUT.mkdir(parents=True, exist_ok=True)
    makers = [fig1_system, fig2_flow, fig3_model, fig4_tristate,
              fig5_sequence, fig6_loop]
    print("专利附图生成：")
    for fn in makers:
        path = fn()
        print(f"  {path.relative_to(ROOT)}")
    print(f"共 {len(makers)} 张，输出目录 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""传输适配器：包怎么"送出去/收回来"。本轮只有文件投递（共享目录/U盘），云上传留接口。

铁律说明：本文件**不 import 任何联网库**——云端服务商/地址/鉴权方式均未定
（config/sync.yaml 全 TODO(plant)，列入 docs/信息需求清单.md 板块 C）。
拿到云端信息并通过安全评审后，另开 CloudTransport 适配器并同步调整
config/review_rules.yaml 联网黑名单；在那之前 make_transport("cloud") 明确报未实现。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

import yaml

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "sync.yaml"


def load_sync_config(path: Path | None = None) -> dict:
    """读 config/sync.yaml（每次调用按需读取；文件不存在返回空 dict）。"""
    p = Path(path) if path is not None else _CONFIG
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_todo(value: object) -> bool:
    """yaml 值是否为 TODO(plant) 占位。"""
    return isinstance(value, str) and value.strip().startswith("TODO(plant)")


class Transport(Protocol):
    """传输接口：push 送包出去，pull 把新包收回本地。"""

    def push(self, pack: Path) -> str:
        """送出一个包，返回目的地描述（供日志/审计）。"""
        ...

    def pull(self, dest_dir: Path) -> list[Path]:
        """拉取新包到 dest_dir（跳过已存在的），返回新收到的包路径。"""
        ...


class FileDropTransport:
    """文件投递：共享目录 / U盘挂载点。不联网也能跑通整条同步链路。"""

    def __init__(self, drop_dir: Path) -> None:
        """drop_dir = 双方约定的投递目录（内网共享盘或 U 盘路径）。"""
        self.drop_dir = Path(drop_dir)

    @classmethod
    def from_config(cls, config: dict | None = None) -> FileDropTransport:
        """从 config/sync.yaml 的 drop_dir 构造；未填（TODO(plant)）则明确报错，不猜路径。"""
        cfg = config if config is not None else load_sync_config()
        drop = cfg.get("drop_dir")
        if drop is None or _is_todo(drop):
            raise ValueError("config/sync.yaml 的 drop_dir 未填（TODO(plant)），无法构造文件投递通道")
        return cls(Path(str(drop)))

    def push(self, pack: Path) -> str:
        """复制包到投递目录。"""
        self.drop_dir.mkdir(parents=True, exist_ok=True)
        dest = self.drop_dir / Path(pack).name
        shutil.copy2(pack, dest)
        return str(dest)

    def pull(self, dest_dir: Path) -> list[Path]:
        """把投递目录里的 zip 包收进 dest_dir（同名已存在则跳过）。"""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        received: list[Path] = []
        if not self.drop_dir.exists():
            return received
        for src in sorted(self.drop_dir.glob("*.zip")):
            target = dest_dir / src.name
            if target.exists():
                continue
            shutil.copy2(src, target)
            received.append(target)
        return received


def make_transport(kind: str, config: dict | None = None) -> Transport:
    """按类型构造传输通道。cloud 未实现（云端信息 TODO(plant)，评审后另补）。"""
    if kind == "file_drop":
        return FileDropTransport.from_config(config)
    if kind == "cloud":
        raise NotImplementedError(
            "云上传通道未实现：云端服务商/地址/鉴权待定（config/sync.yaml TODO(plant)，"
            "见 docs/信息需求清单.md 板块 C）；拿到信息并过安全评审后再补"
        )
    raise ValueError(f"未知传输类型: {kind}（支持 file_drop；cloud 待评审）")

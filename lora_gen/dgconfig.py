"""加载 config/dataset_gen.yaml。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "dataset_gen.yaml"


@dataclass
class DGConfig:
    raw: dict[str, Any]
    config_hash: str

    @property
    def target_size(self) -> int:
        return int(self.raw["target_size"])

    @property
    def backend(self) -> str:
        return self.raw["backend"]

    @property
    def judge_backend(self) -> str:
        return self.raw["judge_backend"]

    @property
    def answerability(self) -> dict:
        return self.raw["answerability"]

    @property
    def quality(self) -> dict:
        return self.raw["quality"]

    @property
    def chunks(self) -> dict:
        return self.raw["chunks"]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_dg_config(path: str | Path | None = None, *, target_override: int | None = None) -> DGConfig:
    p = Path(path) if path else DEFAULT_PATH
    text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if target_override is not None:
        raw["target_size"] = target_override
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return DGConfig(raw=raw, config_hash=digest)

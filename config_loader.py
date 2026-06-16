"""配置加载与路径解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent


@dataclass
class AppConfig:
    raw: dict[str, Any]
    root: Path = ROOT

    @property
    def vehicle_model(self) -> str:
        return self.raw["vehicle"]["model"]

    @property
    def doc_types(self) -> list[str]:
        return self.raw["vehicle"]["doc_types"]

    @property
    def chunk_strategy(self) -> str:
        return self.raw["chunking"]["strategy"]

    @property
    def pdf_dir(self) -> Path:
        return self.root / self.raw["paths"]["pdf_dir"].lstrip("./")

    @property
    def index_dir(self) -> Path:
        return self.root / self.raw["paths"]["index_dir"].lstrip("./")

    @property
    def log_dir(self) -> Path:
        return self.root / self.raw["paths"]["log_dir"].lstrip("./")

    @property
    def rewrite_enabled(self) -> bool:
        return bool(self.get("query_rewrite", "enabled", default=False))

    @property
    def keyword_search_enabled(self) -> bool:
        return bool(self.get("retrieval", "keyword_search", "enabled", default=True))

    @property
    def bookmark_match_enabled(self) -> bool:
        return bool(self.get("retrieval", "bookmark_match", "enabled", default=False))

    @property
    def bookmark_max_matches(self) -> int:
        return int(self.get("retrieval", "bookmark_match", "max_matches", default=3))

    @property
    def verification_enabled(self) -> bool:
        return bool(self.get("verification", "enabled", default=False))

    def index_path(self, strategy: str | None = None) -> Path:
        s = strategy or self.chunk_strategy
        return self.index_dir / s / self.vehicle_model

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load_config(path: str | Path | None = None, strategy_override: str | None = None) -> AppConfig:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if strategy_override:
        raw["chunking"]["strategy"] = strategy_override
    return AppConfig(raw=raw)

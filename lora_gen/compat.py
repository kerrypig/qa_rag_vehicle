"""config 接口探测：确认所需接口存在，缺失则用 fallback adapter 补齐。"""
from __future__ import annotations

from pathlib import Path

REQUIRED = ["index_path", "model_display", "model_aliases", "models", "doc_types"]
# 可被 adapter 补齐的软缺失；其余缺失视为硬错误。
SOFT = {"index_path", "model_display"}


class ConfigAdapter:
    """在原 config 上补齐缺失接口的轻量包装；其余属性透传。"""

    def __init__(self, config):
        self._c = config

    def __getattr__(self, name):
        return getattr(self._c, name)

    def model_display(self, model_id: str) -> str:
        for m in self._c.models:
            if m.get("id") == model_id:
                return m.get("name", model_id)
        return model_id

    def index_path(self, strategy: str | None = None) -> Path:
        s = strategy or self._c.raw["chunking"]["strategy"]
        return Path(self._c.index_dir) / s / "corpus"


def probe_config(config):
    missing = [name for name in REQUIRED if not hasattr(config, name)]
    if not missing:
        return config
    hard = [m for m in missing if m not in SOFT]
    if hard:
        raise AttributeError(f"config 缺少必需接口且无 fallback: {hard}")
    return ConfigAdapter(config)

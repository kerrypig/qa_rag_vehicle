"""把点路径覆盖项就地写入 config.raw（用于关闭检索中的 Ollama 步骤）。"""
from __future__ import annotations

from typing import Any


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    """overrides 形如 {"query_rewrite.enabled": False}；按 '.' 分段写入 raw。

    中间节点不存在或不是 dict 时，新建为 dict 后继续。
    """
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node = raw
        for key in keys[:-1]:
            nxt = node.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                node[key] = nxt
            node = nxt
        node[keys[-1]] = value

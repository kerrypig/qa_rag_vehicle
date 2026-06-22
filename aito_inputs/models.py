"""把用户输入的车型 token 解析为 config 中的车型 id（用于 RAG 按车型过滤）。

支持「家族式」模糊匹配：token 拆成 latin/数字段与中文段，要求某车型的
id/name/aliases 归一文本里同时包含所有分段。例如：
- "M7"       → 所有 M7 变体
- "M9纯电"   → 所有 M9 纯电变体
- "问界M9-2025款纯电版" → 精确单个
"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_/]+", "", str(s or "")).lower()


def _segments(token: str) -> list[str]:
    """拆成连续的 [a-z0-9] 段与中文段。"""
    return re.findall(r"[a-z0-9]+|[一-鿿]+", _norm(token))


def _model_blob(m: dict) -> str:
    parts = [m.get("id", ""), m.get("name", ""), *m.get("aliases", [])]
    return _norm("".join(parts))


def resolve_models(tokens: list[str], config) -> tuple[list[str], list[str], list[str]]:
    """返回 (车型 id 列表, 展示标签列表, 未解析 token 列表)。

    标签用于改写前缀：缺少「问界」前缀时自动补上（如 "M7" → "问界M7"）。
    """
    models = config.models
    ids: list[str] = []
    labels: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        segs = _segments(token)
        hits = [m["id"] for m in models if segs and all(s in _model_blob(m) for s in segs)]
        if not hits:
            unresolved.append(token)
            continue
        for mid in hits:
            if mid not in seen:
                seen.add(mid)
                ids.append(mid)
        labels.append(token if token.startswith("问界") else f"问界{token}")
    return ids, labels, unresolved


def available_models(config) -> list[str]:
    """返回所有可选车型展示名（用于报错提示）。"""
    return [m.get("name", m["id"]) for m in config.models]

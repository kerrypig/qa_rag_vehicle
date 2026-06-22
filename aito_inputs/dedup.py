"""精确去重 + 近重复过滤（纯逻辑，无 I/O）。

近重复用字符 bigram 的 Jaccard 相似度衡量，避免「胎压灯亮了怎么办」与
「胎压报警灯亮了怎么办呀」这类高度雷同的问题重复入选。
"""
from __future__ import annotations

import re


def normalize(text: str) -> str:
    """去除空白与标点，便于精确比较。"""
    return re.sub(r"[\s\W_]+", "", str(text or "")).lower()


def _bigrams(s: str) -> set[str]:
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a: str, b: str) -> float:
    """两段文本归一后的字符 bigram Jaccard 相似度。"""
    sa, sb = _bigrams(normalize(a)), _bigrams(normalize(b))
    if not sa or not sb:
        return 1.0 if sa == sb else 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def dedup_questions(questions: list[str], *, threshold: float = 0.85) -> list[str]:
    """先精确去重，再剔除与已保留项相似度 ≥ threshold 的近重复，保序返回。"""
    seen_exact: set[str] = set()
    kept: list[str] = []
    kept_norm: list[str] = []
    for q in questions:
        n = normalize(q)
        if not n or n in seen_exact:
            continue
        if any(jaccard(q, k) >= threshold for k in kept):
            continue
        seen_exact.add(n)
        kept.append(q)
        kept_norm.append(n)
    return kept

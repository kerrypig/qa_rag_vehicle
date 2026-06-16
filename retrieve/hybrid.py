"""RRF 融合向量与 BM25 检索结果。"""

from __future__ import annotations

import re

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    weights: list[float],
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}

    for lst, weight in zip(ranked_lists, weights, strict=True):
        for rank, (doc, _) in enumerate(lst, start=1):
            cid = doc.metadata.get("chunk_id", doc.page_content[:40])
            docs[cid] = doc
            scores[cid] = scores.get(cid, 0.0) + weight / (rrf_k + rank)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs[cid], score) for cid, score in merged[:top_k]]


def apply_section_boost(
    ranked: list[tuple[Document, float]],
    keywords: str,
    boost: float = 1.1,
) -> list[tuple[Document, float]]:
    """keyword 命中 section_path 时提升 RRF 分数（方案 B，轻量标签 boost）。"""
    if boost <= 1.0 or not keywords:
        return ranked
    tokens = {t for t in re.findall(r"[\w\u4e00-\u9fff]+", keywords.lower()) if len(t) >= 2}
    if not tokens:
        return ranked
    boosted: list[tuple[Document, float]] = []
    for doc, score in ranked:
        sp = doc.metadata.get("section_path", "").lower()
        if any(t in sp for t in tokens):
            score *= boost
        boosted.append((doc, score))
    return sorted(boosted, key=lambda x: x[1], reverse=True)

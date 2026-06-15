"""RRF 融合向量与 BM25 检索结果。"""

from __future__ import annotations

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

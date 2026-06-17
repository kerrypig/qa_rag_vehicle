"""向量检索 + metadata 过滤。"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def vector_search(
    db: FAISS,
    query: str,
    k: int = 20,
    *,
    vehicle_models: set[str] | None = None,
    doc_types: list[str] | None = None,
) -> list[tuple[Document, float]]:
    docs_scores = db.similarity_search_with_score(query, k=k * 3)

    results: list[tuple[Document, float]] = []
    for doc, dist in docs_scores:
        meta = doc.metadata
        if vehicle_models and meta.get("vehicle_model") not in vehicle_models:
            continue
        if doc_types and meta.get("doc_type") not in doc_types:
            continue
        sim = 1.0 - dist / 2.0
        results.append((doc, sim))
        if len(results) >= k:
            break
    return results

"""BM25 关键词检索。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


class BM25Store:
    def __init__(self, corpus_path: Path):
        with open(corpus_path, encoding="utf-8") as f:
            data = json.load(f)
        self.chunk_ids: list[str] = data["chunk_ids"]
        self.texts: list[str] = data["texts"]
        self.metadatas: list[dict] = data["metadatas"]
        self._id_to_doc = {
            cid: Document(page_content=t, metadata=m)
            for cid, t, m in zip(self.chunk_ids, self.texts, self.metadatas, strict=True)
        }
        tokenized = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    @classmethod
    def load(cls, index_path: Path) -> BM25Store:
        return cls(index_path / "bm25_corpus.json")

    def search(
        self,
        query: str,
        k: int = 20,
        *,
        vehicle_models: set[str] | None = None,
        doc_types: list[str] | None = None,
    ) -> list[tuple[Document, float]]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[tuple[Document, float]] = []
        for idx, score in ranked:
            meta = self.metadatas[idx]
            if vehicle_models and meta.get("vehicle_model") not in vehicle_models:
                continue
            if doc_types and meta.get("doc_type") not in doc_types:
                continue
            doc = self._id_to_doc[self.chunk_ids[idx]]
            results.append((doc, float(score)))
            if len(results) >= k:
                break
        return results

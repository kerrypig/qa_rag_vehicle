"""MVP 内存会话：对话历史 + chunk 缓存池。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document


@dataclass
class CachedChunk:
    doc: Document
    score: float = 0.0


@dataclass
class SessionState:
    response_id: str | None = None
    turns: list[tuple[str, str]] = field(default_factory=list)
    chunk_cache: OrderedDict[str, CachedChunk] = field(default_factory=OrderedDict)
    last_retrieval: dict | None = None

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))

    def recent_user_questions(self, n: int) -> list[str]:
        return [t[0] for t in self.turns[-n:]]

    def section_hints(self, limit: int = 5) -> list[str]:
        hints = []
        for item in self.chunk_cache.values():
            sp = item.doc.metadata.get("section_path", "")
            if sp and sp not in hints:
                hints.append(sp)
            if len(hints) >= limit:
                break
        return hints

    def match_cache(
        self,
        query: str,
        embeddings,
        *,
        threshold: float,
        max_reuse: int,
    ) -> tuple[list[Document], dict[str, float]]:
        if not self.chunk_cache:
            return [], {}

        q_vec = np.array(embeddings.embed_query(query))
        scored: list[tuple[str, float]] = []
        for cid, item in self.chunk_cache.items():
            d_vec = np.array(embeddings.embed_query(item.doc.page_content[:512]))
            sim = float(np.dot(q_vec, d_vec))
            if sim >= threshold:
                scored.append((cid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        docs: list[Document] = []
        scores: dict[str, float] = {}
        for cid, sim in scored[:max_reuse]:
            docs.append(self.chunk_cache[cid].doc)
            scores[cid] = sim
        return docs, scores

    def update_cache(self, docs: list[Document], scores: dict[str, float], max_size: int) -> None:
        for doc in docs:
            cid = doc.metadata["chunk_id"]
            self.chunk_cache[cid] = CachedChunk(doc=doc, score=scores.get(cid, 0.0))
            self.chunk_cache.move_to_end(cid)

        while len(self.chunk_cache) > max_size:
            self.chunk_cache.popitem(last=False)

    def clear(self) -> None:
        self.response_id = None
        self.turns.clear()
        self.chunk_cache.clear()
        self.last_retrieval = None

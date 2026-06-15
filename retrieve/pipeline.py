"""检索编排：Rewrite → 缓存 → Hybrid/向量 → 合并。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ingest.indexer import build_embeddings
from retrieve.bm25_store import BM25Store
from retrieve.hybrid import reciprocal_rank_fusion
from retrieve.query_rewrite import rewrite_query
from retrieve.vector_store import vector_search
from session.memory import SessionState

log = logging.getLogger(__name__)


def format_retrieved_chunks(
    docs: list[Document],
    scores: dict[str, float],
    *,
    cache_hits: int,
    new_retrieved: int,
) -> str:
    """格式化检索 chunk 全文，供 log 文件与 QA 日志使用。"""
    lines = [
        f"[检索] 缓存命中 {cache_hits} | 新检索 {new_retrieved} | 合计 {len(docs)} 条"
    ]
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        cid = meta.get("chunk_id", "")
        score = scores.get(cid, 0.0)
        page = meta.get("page", "?")
        section = meta.get("section_path", "")
        lines.append(
            f"  [{i}] P.{page} | {section} | score={score:.4f}\n"
            f"      {doc.page_content.strip()}"
        )
    return "\n".join(lines)


@dataclass
class RetrievalResult:
    query: str
    rewritten_query: str
    docs: list[Document]
    cache_hits: int = 0
    new_retrieved: int = 0
    scores: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, config, index_path):
        self.config = config
        self.index_path = index_path
        self.db: FAISS | None = None
        self.bm25: BM25Store | None = None
        self.embeddings = build_embeddings(config)

    def load(self) -> None:
        from ingest.indexer import load_vectorstore

        self.db = load_vectorstore(self.index_path, self.config)
        corpus = self.index_path / "bm25_corpus.json"
        if corpus.exists():
            self.bm25 = BM25Store.load(self.index_path)

    def retrieve(self, question: str, session: SessionState) -> RetrievalResult:
        cfg = self.config.raw
        rw_cfg = cfg["query_rewrite"]
        ret_cfg = cfg["retrieval"]
        top_k = ret_cfg["top_k"]
        filter_model = self.config.vehicle_model if ret_cfg["metadata_filter"]["enabled"] else None
        filter_types = self.config.doc_types if ret_cfg["metadata_filter"]["enabled"] else None

        rewritten = question
        if rw_cfg.get("enabled"):
            history = session.recent_user_questions(rw_cfg.get("max_history_turns", 2))
            hints = session.section_hints(limit=5)
            rewritten = rewrite_query(
                question,
                vehicle_model=self.config.vehicle_model,
                history=history,
                section_hints=hints,
                model=rw_cfg.get("model", "qwen2.5:7b"),
                temperature=rw_cfg.get("temperature", 0.0),
            )

        cache_cfg = cfg["session"]["chunk_cache"]
        cached_docs, cache_scores = session.match_cache(
            rewritten,
            self.embeddings,
            threshold=cache_cfg.get("reuse_threshold", 0.55),
            max_reuse=cache_cfg.get("max_reuse", 3),
        )

        fetch_k = max(top_k - len(cached_docs), 1)
        if ret_cfg["hybrid_search"].get("enabled") and self.bm25:
            vec = vector_search(
                self.db,
                rewritten,
                k=20,
                vehicle_model=filter_model,
                doc_types=filter_types,
            )
            bm25 = self.bm25.search(
                rewritten,
                k=20,
                vehicle_model=filter_model,
                doc_types=filter_types,
            )
            hs = ret_cfg["hybrid_search"]
            w_bm25 = hs.get("bm25_weight", 0.4)
            merged = reciprocal_rank_fusion(
                [vec, bm25],
                [1.0 - w_bm25, w_bm25],
                rrf_k=hs.get("rrf_k", 60),
                top_k=fetch_k,
            )
            new_docs = [d for d, _ in merged]
            new_scores = {d.metadata["chunk_id"]: s for d, s in merged}
        else:
            vec = vector_search(
                self.db,
                rewritten,
                k=fetch_k,
                vehicle_model=filter_model,
                doc_types=filter_types,
            )
            new_docs = [d for d, _ in vec]
            new_scores = {d.metadata["chunk_id"]: s for d, _ in vec}

        seen = {d.metadata["chunk_id"] for d in cached_docs}
        final: list[Document] = list(cached_docs)
        scores = dict(cache_scores)
        new_count = 0
        for doc in new_docs:
            cid = doc.metadata["chunk_id"]
            if cid in seen:
                continue
            final.append(doc)
            scores[cid] = new_scores.get(cid, 0.0)
            seen.add(cid)
            new_count += 1
            if len(final) >= top_k:
                break

        threshold = ret_cfg.get("score_threshold", 0.0)
        if threshold > 0:
            final = [d for d in final if scores.get(d.metadata["chunk_id"], 0) >= threshold]

        session.update_cache(final, scores, max_size=cache_cfg.get("max_size", 20))

        log.info(
            "\n%s",
            format_retrieved_chunks(
                final,
                scores,
                cache_hits=len(cached_docs),
                new_retrieved=new_count,
            ),
        )

        return RetrievalResult(
            query=question,
            rewritten_query=rewritten,
            docs=final,
            cache_hits=len(cached_docs),
            new_retrieved=new_count,
            scores=scores,
        )

"""检索编排：Rewrite → 双路检索 → 合并 → 核对。"""



from __future__ import annotations



import logging

from dataclasses import dataclass, field



from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document



from ingest.indexer import build_embeddings

from retrieve.bm25_store import BM25Store

from retrieve.hybrid import apply_section_boost, reciprocal_rank_fusion

from retrieve.query_rewrite import extract_keyword, rewrite_query, rewrite_question

from retrieve.vector_store import vector_search

from retrieve.verifier import verify_chunks

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

    keyword: str

    rewritten_query: str

    docs: list[Document]

    cache_hits: int = 0

    new_retrieved: int = 0

    pre_verify_count: int = 0

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



    def _hybrid_search(

        self,

        query: str,

        k: int,

        *,

        filter_model: str | None,

        filter_types: list[str] | None,

        bm25_weight: float | None = None,

        section_boost: float = 1.0,

        keywords_for_boost: str = "",

    ) -> tuple[list[Document], dict[str, float], dict[str, float]]:

        """返回 (docs, rrf_scores, vector_sim_by_id)。"""

        ret_cfg = self.config.raw["retrieval"]

        vector_sim_by_id: dict[str, float] = {}



        if ret_cfg["hybrid_search"].get("enabled") and self.bm25:

            vec = vector_search(

                self.db,

                query,

                k=20,

                vehicle_model=filter_model,

                doc_types=filter_types,

            )

            vector_sim_by_id = {d.metadata["chunk_id"]: s for d, s in vec}

            bm25 = self.bm25.search(

                query,

                k=20,

                vehicle_model=filter_model,

                doc_types=filter_types,

            )

            hs = ret_cfg["hybrid_search"]

            w_bm25 = bm25_weight if bm25_weight is not None else hs.get("bm25_weight", 0.4)

            merged = reciprocal_rank_fusion(

                [vec, bm25],

                [1.0 - w_bm25, w_bm25],

                rrf_k=hs.get("rrf_k", 60),

                top_k=k,

            )

            if section_boost > 1.0 and keywords_for_boost:

                merged = apply_section_boost(merged, keywords_for_boost, section_boost)

                merged = merged[:k]

            docs = [d for d, _ in merged]

            scores = {d.metadata["chunk_id"]: s for d, s in merged}

        else:

            vec = vector_search(

                self.db,

                query,

                k=k,

                vehicle_model=filter_model,

                doc_types=filter_types,

            )

            vector_sim_by_id = {d.metadata["chunk_id"]: s for d, s in vec}

            docs = [d for d, _ in vec]

            scores = dict(vector_sim_by_id)



        return docs, scores, vector_sim_by_id



    def _prepare_queries(

        self,

        question: str,

        session: SessionState | None,

    ) -> tuple[str, str]:

        cfg = self.config.raw

        rw_cfg = cfg["query_rewrite"]

        ret_cfg = cfg["retrieval"]

        rw_enabled = rw_cfg.get("enabled", False)

        kw_enabled = ret_cfg.get("keyword_search", {}).get("enabled", True)



        keyword = question

        rewritten = question



        if not rw_enabled and not kw_enabled:

            return keyword, rewritten



        max_earlier = rw_cfg.get("max_history_turns", 2)

        hints = session.section_hints(limit=5) if session else []

        last_turn = session.last_turn() if session else None

        earlier = session.earlier_user_questions(max_earlier) if session else []

        model = rw_cfg.get("model", "qwen2.5:7b")

        temperature = rw_cfg.get("temperature", 0.0)

        common = dict(

            vehicle_model=self.config.vehicle_model,

            last_turn=last_turn,

            section_hints=hints,

            model=model,

            temperature=temperature,

        )



        if rw_enabled and kw_enabled:

            rw_result = rewrite_question(

                question,

                earlier_questions=earlier,

                **common,

            )

            return rw_result.keyword, rw_result.rewritten_query

        if rw_enabled:

            rewritten = rewrite_query(question, earlier_questions=earlier, **common)

            return question, rewritten

        keyword = extract_keyword(question, **common)

        return keyword, question



    def _retrieve_core(

        self,

        question: str,

        *,

        session: SessionState | None = None,

    ) -> RetrievalResult:

        cfg = self.config.raw

        ret_cfg = cfg["retrieval"]

        keyword_top_k = ret_cfg.get("keyword_top_k", 2)

        rewritten_top_k = ret_cfg.get("rewritten_top_k", 5)

        kw_enabled = ret_cfg.get("keyword_search", {}).get("enabled", True)

        section_boost = ret_cfg.get("section_boost", 1.1)

        filter_model = self.config.vehicle_model if ret_cfg["metadata_filter"]["enabled"] else None

        filter_types = self.config.doc_types if ret_cfg["metadata_filter"]["enabled"] else None



        keyword, rewritten = self._prepare_queries(question, session)

        max_total = (keyword_top_k if kw_enabled else 0) + rewritten_top_k



        cached_docs: list[Document] = []

        cache_scores: dict[str, float] = {}

        if session is not None:

            cache_cfg = cfg["session"]["chunk_cache"]

            cached_docs, cache_scores = session.match_cache(

                rewritten,

                self.embeddings,

                threshold=cache_cfg.get("reuse_threshold", 0.55),

                max_reuse=cache_cfg.get("max_reuse", 3),

            )



        hs = ret_cfg.get("hybrid_search", {})

        kw_bm25_weight = hs.get("keyword_bm25_weight", 0.6)

        vector_sim_by_id: dict[str, float] = {}



        rw_docs: list[Document] = []

        rw_scores: dict[str, float] = {}

        if rewritten_top_k > 0:

            rw_docs, rw_scores, rw_vec_sim = self._hybrid_search(

                rewritten,

                rewritten_top_k,

                filter_model=filter_model,

                filter_types=filter_types,

            )

            vector_sim_by_id.update(rw_vec_sim)



        kw_docs: list[Document] = []

        kw_scores: dict[str, float] = {}

        if kw_enabled and keyword_top_k > 0:

            kw_docs, kw_scores, kw_vec_sim = self._hybrid_search(

                keyword,

                keyword_top_k,

                filter_model=filter_model,

                filter_types=filter_types,

                bm25_weight=kw_bm25_weight,

                section_boost=section_boost,

                keywords_for_boost=keyword,

            )

            vector_sim_by_id.update(kw_vec_sim)



        seen = {d.metadata["chunk_id"] for d in cached_docs}

        final: list[Document] = list(cached_docs)

        scores = dict(cache_scores)

        new_count = 0



        for doc in rw_docs + kw_docs:

            cid = doc.metadata["chunk_id"]

            if cid in seen:

                continue

            final.append(doc)

            scores[cid] = rw_scores.get(cid, kw_scores.get(cid, 0.0))

            seen.add(cid)

            new_count += 1

            if len(final) >= max_total:

                break



        threshold = ret_cfg.get("score_threshold", 0.0)

        if threshold > 0:

            kept: list[Document] = []

            for d in final:

                cid = d.metadata["chunk_id"]

                sim = vector_sim_by_id.get(cid)

                if sim is None or sim >= threshold:

                    kept.append(d)

                else:

                    scores[cid] = sim

            if len(kept) < len(final):

                log.info(

                    "score_threshold=%.2f 过滤 %d/%d 条（向量相似度不足）",

                    threshold,

                    len(final) - len(kept),

                    len(final),

                )

            final = kept



        for d in final:

            cid = d.metadata["chunk_id"]

            if cid in vector_sim_by_id:

                scores[cid] = vector_sim_by_id[cid]



        pre_verify_count = len(final)

        ver_cfg = cfg.get("verification", {})

        if ver_cfg.get("enabled") and final:

            pre_verify_docs = list(final)

            verified = verify_chunks(

                question,

                final,

                model=ver_cfg.get("model", "qwen2.5:7b"),

                temperature=ver_cfg.get("temperature", 0.0),

            )

            if verified:

                log.info("[Verify] %d/%d 条通过核对", len(verified), pre_verify_count)

                final = verified

            else:

                log.warning(

                    "[Verify] 全部 %d 条未通过核对，回退为全部召回",

                    pre_verify_count,

                )

                final = pre_verify_docs



        if session is not None:

            cache_cfg = cfg["session"]["chunk_cache"]

            session.update_cache(final, scores, max_size=cache_cfg.get("max_size", 20))



        return RetrievalResult(

            query=question,

            keyword=keyword,

            rewritten_query=rewritten,

            docs=final,

            cache_hits=len(cached_docs),

            new_retrieved=new_count,

            pre_verify_count=pre_verify_count,

            scores=scores,

        )



    def retrieve_stateless(self, question: str) -> RetrievalResult:

        """无会话、无缓存的检索，供 benchmark 使用。"""

        return self._retrieve_core(question, session=None)



    def retrieve(self, question: str, session: SessionState) -> RetrievalResult:

        return self._retrieve_core(question, session=session)



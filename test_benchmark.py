"""
独立检索性能测试脚本 (不修改原有项目代码)
运行方式: python test_benchmark.py
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
from pathlib import Path
from typing import List
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config
from retrieve.pipeline import Retriever
from retrieve.hybrid import reciprocal_rank_fusion, apply_section_boost
from retrieve.query_rewrite import rewrite_question
from retrieve.vector_store import vector_search


class BenchmarkRetriever(Retriever):
    def retrieve_stateless(self, question: str) -> List[Document]:
        """无状态检索评测专用方法（含双路检索，不含核对）。"""
        cfg = self.config.raw
        rw_cfg = cfg.get("query_rewrite", {})
        ret_cfg = cfg.get("retrieval", {})
        keyword_top_k = ret_cfg.get("keyword_top_k", 2)
        rewritten_top_k = ret_cfg.get("rewritten_top_k", 5)
        max_total = keyword_top_k + rewritten_top_k
        section_boost = ret_cfg.get("section_boost", 1.1)

        filter_model = self.config.vehicle_model if ret_cfg.get("metadata_filter", {}).get("enabled") else None
        filter_types = self.config.doc_types if ret_cfg.get("metadata_filter", {}).get("enabled") else None

        keyword = question
        rewritten = question
        if rw_cfg.get("enabled"):
            rw = rewrite_question(
                question,
                vehicle_model=self.config.vehicle_model,
                last_turn=None,
                earlier_questions=None,
                section_hints=None,
                model=rw_cfg.get("model", "qwen2.5:7b"),
                temperature=rw_cfg.get("temperature", 0.0),
            )
            keyword = rw.keyword
            rewritten = rw.rewritten_query
            print(f"====== 关键词：「{keyword}」")
            print(f"====== 改写后问题：「{rewritten}」")

        vector_sim_by_id: dict[str, float] = {}
        hs = ret_cfg.get("hybrid_search", {})
        kw_bm25_weight = hs.get("keyword_bm25_weight", 0.6)

        def _search(query: str, k: int, bm25_w: float | None, boost_kw: str = "") -> list[Document]:
            nonlocal vector_sim_by_id
            if hs.get("enabled") and self.bm25:
                vec = vector_search(self.db, query, k=20, vehicle_model=filter_model, doc_types=filter_types)
                vector_sim_by_id.update({d.metadata["chunk_id"]: s for d, s in vec})
                bm25 = self.bm25.search(query, k=20, vehicle_model=filter_model, doc_types=filter_types)
                w_bm25 = bm25_w if bm25_w is not None else hs.get("bm25_weight", 0.4)
                merged = reciprocal_rank_fusion(
                    [vec, bm25], [1.0 - w_bm25, w_bm25], rrf_k=hs.get("rrf_k", 60), top_k=k
                )
                if section_boost > 1.0 and boost_kw:
                    merged = apply_section_boost(merged, boost_kw, section_boost)[:k]
                return [d for d, _ in merged]
            vec = vector_search(self.db, query, k=k, vehicle_model=filter_model, doc_types=filter_types)
            vector_sim_by_id.update({d.metadata["chunk_id"]: s for d, s in vec})
            return [d for d, _ in vec]

        rw_docs = _search(rewritten, rewritten_top_k, None)
        kw_docs = _search(keyword, keyword_top_k, kw_bm25_weight, keyword)

        seen: set[str] = set()
        raw_docs: List[Document] = []
        for doc in rw_docs + kw_docs:
            cid = doc.metadata["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            raw_docs.append(doc)
            if len(raw_docs) >= max_total:
                break

        threshold = ret_cfg.get("score_threshold", 0.0)
        final_docs: List[Document] = []

        for d in raw_docs:
            cid = d.metadata["chunk_id"]
            sim = vector_sim_by_id.get(cid)
            if sim is None or sim >= threshold:
                if sim is not None:
                    d.metadata["test_vector_score"] = sim
                final_docs.append(d)

        return final_docs


if __name__ == "__main__":
    print("=== 正在初始化测试环境 ===")

    config_path = ROOT / "config.yaml"
    config = load_config(str(config_path))
    index_path = config.index_path()

    if not (index_path / "faiss").exists():
        print(f"报错：找不到向量索引，请先执行 main.py build")
        sys.exit(1)

    print(f"正在加载向量库...")
    tester = BenchmarkRetriever(config, index_path)
    tester.load()

    questions = ["夏天车里被晒得像烤箱，怎么能最快把冷风开到最大？",
                 "仪表盘右下角那个 0% PWR 进度条是干嘛的？",
                 "我是五座版 M9，后排中间的那个座位能不能用 ISOFIX 固定接口装儿童座椅？",
                 "车门外把手没有自己弹出来，我现在在车外，怎么把门打开？",
                 "我车后面挂了房车，用了那个电动拖挂辅助，怎么现在 NCA（领航辅助）和自适应巡航不能用了？"]

    for question in questions:
        print(f"\n========== 开始单次检索测试: '{question}' ==========\n")
        docs = tester.retrieve_stateless(question=question)

        print(f"共检索到 {len(docs)} 条结果:\n")
        for i, doc in enumerate(docs, 1):
            score = doc.metadata.get("test_vector_score", 0.0)
            chunk_id = doc.metadata.get("chunk_id", "未知ID")
            content_preview = doc.page_content.strip().replace("\n", " ")

            print(f"[Top {i}] Score: {score:.4f} | Chunk: {chunk_id}")
            print(f"内容: {content_preview}...")
            print("-" * 50)

"""建库：FAISS 向量索引 + BM25 语料持久化。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

log = logging.getLogger(__name__)


def build_embeddings(config) -> HuggingFaceEmbeddings:
    cfg = config.raw["embedding"]
    return HuggingFaceEmbeddings(
        model_name=cfg["model"],
        model_kwargs={"device": cfg.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": cfg.get("normalize", True)},
    )


def save_index(documents: list[Document], index_path: Path, config) -> None:
    index_path.mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings(config)

    log.info("向量化 %d 个 chunk …", len(documents))
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(str(index_path / "faiss"))

    corpus = {
        "chunk_ids": [d.metadata["chunk_id"] for d in documents],
        "texts": [d.page_content for d in documents],
        "metadatas": [d.metadata for d in documents],
    }
    with open(index_path / "bm25_corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    per_model: dict[str, int] = {}
    for d in documents:
        mid = d.metadata.get("vehicle_model", "")
        per_model[mid] = per_model.get(mid, 0) + 1

    with open(index_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": per_model,
                "doc_types": config.doc_types,
                "strategy": config.chunk_strategy,
                "chunk_count": len(documents),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info("索引已保存至 %s", index_path)


def load_vectorstore(index_path: Path, config) -> FAISS:
    embeddings = build_embeddings(config)
    return FAISS.load_local(
        str(index_path / "faiss"),
        embeddings,
        allow_dangerous_deserialization=True,
    )

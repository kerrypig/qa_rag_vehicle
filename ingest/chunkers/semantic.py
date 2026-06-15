"""基于 embedding 语义断点的切分。"""

from __future__ import annotations

import uuid

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ingest.pdf_loader import PageBlock


class SemanticChunker:
    def __init__(self, opts: dict, embed_cfg: dict):
        self.min_chars = opts.get("min_chunk_chars", 100)
        self.max_chars = opts.get("max_chunk_chars", 2000)
        self.threshold_pct = opts.get("breakpoint_threshold", 85)
        device = embed_cfg.get("device", "cpu")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=opts.get("embed_model", embed_cfg.get("model", "BAAI/bge-large-zh-v1.5")),
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": embed_cfg.get("normalize", True)},
        )

    def chunk(
        self,
        pages: list[PageBlock],
        *,
        vehicle_model: str,
        doc_type: str,
        source_file: str,
    ) -> list[Document]:
        paragraphs: list[tuple[str, int]] = []
        for page in pages:
            for para in page.text.split("\n"):
                t = para.strip()
                if t:
                    paragraphs.append((t, page.page))

        if not paragraphs:
            return []

        raw_chunks = self._semantic_split([p[0] for p in paragraphs])
        docs: list[Document] = []
        para_idx = 0

        for chunk_text in raw_chunks:
            page = paragraphs[min(para_idx, len(paragraphs) - 1)][1]
            para_idx += max(1, chunk_text.count("\n") + 1)
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_id": str(uuid.uuid4()),
                        "vehicle_model": vehicle_model,
                        "doc_type": doc_type,
                        "source_file": source_file,
                        "page": page,
                        "section_path": "semantic",
                        "image_refs": [],
                    },
                )
            )
        return self._enforce_size(docs)

    def _semantic_split(self, paragraphs: list[str]) -> list[str]:
        if len(paragraphs) <= 1:
            return ["\n".join(paragraphs)]

        vecs = self.embeddings.embed_documents(paragraphs)
        distances = []
        for i in range(len(vecs) - 1):
            sim = float(np.dot(vecs[i], vecs[i + 1]))
            distances.append(1.0 - sim)

        threshold = float(np.percentile(distances, self.threshold_pct))
        chunks: list[str] = []
        buf = [paragraphs[0]]
        for i, dist in enumerate(distances):
            if dist >= threshold:
                chunks.append("\n".join(buf))
                buf = [paragraphs[i + 1]]
            else:
                buf.append(paragraphs[i + 1])
        chunks.append("\n".join(buf))
        return chunks

    def _enforce_size(self, docs: list[Document]) -> list[Document]:
        out: list[Document] = []
        for doc in docs:
            text = doc.page_content
            if len(text) <= self.max_chars:
                out.append(doc)
                continue
            start = 0
            while start < len(text):
                piece = text[start : start + self.max_chars]
                out.append(
                    Document(
                        page_content=piece,
                        metadata={**doc.metadata, "chunk_id": str(uuid.uuid4())},
                    )
                )
                start += self.max_chars - self.min_chars // 2
        return out

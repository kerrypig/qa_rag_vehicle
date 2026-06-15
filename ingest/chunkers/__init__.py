"""切分策略统一接口与工厂。"""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document

from ingest.pdf_loader import PageBlock


class BaseChunker(Protocol):
    def chunk(
        self,
        pages: list[PageBlock],
        *,
        vehicle_model: str,
        doc_type: str,
        source_file: str,
    ) -> list[Document]: ...


def get_chunker(strategy: str, config) -> BaseChunker:
    if strategy == "hierarchy":
        from ingest.chunkers.hierarchy import HierarchyChunker

        return HierarchyChunker(config.raw["chunking"]["hierarchy"])
    if strategy == "semantic":
        from ingest.chunkers.semantic import SemanticChunker

        return SemanticChunker(config.raw["chunking"]["semantic"], config.raw["embedding"])
    if strategy == "fixed_size":
        from ingest.chunkers.fixed_size import FixedSizeChunker

        return FixedSizeChunker(config.raw["chunking"]["fixed_size"])
    raise ValueError(f"未知切分策略: {strategy}")

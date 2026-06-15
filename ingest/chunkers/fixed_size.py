"""字符数 + overlap 切分。"""

from __future__ import annotations

import uuid

from langchain_core.documents import Document

from ingest.pdf_loader import PageBlock


class FixedSizeChunker:
    def __init__(self, opts: dict):
        self.chunk_size = opts.get("chunk_size", 800)
        self.overlap = opts.get("overlap", 150)

    def chunk(
        self,
        pages: list[PageBlock],
        *,
        vehicle_model: str,
        doc_type: str,
        source_file: str,
    ) -> list[Document]:
        full_text = "\n\n".join(p.text for p in pages if p.text.strip())
        page_map = self._build_page_map(pages)
        docs: list[Document] = []
        total = len(full_text)

        for start in range(0, total, self.chunk_size):
            win_start = max(0, start - self.overlap)
            win_end = min(total, start + self.chunk_size + self.overlap)
            content = full_text[win_start:win_end].strip()
            if not content:
                continue
            page = page_map.get(win_start, 1)
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "chunk_id": str(uuid.uuid4()),
                        "vehicle_model": vehicle_model,
                        "doc_type": doc_type,
                        "source_file": source_file,
                        "page": page,
                        "section_path": "fixed_size",
                        "image_refs": [],
                    },
                )
            )
        return docs

    def _build_page_map(self, pages: list[PageBlock]) -> dict[int, int]:
        mapping: dict[int, int] = {}
        offset = 0
        for page in pages:
            text = page.text
            if not text.strip():
                continue
            mapping[offset] = page.page
            offset += len(text) + 2
        return mapping

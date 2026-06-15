"""按 PDF 标题层级切分。"""

from __future__ import annotations

import uuid
from statistics import median

from langchain_core.documents import Document

from ingest.pdf_loader import PageBlock, infer_heading_level


class HierarchyChunker:
    def __init__(self, opts: dict):
        self.min_chars = opts.get("min_chunk_chars", 200)
        self.max_chars = opts.get("max_chunk_chars", 1500)
        self.merge_short = opts.get("merge_short_siblings", True)

    def chunk(
        self,
        pages: list[PageBlock],
        *,
        vehicle_model: str,
        doc_type: str,
        source_file: str,
    ) -> list[Document]:
        all_blocks = [b for p in pages for b in p.blocks]
        sizes = [b.font_size for b in all_blocks if b.text.strip()]
        med = median(sizes) if sizes else 12.0

        sections: list[dict] = []
        heading_stack: list[str] = []

        for page in pages:
            for block in page.blocks:
                level = infer_heading_level(block, med)
                if level is not None:
                    heading_stack = heading_stack[: level - 1]
                    heading_stack.append(block.text.strip())
                    continue
                if not block.text.strip():
                    continue
                sections.append(
                    {
                        "text": block.text.strip(),
                        "page": block.page,
                        "section_path": ">".join(heading_stack) if heading_stack else "正文",
                    }
                )

        merged = self._merge_sections(sections)
        return [
            Document(
                page_content=s["text"],
                metadata={
                    "chunk_id": str(uuid.uuid4()),
                    "vehicle_model": vehicle_model,
                    "doc_type": doc_type,
                    "source_file": source_file,
                    "page": s["page"],
                    "section_path": s["section_path"],
                    "image_refs": [],
                },
            )
            for s in merged
        ]

    def _merge_sections(self, sections: list[dict]) -> list[dict]:
        if not sections:
            return []

        groups: list[dict] = []
        buf = sections[0].copy()
        buf["text"] = sections[0]["text"]

        for sec in sections[1:]:
            same = sec["section_path"] == buf["section_path"]
            combined_len = len(buf["text"]) + len(sec["text"]) + 1

            if same and combined_len <= self.max_chars:
                buf["text"] += "\n" + sec["text"]
                buf["page"] = min(buf["page"], sec["page"])
            elif len(buf["text"]) < self.min_chars and self.merge_short:
                buf["text"] += "\n" + sec["text"]
                buf["page"] = min(buf["page"], sec["page"])
                if len(buf["text"]) > self.max_chars:
                    groups.append(buf)
                    buf = sec.copy()
            else:
                groups.append(buf)
                buf = sec.copy()
                buf["text"] = sec["text"]

        groups.append(buf)
        return self._split_oversized(groups)

    def _split_oversized(self, groups: list[dict]) -> list[dict]:
        out: list[dict] = []
        for g in groups:
            if len(g["text"]) <= self.max_chars:
                out.append(g)
                continue
            paras = g["text"].split("\n")
            chunk = ""
            for para in paras:
                if len(chunk) + len(para) + 1 > self.max_chars and chunk:
                    out.append({**g, "text": chunk.strip()})
                    chunk = para
                else:
                    chunk = f"{chunk}\n{para}" if chunk else para
            if chunk.strip():
                out.append({**g, "text": chunk.strip()})
        return out

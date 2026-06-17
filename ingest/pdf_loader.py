"""PDF 解析：按页提取文本与标题层级线索。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TextBlock:
    text: str
    page: int
    font_size: float
    is_bold: bool


@dataclass
class PageBlock:
    page: int
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text.strip())


@dataclass
class BookmarkEntry:
    level: int
    title: str
    page: int
    path: str
    source_file: str = ""
    vehicle_model: str = ""


def _is_bold(flags: int) -> bool:
    # PyMuPDF font flags: bit 4 = bold
    return bool(flags & 2**4)


def load_pdf_bookmarks(
    pdf_path: str, *, source_file: str = "", vehicle_model: str = ""
) -> list[BookmarkEntry]:
    """从 PDF outline（书签）提取目录项。page 为 1-based 页码。"""
    import fitz

    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    doc.close()

    entries: list[BookmarkEntry] = []
    stack: list[str] = []
    for level, title, page in toc:
        title = title.strip()
        if not title or page < 1:
            continue
        while len(stack) >= level:
            stack.pop()
        stack.append(title)
        entries.append(
            BookmarkEntry(
                level=level,
                title=title,
                page=page,
                path=">".join(stack),
                source_file=source_file,
                vehicle_model=vehicle_model,
            )
        )
    return entries


def load_pdf(pdf_path: str) -> list[PageBlock]:
    import fitz

    doc = fitz.open(pdf_path)
    pages: list[PageBlock] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        pb = PageBlock(page=page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                max_span = max(spans, key=lambda s: s.get("size", 0))
                pb.blocks.append(
                    TextBlock(
                        text=text,
                        page=page_num,
                        font_size=float(max_span.get("size", 12)),
                        is_bold=_is_bold(int(max_span.get("flags", 0))),
                    )
                )
        pages.append(pb)

    doc.close()
    return pages


def infer_heading_level(block: TextBlock, median_size: float) -> int | None:
    """根据字号与粗体推断标题层级，非标题返回 None。"""
    text = block.text.strip()
    if len(text) > 80:
        return None
    if re.match(r"^第[一二三四五六七八九十\d]+[章节篇]", text):
        return 1
    if block.font_size >= median_size * 1.25 or (block.is_bold and block.font_size >= median_size * 1.1):
        if len(text) <= 40:
            return 2 if block.font_size < median_size * 1.5 else 1
    if block.is_bold and len(text) <= 30:
        return 3
    return None

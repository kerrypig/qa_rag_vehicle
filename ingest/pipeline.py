"""建库入口：PDF → 切分 → 索引。"""

from __future__ import annotations

import logging
from pathlib import Path

from ingest.chunkers import get_chunker
from ingest.indexer import save_index
from ingest.pdf_loader import load_pdf, load_pdf_bookmarks
from retrieve.bookmark_match import save_bookmarks

log = logging.getLogger(__name__)


def run_build(config, pdf_path: Path | None = None) -> int:
    pdf_dir = config.pdf_dir
    pdfs = [pdf_path] if pdf_path else sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"未在 {pdf_dir} 找到 PDF。请将手册放入该目录，例如 m9-2025-evr-product-manual-20260415.pdf"
        )

    strategy = config.chunk_strategy
    chunker = get_chunker(strategy, config)
    all_docs = []
    all_bookmarks: list = []

    for pdf in pdfs:
        log.info("解析 PDF: %s", pdf.name)
        pages = load_pdf(str(pdf))
        all_bookmarks.extend(load_pdf_bookmarks(str(pdf), source_file=pdf.name))
        docs = chunker.chunk(
            pages,
            vehicle_model=config.vehicle_model,
            doc_type=config.doc_types[0],
            source_file=pdf.name,
        )
        log.info("  → %d chunks", len(docs))
        all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError("切分结果为空，请检查 PDF 内容。")

    index_path = config.index_path(strategy)
    save_index(all_docs, index_path, config)
    if all_bookmarks:
        save_bookmarks(all_bookmarks, index_path)
        log.info("书签已保存：%d 条", len(all_bookmarks))
    return len(all_docs)

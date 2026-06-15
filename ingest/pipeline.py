"""建库入口：PDF → 切分 → 索引。"""

from __future__ import annotations

import logging
from pathlib import Path

from ingest.chunkers import get_chunker
from ingest.indexer import save_index
from ingest.pdf_loader import load_pdf

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

    for pdf in pdfs:
        log.info("解析 PDF: %s", pdf.name)
        pages = load_pdf(str(pdf))
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
    return len(all_docs)

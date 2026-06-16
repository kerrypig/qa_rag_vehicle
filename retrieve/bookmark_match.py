"""PDF 书签匹配：用 Ollama 选章节并解析为 chunk。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import ollama
from langchain_core.documents import Document

from ingest.pdf_loader import BookmarkEntry, load_pdf_bookmarks
from prompts.template import BOOKMARK_MATCH_TEMPLATE
from retrieve.bm25_store import BM25Store

log = logging.getLogger(__name__)

BOOKMARK_SCORE = 1.0


def save_bookmarks(entries: list[BookmarkEntry], index_path: Path) -> None:
    index_path.mkdir(parents=True, exist_ok=True)
    data = {
        "entries": [
            {
                "level": e.level,
                "title": e.title,
                "page": e.page,
                "path": e.path,
                "source_file": e.source_file,
            }
            for e in entries
        ]
    }
    with open(index_path / "bookmarks.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_bookmarks(index_path: Path, pdf_dir: Path | None = None) -> list[BookmarkEntry]:
    """优先读索引内 bookmarks.json，否则从 PDF 目录解析。"""
    json_path = index_path / "bookmarks.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return [
            BookmarkEntry(
                level=e["level"],
                title=e["title"],
                page=e["page"],
                path=e["path"],
                source_file=e.get("source_file", ""),
            )
            for e in data.get("entries", [])
        ]

    if pdf_dir and pdf_dir.exists():
        entries: list[BookmarkEntry] = []
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            entries.extend(load_pdf_bookmarks(str(pdf), source_file=pdf.name))
        return entries
    return []


def _bookmark_page_end(bookmarks: list[BookmarkEntry], index: int, max_page: int) -> int:
    level = bookmarks[index].level
    for j in range(index + 1, len(bookmarks)):
        if bookmarks[j].level <= level:
            return max(bookmarks[index].page, bookmarks[j].page - 1)
    return max_page


def _parse_selected_indices(text: str, max_index: int, max_matches: int) -> list[int]:
    line = text.strip().split("\n")[0].strip()
    if not line or line in ("无", "none", "None", "没有"):
        return []
    nums = [int(n) for n in re.findall(r"\d+", line)]
    seen: set[int] = set()
    picked: list[int] = []
    for n in nums:
        if 1 <= n <= max_index and n not in seen:
            seen.add(n)
            picked.append(n)
        if len(picked) >= max_matches:
            break
    return picked


def select_bookmarks_with_llm(
    question: str,
    bookmarks: list[BookmarkEntry],
    *,
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
    max_matches: int = 3,
) -> list[BookmarkEntry]:
    if not bookmarks or max_matches <= 0:
        return []

    lines = []
    for i, bm in enumerate(bookmarks, start=1):
        indent = "  " * (bm.level - 1)
        lines.append(f"{i}. {indent}{bm.title}（第 {bm.page} 页）")
    catalog = "\n".join(lines)

    prompt = BOOKMARK_MATCH_TEMPLATE.format(
        question=question,
        bookmark_catalog=catalog,
        max_matches=max_matches,
    )
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": 64},
        )
        indices = _parse_selected_indices(resp["response"], len(bookmarks), max_matches)
        selected = [bookmarks[i - 1] for i in indices]
        if selected:
            log.info(
                "[Bookmark] %s → %s",
                question,
                ", ".join(b.title for b in selected),
            )
        return selected
    except Exception as e:
        log.warning("[Bookmark] Ollama 不可用: %s", e)
        return []


def _title_in_section(title: str, section_path: str) -> bool:
    title = title.strip()
    if not title or not section_path:
        return False
    if title in section_path:
        return True
    for part in section_path.split(">"):
        if title in part or part in title:
            return True
    return False


def resolve_chunks_for_bookmarks(
    selected: list[BookmarkEntry],
    all_bookmarks: list[BookmarkEntry],
    bm25: BM25Store,
    *,
    chunks_per_match: int = 2,
    filter_model: str | None = None,
    filter_types: list[str] | None = None,
) -> list[Document]:
    if not selected:
        return []

    max_page = max((m.get("page", 0) for m in bm25.metadatas), default=9999)
    docs: list[Document] = []
    seen: set[str] = set()

    for bm in selected:
        try:
            idx = all_bookmarks.index(bm)
        except ValueError:
            idx = next((i for i, b in enumerate(all_bookmarks) if b.title == bm.title and b.page == bm.page), -1)
        if idx < 0:
            continue

        start_page = bm.page
        end_page = _bookmark_page_end(all_bookmarks, idx, max_page)
        candidates: list[tuple[int, Document]] = []

        for i, meta in enumerate(bm25.metadatas):
            if filter_model and meta.get("vehicle_model") != filter_model:
                continue
            if filter_types and meta.get("doc_type") not in filter_types:
                continue
            page = int(meta.get("page", 0))
            if page < start_page or page > end_page:
                continue
            cid = bm25.chunk_ids[i]
            doc = bm25._id_to_doc[cid]
            section = meta.get("section_path", "")
            rank = 0 if _title_in_section(bm.title, section) else 1
            candidates.append((rank, doc))

        candidates.sort(key=lambda x: (x[0], int(x[1].metadata.get("page", 0))))
        for _, doc in candidates[:chunks_per_match]:
            cid = doc.metadata["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            marked = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "retrieval_source": "bookmark",
                    "bookmark_title": bm.title,
                    "bookmark_path": bm.path,
                },
            )
            docs.append(marked)

    return docs


def retrieve_by_bookmarks(
    question: str,
    *,
    index_path: Path,
    pdf_dir: Path | None,
    bm25: BM25Store | None,
    filter_model: str | None,
    filter_types: list[str] | None,
    model: str,
    temperature: float,
    max_matches: int,
    chunks_per_match: int,
) -> tuple[list[Document], list[str]]:
    if not bm25 or max_matches <= 0:
        return [], []

    bookmarks = load_bookmarks(index_path, pdf_dir)
    if not bookmarks:
        log.warning("[Bookmark] 未找到书签数据")
        return [], []

    selected = select_bookmarks_with_llm(
        question,
        bookmarks,
        model=model,
        temperature=temperature,
        max_matches=max_matches,
    )
    docs = resolve_chunks_for_bookmarks(
        selected,
        bookmarks,
        bm25,
        chunks_per_match=chunks_per_match,
        filter_model=filter_model,
        filter_types=filter_types,
    )
    titles = [b.title for b in selected]
    return docs, titles

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
# 泛化章节：问题未明确提及时，关键词回退时降权
_GENERIC_PATH_KEYWORDS = ("重要提示", "前言", "用户指南", "车辆介绍", "目录", "免责声明")


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
                "vehicle_model": e.vehicle_model,
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
                vehicle_model=e.get("vehicle_model", ""),
            )
            for e in data.get("entries", [])
        ]

    if pdf_dir and pdf_dir.exists():
        entries: list[BookmarkEntry] = []
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            entries.extend(load_pdf_bookmarks(str(pdf), source_file=pdf.name))
        return entries
    return []


def leaf_bookmarks(bookmarks: list[BookmarkEntry]) -> list[BookmarkEntry]:
    """仅保留叶子小节（无子章节的最细粒度书签）。"""
    if not bookmarks:
        return []
    leaves: list[BookmarkEntry] = []
    for i, bm in enumerate(bookmarks):
        has_child = i + 1 < len(bookmarks) and bookmarks[i + 1].level > bm.level
        if not has_child:
            leaves.append(bm)
    return leaves


def bookmarks_for_matching(
    bookmarks: list[BookmarkEntry],
    *,
    leaf_only: bool = True,
) -> list[BookmarkEntry]:
    return leaf_bookmarks(bookmarks) if leaf_only else bookmarks


def _bookmark_page_end(bookmarks: list[BookmarkEntry], index: int, max_page: int) -> int:
    level = bookmarks[index].level
    for j in range(index + 1, len(bookmarks)):
        if bookmarks[j].level <= level:
            return max(bookmarks[index].page, bookmarks[j].page - 1)
    return max_page


def _pick_valid_indices(nums: list[int], max_index: int, max_matches: int) -> list[int]:
    seen: set[int] = set()
    picked: list[int] = []
    for n in nums:
        if 1 <= n <= max_index and n not in seen:
            seen.add(n)
            picked.append(n)
        if len(picked) >= max_matches:
            break
    return picked


def _collect_tokens(question: str, keyword: str = "", rewritten: str = "") -> list[str]:
    tokens: list[str] = []
    for text in (question, rewritten):
        tokens.extend(t for t in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()) if len(t) >= 2)
    for part in keyword.replace("，", ",").split(","):
        part = part.strip().lower()
        if len(part) >= 2:
            tokens.append(part)
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _prefilter_catalog(
    question: str,
    catalog: list[BookmarkEntry],
    *,
    keyword: str = "",
    rewritten: str = "",
    max_items: int = 80,
) -> list[BookmarkEntry]:
    """用问题 + 改写 + 关键词缩小目录。"""
    tokens = _collect_tokens(question, keyword, rewritten)
    if not tokens:
        return catalog[:max_items] if len(catalog) > max_items else catalog

    scored: list[tuple[float, BookmarkEntry]] = []
    for bm in catalog:
        path_parts = [p.lower() for p in bm.path.split(">") if p]
        hay = f"{bm.path}>{bm.title}".lower()
        score = 0.0
        for t in tokens:
            if t in hay:
                score += 1.0
            for part in path_parts:
                if t in part or part in t:
                    score += 2.0
        if score > 0:
            score += len(bm.path) * 0.01
            scored.append((score, bm))

    if not scored:
        return sorted(catalog, key=lambda b: (-b.level, b.page))[:max_items]

    scored.sort(key=lambda x: -x[0])
    return [bm for _, bm in scored[:max_items]]


def _score_bookmark(bm: BookmarkEntry, tokens: list[str], question: str) -> float:
    path_parts = [p.lower() for p in bm.path.split(">") if p]
    hay = f"{bm.path}>{bm.title}".lower()
    score = 0.0
    for t in tokens:
        if t in hay:
            score += 1.0
        for part in path_parts:
            if t in part or part in t:
                score += 2.0
    for generic in _GENERIC_PATH_KEYWORDS:
        if generic in bm.path and generic not in question:
            score -= 3.0
    return score + len(bm.path) * 0.01


def _keyword_fallback_match(
    question: str,
    catalog: list[BookmarkEntry],
    *,
    keyword: str = "",
    rewritten: str = "",
    max_matches: int,
) -> list[BookmarkEntry]:
    tokens = _collect_tokens(question, keyword, rewritten)
    if not tokens:
        return []

    scored: list[tuple[float, BookmarkEntry]] = []
    for bm in catalog:
        score = _score_bookmark(bm, tokens, question)
        if score > 0:
            scored.append((score, bm))

    if not scored:
        return []

    scored.sort(key=lambda x: -x[0])
    return [bm for _, bm in scored[:max_matches]]


def _parse_paths_from_response(text: str, catalog: list[BookmarkEntry], max_matches: int) -> list[BookmarkEntry]:
    """从 LLM 回复中提取目录里存在的完整路径（按路径长度降序，避免短串误匹配）。"""
    if not text.strip():
        return []
    compact = text.strip().replace("，", ",")
    if compact in ("无", "none", "None", "没有"):
        return []

    paths = sorted({bm.path for bm in catalog if bm.path}, key=len, reverse=True)
    picked: list[BookmarkEntry] = []
    seen: set[str] = set()
    for path in paths:
        if path in text and path not in seen:
            seen.add(path)
            for bm in catalog:
                if bm.path == path:
                    picked.append(bm)
                    break
        if len(picked) >= max_matches:
            break
    return picked


def _parse_selected_indices(text: str, max_index: int, max_matches: int) -> list[int]:
    """仅接受纯数字编号行；绝不把「2. 车辆介绍」这类 LLM 列表误解析为编号 2。"""
    text = text.strip()
    if not text:
        return []
    compact = text.replace(" ", "").replace("，", ",")
    if compact in ("无", "none", "None", "没有"):
        return []

    if re.fullmatch(r"[\d,、]+", compact):
        return _pick_valid_indices([int(n) for n in re.findall(r"\d+", compact)], max_index, max_matches)

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in reversed(lines):
        cleaned = line.replace(" ", "").replace("，", ",")
        if re.fullmatch(r"[\d,、]+", cleaned):
            picked = _pick_valid_indices([int(n) for n in re.findall(r"\d+", cleaned)], max_index, max_matches)
            if picked:
                return picked

    return []


def select_bookmarks_with_llm(
    question: str,
    bookmarks: list[BookmarkEntry],
    *,
    keyword: str = "",
    rewritten_query: str = "",
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
    max_matches: int = 3,
    raw_out: list[str] | None = None,
    fallback_note: list[str] | None = None,
) -> list[BookmarkEntry]:
    if not bookmarks or max_matches <= 0:
        return []

    kw = keyword if keyword != question else ""
    rw = rewritten_query if rewritten_query != question else ""

    filtered = _prefilter_catalog(question, bookmarks, keyword=kw, rewritten=rw)
    catalog_text = _format_catalog(filtered)
    prompt = BOOKMARK_MATCH_TEMPLATE.format(
        question=question,
        rewritten_query=rw or question,
        keyword=kw or "（无）",
        bookmark_catalog=catalog_text,
        max_matches=max_matches,
        catalog_count=len(filtered),
    )
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": 128},
        )
        raw = resp["response"].strip()
        if raw_out is not None:
            raw_out.append(raw)

        selected = _parse_paths_from_response(raw, filtered, max_matches)
        if not selected:
            indices = _parse_selected_indices(raw, len(filtered), max_matches)
            selected = [filtered[i - 1] for i in indices]

        llm_empty = raw.replace(" ", "") in ("无", "none", "None", "没有") or not selected

        if llm_empty:
            selected = _keyword_fallback_match(
                question, filtered, keyword=kw, rewritten=rw, max_matches=max_matches,
            )
            if selected and fallback_note is not None:
                fallback_note.append(f"关键词回退（预筛 {len(filtered)} 条）")
            if not selected:
                selected = _keyword_fallback_match(
                    question, bookmarks, keyword=kw, rewritten=rw, max_matches=max_matches,
                )
                if selected and fallback_note is not None:
                    fallback_note.append(f"关键词回退（全目录 {len(bookmarks)} 条）")
            if selected:
                log.info(
                    "[Bookmark] LLM 返回「无」或不可解析，关键词回退 → %s",
                    ", ".join(b.path for b in selected),
                )
        elif selected:
            log.info("[Bookmark] %s → %s", question, ", ".join(b.path for b in selected))
        return selected
    except Exception as e:
        log.warning("[Bookmark] Ollama 不可用: %s", e)
        fb = _keyword_fallback_match(question, bookmarks, keyword=kw, rewritten=rw, max_matches=max_matches)
        if fb and fallback_note is not None:
            fallback_note.append("Ollama 不可用，关键词回退")
        return fb


def _format_catalog(bookmarks: list[BookmarkEntry]) -> str:
    lines = []
    for i, bm in enumerate(bookmarks, start=1):
        path = bm.path or bm.title
        lines.append(f"{i}. {path}（第 {bm.page} 页）")
    return "\n".join(lines)


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
    filter_models: set[str] | None = None,
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
            idx = next(
                (i for i, b in enumerate(all_bookmarks) if b.title == bm.title and b.page == bm.page),
                -1,
            )
        if idx < 0:
            continue

        start_page = bm.page
        end_page = _bookmark_page_end(all_bookmarks, idx, max_page)
        candidates: list[tuple[int, Document]] = []

        for i, meta in enumerate(bm25.metadatas):
            if filter_models and meta.get("vehicle_model") not in filter_models:
                continue
            if filter_types and meta.get("doc_type") not in filter_types:
                continue
            page = int(meta.get("page", 0))
            if page < start_page or page > end_page:
                continue
            cid = bm25.chunk_ids[i]
            doc = bm25._id_to_doc[cid]
            section = meta.get("section_path", "")
            path_match = _title_in_section(bm.title, section) or _title_in_section(bm.path, section)
            rank = 0 if path_match else 1
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
    keyword: str = "",
    rewritten_query: str = "",
    index_path: Path,
    pdf_dir: Path | None,
    bm25: BM25Store | None,
    filter_models: set[str] | None,
    filter_types: list[str] | None,
    model: str,
    temperature: float,
    max_matches: int,
    chunks_per_match: int,
    leaf_only: bool = True,
    raw_out: list[str] | None = None,
    fallback_note: list[str] | None = None,
) -> tuple[list[Document], list[str]]:
    if not bm25 or max_matches <= 0:
        return [], []

    all_bookmarks = load_bookmarks(index_path, pdf_dir)
    if not all_bookmarks:
        log.warning("[Bookmark] 未找到书签数据")
        return [], []

    # 按车型分组匹配：每个车型用其自身叶子目录做 LLM 匹配，并在该车型书签
    # 的页码范围内解析 chunk，避免多 PDF 页码串档。
    if filter_models:
        target_models: list[str | None] = list(filter_models)
    else:
        present = sorted({b.vehicle_model for b in all_bookmarks if b.vehicle_model})
        target_models = present or [None]

    docs_all: list[Document] = []
    titles_all: list[str] = []
    for mid in target_models:
        model_bms = [
            b for b in all_bookmarks if mid is None or b.vehicle_model == mid
        ]
        if not model_bms:
            continue
        catalog = bookmarks_for_matching(model_bms, leaf_only=leaf_only)
        if not catalog:
            continue
        log.info(
            "[Bookmark] 车型 %s 叶子目录 %d 条（全书签 %d 条，leaf_only=%s）",
            mid or "(全部)", len(catalog), len(model_bms), leaf_only,
        )
        selected = select_bookmarks_with_llm(
            question,
            catalog,
            keyword=keyword,
            rewritten_query=rewritten_query,
            model=model,
            temperature=temperature,
            max_matches=max_matches,
            raw_out=raw_out,
            fallback_note=fallback_note,
        )
        sub_filter = {mid} if mid is not None else None
        docs = resolve_chunks_for_bookmarks(
            selected,
            model_bms,
            bm25,
            chunks_per_match=chunks_per_match,
            filter_models=sub_filter,
            filter_types=filter_types,
        )
        docs_all.extend(docs)
        titles_all.extend(b.path for b in selected)

    return docs_all, titles_all

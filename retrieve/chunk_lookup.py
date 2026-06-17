"""在索引语料中按文本片段查找 chunk_id。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkMatch:
    chunk_id: str
    needle: str
    page: str | int
    section_path: str
    snippet: str


def _load_corpus(index_path: Path) -> dict:
    corpus_path = index_path / "bm25_corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"语料文件不存在: {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        return json.load(f)


def find_chunk_ids_by_text(
    needle: str,
    index_path: Path,
    *,
    case_sensitive: bool = False,
    max_snippet: int = 120,
) -> list[ChunkMatch]:
    """返回 chunk 正文中包含 needle 的所有匹配项。"""
    if not needle.strip():
        return []

    corpus = _load_corpus(index_path)
    hay = needle if case_sensitive else needle.lower()
    matches: list[ChunkMatch] = []

    for cid, text, meta in zip(
        corpus["chunk_ids"],
        corpus["texts"],
        corpus["metadatas"],
        strict=True,
    ):
        body = text if case_sensitive else text.lower()
        pos = body.find(hay)
        if pos < 0:
            continue
        start = max(0, pos - 40)
        end = min(len(text), pos + len(needle) + 80)
        snippet = text[start:end].replace("\n", " ")
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"

        matches.append(
            ChunkMatch(
                chunk_id=cid,
                needle=needle,
                page=meta.get("page", "?"),
                section_path=meta.get("section_path", ""),
                snippet=snippet,
            )
        )
    return matches


def find_chunk_ids_by_texts(
    needles: list[str],
    index_path: Path,
    *,
    case_sensitive: bool = False,
) -> dict[str, list[ChunkMatch]]:
    return {
        needle: find_chunk_ids_by_text(needle, index_path, case_sensitive=case_sensitive)
        for needle in needles
    }


def unique_chunk_ids(matches: list[ChunkMatch]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for m in matches:
        if m.chunk_id not in seen:
            seen.add(m.chunk_id)
            ordered.append(m.chunk_id)
    return ordered


def format_matches(matches: list[ChunkMatch]) -> str:
    if not matches:
        return "（无匹配）"
    lines: list[str] = []
    for i, m in enumerate(matches, start=1):
        lines.append(
            f"  [{i}] chunk_id={m.chunk_id}\n"
            f"      P.{m.page} | {m.section_path}\n"
            f"      {m.snippet}"
        )
    return "\n".join(lines)

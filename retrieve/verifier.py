"""Ollama 串行核对检索片段相关性（偏宽松，避免误杀 GT）。"""

from __future__ import annotations

import logging

import ollama
from langchain_core.documents import Document

from prompts.template import CHUNK_VERIFY_TEMPLATE

log = logging.getLogger(__name__)


def _is_bookmark_chunk(doc: Document) -> bool:
    return doc.metadata.get("retrieval_source") == "bookmark"


def _parse_verdict(text: str) -> bool:
    answer = text.strip().split("\n")[0].strip()
    if answer in ("是", "yes", "Yes", "YES", "相关"):
        return True
    if answer in ("否", "no", "No", "NO", "无关"):
        return False
    if "否" in answer and "是" not in answer:
        return False
    if "是" in answer:
        return True
    # 不确定时保留，避免误杀
    return True


def verify_chunks(
    question: str,
    docs: list[Document],
    *,
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
    min_keep: int = 2,
    scores: dict[str, float] | None = None,
) -> list[Document]:
    """串行核对；书签 chunk 跳过；过严时保底 min_keep 或回退全量。"""
    if not docs:
        return []

    to_verify = [d for d in docs if not _is_bookmark_chunk(d)]
    bookmark_docs = [d for d in docs if _is_bookmark_chunk(d)]

    if not to_verify:
        return list(docs)

    kept: list[Document] = []
    rejected: list[Document] = []
    score_map = scores or {}

    for i, doc in enumerate(to_verify, start=1):
        meta = doc.metadata
        section = meta.get("section_path", "")
        page = meta.get("page", "?")
        prompt = CHUNK_VERIFY_TEMPLATE.format(
            question=question,
            section=section,
            page=page,
            chunk_text=doc.page_content.strip()[:1500],
        )
        try:
            resp = ollama.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": 16},
            )
            verdict = _parse_verdict(resp["response"])
            cid = meta.get("chunk_id", "")
            log.info("[Verify %d/%d] %s P.%s → %s", i, len(to_verify), cid, page, "通过" if verdict else "拒绝")
            if verdict:
                kept.append(doc)
            else:
                rejected.append(doc)
        except Exception as e:
            log.warning("[Verify %d/%d] Ollama 失败，保留该条: %s", i, len(to_verify), e)
            kept.append(doc)

    if not kept:
        log.warning("[Verify] 全部 %d 条未通过，回退为全部 hybrid 召回", len(to_verify))
        kept = list(to_verify)
    elif len(kept) < min_keep and rejected:
        need = min_keep - len(kept)
        rejected.sort(
            key=lambda d: score_map.get(d.metadata.get("chunk_id", ""), 0.0),
            reverse=True,
        )
        for doc in rejected[:need]:
            kept.append(doc)
        log.info("[Verify] 保底 min_keep=%d，补回 %d 条", min_keep, min(need, len(rejected)))

    return bookmark_docs + kept

"""Ollama 串行核对检索片段相关性。"""

from __future__ import annotations

import logging

import ollama
from langchain_core.documents import Document

from prompts.template import CHUNK_VERIFY_TEMPLATE

log = logging.getLogger(__name__)


def _parse_verdict(text: str) -> bool:
    answer = text.strip().split("\n")[0].strip()
    if answer in ("是", "yes", "Yes", "YES"):
        return True
    if answer in ("否", "no", "No", "NO"):
        return False
    return "是" in answer and "否" not in answer


def verify_chunks(
    question: str,
    docs: list[Document],
    *,
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
) -> list[Document]:
    """串行调用 Ollama，逐条判断片段是否含关键信息。"""
    kept: list[Document] = []
    for i, doc in enumerate(docs, start=1):
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
            log.info("[Verify %d/%d] %s P.%s → %s", i, len(docs), cid, page, "通过" if verdict else "拒绝")
            if verdict:
                kept.append(doc)
        except Exception as e:
            log.warning("[Verify %d/%d] Ollama 失败，保留该条: %s", i, len(docs), e)
            kept.append(doc)
    return kept

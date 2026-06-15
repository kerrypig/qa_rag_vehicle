"""组装 RAG Prompt context。"""

from __future__ import annotations

from langchain_core.documents import Document

from prompts.template import RAG_QA_TEMPLATE


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        page = meta.get("page", "?")
        section = meta.get("section_path", "")
        header = f"[资料{i}] (P.{page} | {section})" if section else f"[资料{i}] (P.{page})"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(parts) if parts else "（无检索到相关资料）"


def build_prompt(question: str, docs: list[Document]) -> str:
    return RAG_QA_TEMPLATE.format(
        retrieved_context=format_context(docs),
        question=question,
    )

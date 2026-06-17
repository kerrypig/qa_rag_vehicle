"""Ollama 本地 Query Rewrite：串行提取 keyword + 改写问句。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import ollama

from prompts.template import KEYWORD_EXTRACT_TEMPLATE, QUERY_REWRITE_TEMPLATE

log = logging.getLogger(__name__)

_LAST_ASSISTANT_MAX = 400


@dataclass
class RewriteResult:
    keyword: str
    rewritten_query: str


def _format_last_turn(last_turn: tuple[str, str] | None) -> str:
    if not last_turn:
        return "（无，这是首轮对话）"
    user_q, assistant_a = last_turn
    answer = assistant_a.strip()
    if len(answer) > _LAST_ASSISTANT_MAX:
        answer = answer[:_LAST_ASSISTANT_MAX] + "……"
    return f"用户：{user_q.strip()}\n助手：{answer}"


def _format_earlier_history(questions: list[str]) -> str:
    if not questions:
        return "（无）"
    return "\n".join(f"- {q}" for q in questions)


def _format_hints(section_hints: list[str] | None) -> str:
    if not section_hints:
        return "（无）"
    return "\n".join(f"- {s}" for s in section_hints)


def _clean_keyword(text: str) -> str:
    line = text.strip().split("\n")[0].strip()
    line = re.sub(r"^关键词[：:\s]*", "", line).strip()
    return line or ""


def extract_keyword(
    question: str,
    *,
    vehicle_model: str,
    last_turn: tuple[str, str] | None = None,
    section_hints: list[str] | None = None,
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
) -> str:
    prompt = KEYWORD_EXTRACT_TEMPLATE.format(
        vehicle_model=vehicle_model,
        last_turn=_format_last_turn(last_turn),
        section_hints=_format_hints(section_hints),
        question=question,
    )
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": 128},
        )
        keyword = _clean_keyword(resp["response"])
        if keyword:
            log.info("[Keyword] %s → %s", question, keyword)
            return keyword
    except Exception as e:
        log.warning("[Keyword] Ollama 不可用，使用原问题: %s", e)
    return question


def rewrite_query(
    question: str,
    *,
    vehicle_model: str,
    last_turn: tuple[str, str] | None = None,
    earlier_questions: list[str] | None = None,
    section_hints: list[str] | None = None,
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
) -> str:
    prompt = QUERY_REWRITE_TEMPLATE.format(
        vehicle_model=vehicle_model,
        last_turn=_format_last_turn(last_turn),
        earlier_history=_format_earlier_history(earlier_questions or []),
        section_hints=_format_hints(section_hints),
        question=question,
    )
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": 256},
        )
        rewritten = resp["response"].strip().split("\n")[0].strip()
        if rewritten:
            log.info("[Rewrite] %s → %s", question, rewritten)
            return rewritten
    except Exception as e:
        log.warning("[Rewrite] Ollama 不可用，使用原问题: %s", e)
    return question


def rewrite_question(
    question: str,
    *,
    vehicle_model: str,
    last_turn: tuple[str, str] | None = None,
    earlier_questions: list[str] | None = None,
    section_hints: list[str] | None = None,
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
) -> RewriteResult:
    """串行：先提取 keyword，再改写问句（两次独立 Ollama 会话）。"""
    keyword = extract_keyword(
        question,
        vehicle_model=vehicle_model,
        last_turn=last_turn,
        section_hints=section_hints,
        model=model,
        temperature=temperature,
    )
    rewritten = rewrite_query(
        question,
        vehicle_model=vehicle_model,
        last_turn=last_turn,
        earlier_questions=earlier_questions,
        section_hints=section_hints,
        model=model,
        temperature=temperature,
    )
    return RewriteResult(keyword=keyword, rewritten_query=rewritten)

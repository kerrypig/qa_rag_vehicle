"""Ollama 本地 Query Rewrite。"""

from __future__ import annotations

import logging

import ollama

from prompts.template import QUERY_REWRITE_TEMPLATE

log = logging.getLogger(__name__)


def rewrite_query(
    question: str,
    *,
    vehicle_model: str,
    history: list[str],
    section_hints: list[str],
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
) -> str:
    history_text = "\n".join(f"- {h}" for h in history) if history else "（无）"
    hints = "\n".join(f"- {s}" for s in section_hints) if section_hints else "（无）"
    prompt = QUERY_REWRITE_TEMPLATE.format(
        vehicle_model=vehicle_model,
        history=history_text,
        section_hints=hints,
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

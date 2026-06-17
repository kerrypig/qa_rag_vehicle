"""按问句识别目标车型（规则匹配，支持多车型命中）。"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


def _norm(text: str) -> str:
    """归一：去空白 + 小写，使匹配对空格/大小写不敏感。"""
    return re.sub(r"\s+", "", text).lower()


def detect_models(question: str, rewritten: str, config) -> list[str]:
    """从问题 + 改写问句识别车型 id 列表。

    策略：把所有 (model_id, alias) 按归一别名长度降序匹配；命中文本段后标记
    「已消费」，避免泛化短别名（如「M9纯电」）在更具体别名（如「M9 2025款纯电版」）
    已覆盖的区域重复命中。天然支持一问多车型（比较类问题）。
    无命中返回 []，交由上层走会话粘性 / 追问。
    """
    hay = _norm(f"{question} {rewritten}")
    if not hay:
        return []

    norm_pairs: list[tuple[str, str]] = []
    for mid, alias in config.model_aliases():
        na = _norm(alias)
        if na:
            norm_pairs.append((mid, na))
    norm_pairs.sort(key=lambda x: len(x[1]), reverse=True)

    selected: list[str] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(not (e <= cs or s >= ce) for cs, ce in consumed)

    for mid, alias in norm_pairs:
        if mid in selected:
            continue
        start = hay.find(alias)
        while start != -1:
            end = start + len(alias)
            if not overlaps(start, end):
                selected.append(mid)
                consumed.append((start, end))
                break
            start = hay.find(alias, start + 1)

    if selected:
        log.info("[Router] %s → %s", question, ", ".join(selected))
    return selected

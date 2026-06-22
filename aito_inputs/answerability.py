"""可回答性判定（纯逻辑）。

输入为 RAG 检索结果的衍生量（docs / scores）+ 上游规则旗标，输出判定 Decision。
判定不直接调用 Ollama；如启用 verify，由编排层先跑 verify_chunks，再把
relevant_count 传进来，保持本模块可单测、无 I/O。

answerability 取值：
- answerable_by_rag      ：手册能直接支撑回答（含纠正错误前提）
- safety_fallback_supported：安全类主题，手册支持「安全停车+勿自拆+联系官方」兜底
- needs_clarification    ：车型/动力形式不明，应先追问
- not_answerable         ：检索不到/相关度不足/燃油专属（→ rejected）
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    accepted: bool
    answerability: str
    expected_behavior: str
    reason: str          # rejected 时填剔除原因；accepted 为 ""
    evidence_summary: str


def max_score(scores: dict[str, float]) -> float:
    return max(scores.values()) if scores else 0.0


def summarize_evidence(docs, limit: int = 3, prefer: list[str] | None = None) -> str:
    """用检索 chunk 的章节路径拼简短证据摘要。

    prefer 给定主题词时，优先列出章节路径里命中这些词的小节，使摘要真实反映
    「凭哪个对口小节判定可答」，而不是只列检索分最高的泛章节。
    """
    prefer = prefer or []
    all_sections: list[str] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        sp = meta.get("section_path") or meta.get("section") or ""
        if sp and sp not in all_sections:
            all_sections.append(sp)
    if prefer:
        hit = [s for s in all_sections if any(p in s for p in prefer)]
        rest = [s for s in all_sections if s not in hit]
        ordered = hit + rest
    else:
        ordered = all_sections
    if ordered:
        return "检索到手册章节：" + "、".join(ordered[:limit])
    return "未检索到相关章节"


def judge(
    docs,
    scores: dict[str, float],
    *,
    covered: bool,
    non_generic_hit: bool,
    safety_critical: bool,
    needs_clarification: bool,
    wrong_premise: bool,
    is_fuel_car_only: bool,
    min_chunks: int,
    min_score: float,
    relevant_count: int | None = None,
    evidence_prefer: list[str] | None = None,
) -> Decision:
    """以「证据是否真正覆盖问题核心」为核心判定，而非「检索到东西就 answerable」。

    covered/non_generic_hit 来自 grounding.evidence_covers（词汇覆盖）；
    relevant_count 来自 Ollama verify_chunks（语义复核，默认开启）；
    evidence_prefer 为问题主题词，用于让证据摘要优先列出对口小节。
    维修诊断类问题已在上游 pre-reject，这里不再处理。
    """
    evidence = summarize_evidence(docs, prefer=evidence_prefer)
    n = len(docs)
    ms = max_score(scores)

    # 强证据：检索分达标 + 词汇覆盖 + 有非泛章节命中 +（启用时）Ollama 复核通过
    strong = n >= min_chunks and ms >= min_score and covered and non_generic_hit
    if relevant_count is not None:
        strong = strong and relevant_count >= 1

    # 1) 传统燃油车专属 → 剔除
    if is_fuel_car_only:
        return Decision(False, "not_answerable", "",
                        "传统燃油车专属零件/系统，无法合理迁移到问界新能源车", evidence)

    # 2) 几乎无检索 → 剔除
    if n == 0 or ms < min_score * 0.6:
        return Decision(False, "not_answerable", "", "RAG 检索不到足够相关依据", evidence)

    # 3) 证据未命中问题核心主题（检索到了但不相关）
    if not covered or not non_generic_hit:
        if safety_critical:
            return Decision(True, "safety_fallback_supported",
                            "安全兜底并建议联系 AITO 用户中心", "", evidence)
        return Decision(False, "not_answerable", "",
                        "检索证据未命中问题核心主题（仅泛泛相关），不足以支撑回答", evidence)

    # 4) 覆盖但 Ollama 复核/检索分不足
    if not strong:
        if safety_critical:
            return Decision(True, "safety_fallback_supported",
                            "安全兜底并建议联系 AITO 用户中心", "", evidence)
        return Decision(False, "not_answerable", "",
                        "证据相关度不足（复核未通过/分数偏低），不足以支撑回答", evidence)

    # 5) 覆盖且证据充分
    if wrong_premise:
        return Decision(True, "answerable_by_rag", "纠正错误前提", "", evidence)
    if needs_clarification:
        return Decision(True, "needs_clarification", "先追问车型/动力形式", "", evidence)
    return Decision(True, "answerable_by_rag", "直接回答", "", evidence)

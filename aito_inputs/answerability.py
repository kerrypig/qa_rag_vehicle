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


def summarize_evidence(docs, limit: int = 3) -> str:
    """用检索 chunk 的章节路径拼简短证据摘要。"""
    sections: list[str] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        sp = meta.get("section_path") or meta.get("section") or ""
        if sp and sp not in sections:
            sections.append(sp)
        if len(sections) >= limit:
            break
    if sections:
        return "检索到手册章节：" + "、".join(sections)
    return "未检索到相关章节"


def judge(
    docs,
    scores: dict[str, float],
    *,
    is_safety: bool,
    needs_clarification: bool,
    wrong_premise: bool,
    is_fuel_car_only: bool,
    min_chunks: int,
    min_score: float,
    relevant_count: int | None = None,
) -> Decision:
    """综合检索强度与规则旗标给出判定。relevant_count 为 verify 通过的 chunk 数（可选）。"""
    evidence = summarize_evidence(docs)
    n = len(docs)
    ms = max_score(scores)
    strong = n >= min_chunks and ms >= min_score
    if relevant_count is not None:
        strong = strong and relevant_count >= 1
    moderate = n >= 1 and ms >= min_score * 0.8

    # 1) 传统燃油车专属 → 直接剔除
    if is_fuel_car_only:
        return Decision(False, "not_answerable", "",
                        "传统燃油车专属零件/系统，无法合理迁移到问界新能源车", evidence)

    # 2) 几乎无相关证据 → 剔除
    if n == 0 or ms < min_score * 0.6:
        return Decision(False, "not_answerable", "",
                        "RAG 检索不到足够相关依据", evidence)

    # 3) 安全类主题：手册支持安全兜底即保留
    if is_safety and moderate:
        return Decision(True, "safety_fallback_supported",
                        "安全兜底并建议联系 AITO 用户中心", "", evidence)

    # 4) 错误前提且手册能支撑纠正
    if wrong_premise and (strong or moderate):
        return Decision(True, "answerable_by_rag", "纠正错误前提", "", evidence)

    # 5) 车型/动力不明，适合先追问
    if needs_clarification and moderate:
        return Decision(True, "needs_clarification", "先追问车型/动力形式", "", evidence)

    # 6) 证据充分 → 可直接回答
    if strong:
        return Decision(True, "answerable_by_rag", "直接回答", "", evidence)

    # 7) 有检索但相关度不足
    return Decision(False, "not_answerable", "",
                    "检索相关度不足，证据不充分", evidence)

"""把各环节结果组装成 accepted / rejected 记录（纯逻辑）。"""
from __future__ import annotations

from aito_inputs.answerability import Decision
from aito_inputs.powertrain import ScopeResult


def build_candidate(
    seq: int,
    source_question: str,
    input_text: str,
    scope: ScopeResult,
    task_type: str,
    decision: Decision,
    risk_tags: list[str],
) -> dict:
    """组装一条 accepted 候选（仅 input，不含最终 answer）。"""
    return {
        "id": f"input_{seq:03d}",
        "source_question": source_question,
        "input": input_text,
        "vehicle_scope": scope.vehicle_scope,
        "powertrain": scope.powertrain,
        "task_type": task_type,
        "answerability": decision.answerability,
        "expected_behavior": decision.expected_behavior,
        "rag_evidence_summary": decision.evidence_summary,
        "risk_tags": risk_tags,
    }


def build_rejected(
    source_question: str,
    input_text: str,
    scope: ScopeResult,
    task_type: str,
    decision: Decision,
    retrieved: bool,
) -> dict:
    """组装一条 rejected 记录，便于人工复查与调参。"""
    return {
        "source_question": source_question,
        "input": input_text,
        "vehicle_scope": scope.vehicle_scope,
        "powertrain": scope.powertrain,
        "task_type": task_type,
        "answerability": decision.answerability,
        "reason": decision.reason,
        "retrieved": retrieved,
        "rag_evidence_summary": decision.evidence_summary,
    }

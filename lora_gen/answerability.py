"""RAG 回检门：多信号分档 + partial 配额。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    section_path: str


@dataclass
class Verdict:
    accept: bool
    tier: str
    reason: str
    signals: dict = field(default_factory=dict)


def evaluate(
    *,
    seed_chunk_id: str,
    seed_section: str,
    retrieved: list[RetrievedChunk],
    judge_label: str,
    judge_conflict: bool,
    cfg: dict,
    partial_count: int,
    target: int,
) -> Verdict:
    ordered = sorted(retrieved, key=lambda r: r.score, reverse=True)
    by_id = {r.chunk_id: i for i, r in enumerate(ordered)}
    seed_rank = by_id.get(seed_chunk_id, -1) + 1  # 1-based；0=未命中
    seed_score = next((r.score for r in ordered if r.chunk_id == seed_chunk_id), 0.0)
    max_score = ordered[0].score if ordered else 0.0
    same_section = sum(1 for r in ordered if r.section_path == seed_section)
    n = len(ordered)
    topk = cfg["topk"]
    seed_in_topk = 0 < seed_rank <= topk

    signals = {
        "seed_rank": seed_rank, "seed_score": seed_score, "max_score": max_score,
        "same_section_count": same_section, "retrieved": n,
        "judge_label": judge_label, "judge_conflict": judge_conflict,
    }

    def reject(reason: str) -> Verdict:
        return Verdict(accept=False, tier="", reason=reason, signals=signals)

    def accept(tier: str) -> Verdict:
        return Verdict(accept=True, tier=tier, reason="", signals=signals)

    if judge_conflict:
        return reject("evidence_conflict")
    if not seed_in_topk:
        return reject("seed_not_returned")
    if seed_score < cfg["seed_score_min"]:
        # single_chunk_full 用更高的 seed 阈值单独判
        if not (n == 1 and judge_label == "full" and seed_score >= cfg["seed_score_min_single"]):
            return reject("low_score")

    # strong
    if (
        judge_label == "full"
        and seed_rank <= cfg["strong_seed_rank_max"]
        and max_score >= cfg["strong_max_score_min"]
        and same_section >= cfg["strong_same_section_min"]
        and n >= cfg["min_retrieved"]
    ):
        return accept("strong")

    # single_chunk_full（min_retrieved 例外）
    if n == 1 and judge_label == "full" and seed_score >= cfg["seed_score_min_single"]:
        return accept("single_chunk_full")

    # ok
    if judge_label == "full" and n >= cfg["min_retrieved"] and seed_score >= cfg["seed_score_min"]:
        return accept("ok")

    # partial → 配额内 partial_ok
    if judge_label == "partial" and n >= cfg["min_retrieved"] and seed_score >= cfg["seed_score_min"]:
        quota = math.floor(cfg["partial_ok_quota"] * target)
        if partial_count < quota:
            return accept("partial_ok")
        return reject("partial_quota_full")

    if n < cfg["min_retrieved"]:
        return reject("too_few_chunks")
    return reject("insufficient_evidence")

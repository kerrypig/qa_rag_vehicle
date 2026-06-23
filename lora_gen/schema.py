"""LoRA 样本与中间字段数据结构。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Sample:
    instruction: str
    input: str
    output: str

    def to_record(self) -> dict:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


@dataclass
class SampleMeta:
    qid: str
    model_id: str
    model_display: str
    doc_type: str
    section_path: str
    task_type: str
    seed_chunk_id: str
    seed_preview: str
    seed_score: float
    evidence_chunk_ids: list[str]
    evidence_previews: list[str]
    retrieval_scores: list[float]
    max_score: float
    seed_rank: int
    same_section_count: int
    evidence_sufficiency: str
    accept_tier: str
    backend: str
    gen_question_raw: str

    def to_record(self) -> dict:
        return asdict(self)


@dataclass
class Rejected:
    qid: str
    model_id: str
    section_path: str
    task_type: str
    seed_chunk_id: str
    seed_preview: str
    reject_stage: str
    reject_reason: str
    reject_detail: str
    question: str = ""

    def to_record(self) -> dict:
        return asdict(self)

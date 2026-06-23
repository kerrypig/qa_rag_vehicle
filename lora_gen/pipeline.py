"""编排：plan → 问题生成 → RAG 回检 → 答案生成 → 质检 → 落盘；断点续传。"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from lora_gen.answerability import RetrievedChunk, Verdict, evaluate
from lora_gen.backends import GenerationError, extract_json
from lora_gen.chunks import Chunk, load_corpus, usable_chunks_by_model
from lora_gen.prompts import (
    answer_gen_prompt, judge_prompt, pick_instruction, question_gen_prompt,
)
from lora_gen.quality import normalize_question, run_quality
from lora_gen.registry import build_plan
from lora_gen.schema import Rejected, Sample, SampleMeta

log = logging.getLogger(__name__)


def make_qid(model_id: str, chunk_id: str, task_type: str) -> str:
    raw = f"{model_id}|{chunk_id}|{task_type}"
    return "Q" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def load_done_qids(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    done: set[str] = set()
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["qid"])
    return done


def append_checkpoint(checkpoint_path: Path, qid: str, status: str) -> None:
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"qid": qid, "status": status}, ensure_ascii=False) + "\n")


def preview(text: str, n: int = 50) -> str:
    return text.strip().replace("\n", " ")[:n]


@dataclass
class RunResult:
    accepted: list[Sample] = field(default_factory=list)
    metas: list[SampleMeta] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    task_pairs: list[tuple[Sample, str]] = field(default_factory=list)


def _gen_json(backend, prompt: str, retries: int) -> dict:
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return extract_json(backend.complete(prompt))
        except GenerationError as e:
            last = e
    raise last  # type: ignore[misc]


def run(
    *,
    config,
    dg,
    retriever,
    q_backend,
    answer_backend,
    judge_backend,
    out_dir: Path,
    rng: random.Random | None = None,
) -> RunResult:
    rng = rng or random.Random(dg.raw.get("seed", 0))
    index_path = config.index_path()
    chunks = load_corpus(index_path)
    chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
    by_model = usable_chunks_by_model(
        chunks, min_chars=dg.chunks["min_chars"], blacklist=dg.chunks["section_blacklist"]
    )
    # 过采样：候选量 = ceil(target * oversample_factor)，以 accepted 达标为目标
    plan_target = math.ceil(dg.target_size * dg.raw["oversample_factor"])
    plan = build_plan(
        by_model,
        target=plan_target,
        per_vehicle_min=dg.raw["per_vehicle_min"],
        per_vehicle_max=dg.raw["per_vehicle_max"],
        vehicle_subset=dg.raw["vehicle_subset"],
        rng=rng,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.jsonl"
    done = load_done_qids(checkpoint)
    result = RunResult()
    partial_count = 0
    attempts = 0
    seen_norm: set[str] = set()
    retries = dg.raw["generation_retries"]
    max_attempts = dg.raw["max_attempts"]
    a_cfg = dg.answerability

    for item in plan:
        # 目标：accepted 达到 target_size 即停；attempts 达上限兜底
        if len(result.accepted) >= dg.target_size or attempts >= max_attempts:
            break
        qid = make_qid(item.model_id, item.chunk_id, item.task_type)
        if qid in done:
            continue
        attempts += 1
        seed = chunk_by_id[item.chunk_id]
        model_display = config.model_display(item.model_id)
        base = Rejected(
            qid=qid, model_id=item.model_id, section_path=item.section_path,
            task_type=item.task_type, seed_chunk_id=seed.chunk_id,
            seed_preview=preview(seed.text), reject_stage="", reject_reason="", reject_detail="",
        )

        # 1) 问题生成
        try:
            qj = _gen_json(
                q_backend,
                question_gen_prompt(
                    model_display=model_display, section_path=item.section_path,
                    chunk_text=seed.text, task_type=item.task_type,
                ),
                retries,
            )
            question = (qj.get("question") or "").strip()
        except GenerationError as e:
            base.reject_stage, base.reject_reason, base.reject_detail = "generation", "json_parse_failed", str(e)
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, "rejected:json_parse_failed")
            continue
        if not question:
            base.reject_stage, base.reject_reason = "generation", "missing_required_field"
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, "rejected:missing_required_field")
            continue
        base.question = question

        # 1b) normalized exact 去重（在检索前拦截，省 LLM/检索开销）
        nq = normalize_question(question)
        if nq in seen_norm:
            base.reject_stage, base.reject_reason = "generation", "duplicate"
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, "rejected:duplicate")
            continue
        seen_norm.add(nq)

        # 2) RAG 回检（单车型）
        rr = retriever.retrieve_stateless(question, trace=True, force_models=[item.model_id])
        retrieved = [
            RetrievedChunk(
                chunk_id=d.metadata["chunk_id"],
                score=rr.scores.get(d.metadata["chunk_id"], 0.0),
                section_path=d.metadata.get("section_path", ""),
            )
            for d in rr.docs
        ]
        evidence_docs = rr.docs
        evidence_text = "\n".join(d.page_content.strip() for d in evidence_docs)

        # 3) 证据充分性 judge
        try:
            jj = _gen_json(judge_backend, judge_prompt(question=question, evidence_text=evidence_text), retries)
            judge_label = jj.get("label", "no")
            judge_conflict = bool(jj.get("conflict", False))
        except GenerationError:
            judge_label, judge_conflict = "no", False

        verdict: Verdict = evaluate(
            seed_chunk_id=seed.chunk_id, seed_section=seed.section_path, retrieved=retrieved,
            judge_label=judge_label, judge_conflict=judge_conflict, cfg=a_cfg,
            partial_count=partial_count, target=dg.target_size,
        )
        if not verdict.accept:
            base.reject_stage, base.reject_reason = "answerability", verdict.reason
            base.reject_detail = json.dumps(verdict.signals, ensure_ascii=False)
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, f"rejected:{verdict.reason}")
            continue

        # 4) 答案生成（以回检 evidence 为准）
        instruction = pick_instruction(item.task_type, rng)
        try:
            aj = _gen_json(
                answer_backend,
                answer_gen_prompt(
                    model_display=model_display, instruction=instruction, question=question,
                    evidence_text=evidence_text, task_type=item.task_type,
                ),
                retries,
            )
            output = (aj.get("output") or "").strip()
        except GenerationError as e:
            base.reject_stage, base.reject_reason, base.reject_detail = "generation", "json_parse_failed", str(e)
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, "rejected:json_parse_failed")
            continue

        sample = Sample(instruction=instruction, input=question, output=output)

        # 5) 质检
        max_chars = dg.quality["max_output_chars"].get(item.task_type, 800)
        qv = run_quality(sample, evidence_text=evidence_text, model_id=item.model_id, config=config, max_chars=max_chars)
        if not qv.ok:
            base.reject_stage, base.reject_reason, base.reject_detail = "quality", qv.reason, qv.detail
            result.rejected.append(base)
            append_checkpoint(checkpoint, qid, f"rejected:{qv.reason}")
            continue

        sample.output = qv.cleaned_output
        if verdict.tier == "partial_ok":
            partial_count += 1

        meta = SampleMeta(
            qid=qid, model_id=item.model_id, model_display=model_display, doc_type=seed.doc_type,
            section_path=item.section_path, task_type=item.task_type, seed_chunk_id=seed.chunk_id,
            seed_preview=preview(seed.text), seed_score=verdict.signals["seed_score"],
            evidence_chunk_ids=[r.chunk_id for r in retrieved],
            evidence_previews=[preview(d.page_content) for d in evidence_docs],
            retrieval_scores=[r.score for r in retrieved], max_score=verdict.signals["max_score"],
            seed_rank=verdict.signals["seed_rank"], same_section_count=verdict.signals["same_section_count"],
            evidence_sufficiency=judge_label, accept_tier=verdict.tier, backend=dg.backend,
            gen_question_raw=json.dumps(qj, ensure_ascii=False),
        )
        result.accepted.append(sample)
        result.metas.append(meta)
        result.task_pairs.append((sample, item.task_type))
        append_checkpoint(checkpoint, qid, "accepted")
        log.info("[accept:%s] %s | %s", verdict.tier, item.model_id, question[:30])

    return result

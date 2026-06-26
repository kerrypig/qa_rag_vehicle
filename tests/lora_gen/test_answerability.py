from lora_gen.answerability import RetrievedChunk, evaluate

CFG = {
    "topk": 5, "min_retrieved": 2,
    "seed_score_min": 0.30, "seed_score_min_single": 0.35, "max_score_min": 0.35,
    "strong_seed_rank_max": 3, "strong_max_score_min": 0.45, "strong_same_section_min": 2,
    "partial_ok_quota": 0.08,
}

def _r(cid, score, section="A"):
    return RetrievedChunk(chunk_id=cid, score=score, section_path=section)

def test_strong_accept():
    retr = [_r("seed", 0.5, "A"), _r("c2", 0.46, "A"), _r("c3", 0.2, "B")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert v.accept and v.tier == "strong"

def test_seed_not_returned_reject():
    retr = [_r("c2", 0.5, "A"), _r("c3", 0.4, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert not v.accept and v.reason == "seed_not_returned"

def test_single_chunk_full_accept():
    retr = [_r("seed", 0.4, "A")]  # 只有 1 条，但 full 且 seed_score≥0.35
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert v.accept and v.tier == "single_chunk_full"

def test_too_few_chunks_reject_when_not_single_full():
    retr = [_r("seed", 0.4, "A")]  # 1 条但 judge=partial → 不符合 single_chunk_full
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="partial", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert not v.accept and v.reason == "too_few_chunks"

def test_partial_ok_within_quota():
    retr = [_r("seed", 0.32, "A"), _r("c2", 0.31, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="partial", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert v.accept and v.tier == "partial_ok"

def test_partial_quota_full_reject():
    retr = [_r("seed", 0.32, "A"), _r("c2", 0.31, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="partial", judge_conflict=False, cfg=CFG,
                 partial_count=8, target=100)  # 8 >= floor(0.08*100)=8
    assert not v.accept and v.reason == "partial_quota_full"

def test_evidence_conflict_reject():
    retr = [_r("seed", 0.5, "A"), _r("c2", 0.46, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=True, cfg=CFG,
                 partial_count=0, target=100)
    assert not v.accept and v.reason == "evidence_conflict"

def test_low_score_reject():
    retr = [_r("seed", 0.1, "A"), _r("c2", 0.1, "A")]
    v = evaluate(seed_chunk_id="seed", seed_section="A", retrieved=retr,
                 judge_label="full", judge_conflict=False, cfg=CFG,
                 partial_count=0, target=100)
    assert not v.accept and v.reason == "low_score"

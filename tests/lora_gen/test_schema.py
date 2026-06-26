from lora_gen.schema import Sample, SampleMeta, Rejected

def test_sample_record_strict_keys():
    s = Sample(instruction="i", input="q", output="a")
    assert s.to_record() == {"instruction": "i", "input": "q", "output": "a"}

def test_meta_roundtrip_keys():
    m = SampleMeta(
        qid="Q1", model_id="M", model_display="Md", doc_type="owner_manual",
        section_path="A>B", task_type="直接问答", seed_chunk_id="c1",
        seed_preview="前50字", seed_score=0.4, evidence_chunk_ids=["c1", "c2"],
        evidence_previews=["p1", "p2"], retrieval_scores=[0.4, 0.3], max_score=0.4,
        seed_rank=1, same_section_count=2, evidence_sufficiency="full",
        accept_tier="strong", backend="cloud", gen_question_raw="{}",
    )
    rec = m.to_record()
    assert rec["accept_tier"] == "strong"
    assert rec["evidence_chunk_ids"] == ["c1", "c2"]

def test_rejected_defaults():
    r = Rejected(qid="Q1", model_id="M", section_path="A", task_type="直接问答",
                 seed_chunk_id="c1", seed_preview="p", reject_stage="quality",
                 reject_reason="vehicle_conflict", reject_detail="found M2")
    assert r.to_record()["question"] == ""

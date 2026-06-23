from lora_gen.pipeline import (
    make_qid, load_prior, _append_jsonl, ACCEPTED_FILE, REJECTED_FILE,
)
from lora_gen.quality import normalize_question
from lora_gen.schema import Rejected, Sample, SampleMeta


def test_make_qid_stable():
    a = make_qid("问界M9-2026款增程版", "chunk-7", "步骤指导")
    b = make_qid("问界M9-2026款增程版", "chunk-7", "步骤指导")
    assert a == b and a.startswith("Q")


def test_load_prior_missing_dir(tmp_path):
    st = load_prior(tmp_path)
    assert st.done == set()
    assert st.result.accepted == [] and st.result.rejected == []
    assert st.partial_count == 0 and st.seen_norm == set()


def test_load_prior_repopulates_result_and_state(tmp_path):
    # 一条 accepted（partial_ok）+ 一条 rejected，验证 resume 回放重建
    meta = SampleMeta(
        qid="Q1", model_id="M", model_display="M", doc_type="owner_manual",
        section_path="A", task_type="直接问答", seed_chunk_id="c1", seed_preview="p",
        seed_score=0.4, evidence_chunk_ids=["c1"], evidence_previews=["p"],
        retrieval_scores=[0.4], max_score=0.4, seed_rank=1, same_section_count=1,
        evidence_sufficiency="partial", accept_tier="partial_ok", backend="cloud",
        gen_question_raw="{}",
    )
    sample = Sample(instruction="i", input="问界M9空调怎么开", output="按AUTO键")
    _append_jsonl(tmp_path / ACCEPTED_FILE,
                  {"qid": "Q1", "meta": meta.to_record(), "sample": sample.to_record()})
    rej = Rejected(
        qid="Q2", model_id="M", section_path="A", task_type="直接问答",
        seed_chunk_id="c2", seed_preview="p", reject_stage="answerability",
        reject_reason="low_score", reject_detail="", question="问界M9胎压是多少",
    )
    _append_jsonl(tmp_path / REJECTED_FILE, rej.to_record())

    st = load_prior(tmp_path)
    # 旧产出完整存活
    assert len(st.result.accepted) == 1 and st.result.accepted[0].output == "按AUTO键"
    assert len(st.result.metas) == 1 and st.result.metas[0].qid == "Q1"
    assert st.result.task_pairs[0][1] == "直接问答"
    assert len(st.result.rejected) == 1 and st.result.rejected[0].reject_reason == "low_score"
    # done 覆盖 accepted + rejected
    assert st.done == {"Q1", "Q2"}
    # partial_ok 计入配额
    assert st.partial_count == 1
    # 去重集合包含 accepted 与 rejected 两侧的问题
    assert normalize_question("问界M9空调怎么开") in st.seen_norm
    assert normalize_question("问界M9胎压是多少") in st.seen_norm

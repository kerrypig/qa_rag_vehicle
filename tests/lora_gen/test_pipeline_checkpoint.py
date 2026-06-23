from lora_gen.pipeline import load_done_qids, append_checkpoint, make_qid


def test_make_qid_stable():
    a = make_qid("问界M9-2026款增程版", "chunk-7", "步骤指导")
    b = make_qid("问界M9-2026款增程版", "chunk-7", "步骤指导")
    assert a == b and a.startswith("Q")


def test_checkpoint_roundtrip(tmp_path):
    cp = tmp_path / "checkpoint.jsonl"
    append_checkpoint(cp, "Q1", "accepted")
    append_checkpoint(cp, "Q2", "rejected:low_score")
    done = load_done_qids(cp)
    assert done == {"Q1", "Q2"}


def test_load_done_missing_file(tmp_path):
    assert load_done_qids(tmp_path / "nope.jsonl") == set()

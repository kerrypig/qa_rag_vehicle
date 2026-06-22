from dataset_gen.checkpoint import (
    append_meta,
    load_done_qids,
    load_samples,
    write_samples,
)


def test_write_then_load_roundtrip(tmp_path):
    p = tmp_path / "out.json"
    samples = [{"instruction": "i", "input": "q", "output": "中文输出"}]
    write_samples(str(p), samples)
    assert load_samples(str(p)) == samples
    # 中文不转义
    assert "中文输出" in p.read_text(encoding="utf-8")


def test_load_samples_missing_returns_empty(tmp_path):
    assert load_samples(str(tmp_path / "nope.json")) == []


def test_load_samples_corrupt_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_samples(str(p)) == []


def test_append_meta_and_load_done_qids(tmp_path):
    m = tmp_path / "meta.jsonl"
    append_meta(str(m), {"QID": "Q1", "task_type": "直接问答"})
    append_meta(str(m), {"QID": "Q2", "task_type": "步骤指导"})
    assert load_done_qids(str(m)) == {"Q1", "Q2"}


def test_load_done_qids_missing_returns_empty(tmp_path):
    assert load_done_qids(str(tmp_path / "nope.jsonl")) == set()


def test_load_done_qids_skips_bad_lines(tmp_path):
    m = tmp_path / "meta.jsonl"
    m.write_text('{"QID": "Q1"}\n{bad}\n\n{"no_qid": 1}\n', encoding="utf-8")
    assert load_done_qids(str(m)) == {"Q1"}

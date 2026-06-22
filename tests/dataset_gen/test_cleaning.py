import pandas as pd

from dataset_gen.cleaning import (
    dedup_by_question,
    extract_input_text,
    filter_by_keywords,
    load_rows,
    sample_rows,
)


def test_extract_input_prefers_question():
    assert extract_input_text({"Question": "胎压灯亮了", "Dialogue": "x|y"}) == "胎压灯亮了"


def test_extract_input_falls_back_to_dialogue_first_segment():
    row = {"Question": "", "Dialogue": "技师说蓝牙怎么连|车主说好的"}
    assert extract_input_text(row) == "技师说蓝牙怎么连"


def test_filter_by_keywords_keeps_only_matches():
    rows = [
        {"Question": "胎压灯亮了怎么办"},
        {"Question": "发动机正时皮带多久换"},
        {"Question": "空调不制冷"},
    ]
    kept = filter_by_keywords(rows, ["胎压", "空调"])
    assert [r["Question"] for r in kept] == ["胎压灯亮了怎么办", "空调不制冷"]


def test_dedup_by_question():
    rows = [{"Question": "蓝牙怎么连"}, {"Question": "蓝牙怎么连"}, {"Question": "雷达报警"}]
    assert len(dedup_by_question(rows)) == 2


def test_sample_rows_deterministic_with_seed():
    rows = [{"Question": str(i)} for i in range(20)]
    a = sample_rows(rows, seed=42)
    b = sample_rows(rows, seed=42)
    assert a == b
    assert sorted(r["Question"] for r in a) == sorted(r["Question"] for r in rows)


def test_load_rows_reads_gb18030(tmp_path):
    csv = tmp_path / "t.csv"
    df = pd.DataFrame({"QID": ["Q1"], "Question": ["空调不制冷"], "Dialogue": [""]})
    df.to_csv(csv, index=False, encoding="gb18030")
    rows = load_rows(str(csv), encoding="gb18030")
    assert rows[0]["Question"] == "空调不制冷"
    assert rows[0]["QID"] == "Q1"

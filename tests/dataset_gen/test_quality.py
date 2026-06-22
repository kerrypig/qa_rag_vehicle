from dataset_gen.quality import (
    has_banned_phrase,
    is_weak_retrieval,
    max_score,
    validate_sample,
)


def test_max_score_empty():
    assert max_score({}) == 0.0


def test_weak_when_too_few_chunks():
    assert is_weak_retrieval(["d"], {"a": 0.9}, min_chunks=2, min_score=0.3) is True


def test_weak_when_score_below_threshold():
    assert is_weak_retrieval(["d1", "d2"], {"a": 0.1, "b": 0.2}, min_chunks=2, min_score=0.3) is True


def test_not_weak_when_enough_and_strong():
    assert is_weak_retrieval(["d1", "d2"], {"a": 0.5}, min_chunks=2, min_score=0.3) is False


def test_banned_phrase_detected():
    assert has_banned_phrase("根据手册，应当检查胎压") is True
    assert has_banned_phrase("请检查胎压并充气") is False


def test_validate_ok():
    obj = {"instruction": "回答问题", "input": "胎压灯亮了？", "output": "请检查并充气。"}
    ok, msg = validate_sample(obj)
    assert ok is True and msg == ""


def test_validate_missing_field():
    ok, msg = validate_sample({"instruction": "x", "input": "y"})
    assert ok is False and "output" in msg


def test_validate_empty_field():
    ok, msg = validate_sample({"instruction": "x", "input": "  ", "output": "z"})
    assert ok is False and "input" in msg


def test_validate_extra_field():
    obj = {"instruction": "x", "input": "y", "output": "z", "note": "多余"}
    ok, msg = validate_sample(obj)
    assert ok is False and "多余字段" in msg


def test_validate_banned_in_output():
    obj = {"instruction": "x", "input": "y", "output": "根据手册请检查"}
    ok, msg = validate_sample(obj)
    assert ok is False and "禁语" in msg


def test_validate_non_dict():
    ok, msg = validate_sample(["not", "a", "dict"])
    assert ok is False

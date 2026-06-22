import pytest

from dataset_gen.backends import extract_json


def test_plain_json():
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_fenced_json():
    text = "```json\n{\"a\": 1}\n```"
    assert extract_json(text) == {"a": 1}


def test_json_with_leading_prose():
    text = '好的，结果如下：{"instruction": "i", "input": "q", "output": "o"}'
    assert extract_json(text) == {"instruction": "i", "input": "q", "output": "o"}


def test_invalid_json_raises():
    with pytest.raises(ValueError):
        extract_json("这里没有任何 JSON")

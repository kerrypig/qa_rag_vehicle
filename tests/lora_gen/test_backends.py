import pytest
from lora_gen.backends import extract_json, GenerationError

def test_extract_plain_json():
    assert extract_json('{"output": "ok"}') == {"output": "ok"}

def test_extract_fenced_json():
    raw = "```json\n{\"label\": \"full\", \"conflict\": false}\n```"
    assert extract_json(raw) == {"label": "full", "conflict": False}

def test_extract_with_surrounding_text():
    raw = "好的，结果是 {\"question\": \"空调怎么开\"} 以上。"
    assert extract_json(raw) == {"question": "空调怎么开"}

def test_extract_garbage_raises():
    with pytest.raises(GenerationError):
        extract_json("没有任何 JSON 内容")

def test_extract_broken_json_raises():
    with pytest.raises(GenerationError):
        extract_json('{"output": ')

def test_extract_nested_braces_not_greedy():
    # 贪婪正则会把后面的 } 一起吞掉；brace-balanced 应只取第一个完整对象
    raw = '前言 {"output": {"a": 1}} 结尾 {"b": 2}'
    assert extract_json(raw) == {"output": {"a": 1}}

def test_extract_fenced_priority_over_braces():
    raw = '忽略这个 {x} ```json\n{"label": "full"}\n``` 尾部'
    assert extract_json(raw) == {"label": "full"}

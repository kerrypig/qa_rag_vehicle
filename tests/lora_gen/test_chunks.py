import json
from lora_gen.chunks import (
    Chunk, load_corpus, is_usable_chunk, is_blacklisted, section_to_task_type,
)

def test_section_to_task_type():
    assert section_to_task_type("用户提示>故障警示灯说明") == "故障分析"
    assert section_to_task_type("附录>术语与缩略语") == "术语解释"
    assert section_to_task_type("维护保养>更换雨刮") == "步骤指导"
    assert section_to_task_type("行驶安全>儿童安全座椅") == "安全提醒"
    assert section_to_task_type("车辆控制>空调") == "直接问答"

def test_blacklist():
    bl = ["前言", "目录", "术语"]
    assert is_blacklisted("文档前言", bl) is True
    assert is_blacklisted("车辆控制>空调", bl) is False

def test_is_usable_chunk():
    bl = ["目录"]
    short = Chunk("c1", "太短", "M", "owner_manual", "车辆控制>空调", 1)
    good = Chunk("c2", "正常内容" * 60, "M", "owner_manual", "车辆控制>空调", 1)
    toc = Chunk("c3", "正常内容" * 60, "M", "owner_manual", "目录", 1)
    assert is_usable_chunk(short, min_chars=200, blacklist=bl) is False
    assert is_usable_chunk(good, min_chars=200, blacklist=bl) is True
    assert is_usable_chunk(toc, min_chars=200, blacklist=bl) is False

def test_load_corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    payload = {
        "chunk_ids": ["c1"],
        "texts": ["内容"],
        "metadatas": [{"vehicle_model": "M9", "doc_type": "owner_manual",
                       "section_path": "A>B", "page": 3}],
    }
    (d / "bm25_corpus.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    chunks = load_corpus(d)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1" and chunks[0].vehicle_model == "M9"
    assert chunks[0].section_path == "A>B"

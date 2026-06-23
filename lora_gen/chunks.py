"""corpus 加载、可用性过滤、section→task_type 映射。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 顺序敏感：先匹配到的规则胜出。
TASK_SECTION_RULES: list[tuple[tuple[str, ...], str]] = [
    (("故障", "警示", "报警", "异常", "警告灯", "提示灯"), "故障分析"),
    (("名词", "定义", "简介", "术语", "缩略", "释义"), "术语解释"),
    (("检查", "保养", "操作", "更换", "安装", "加注", "清洗", "调节"), "步骤指导"),
    (("安全", "儿童", "安全带", "气囊", "乘员", "警告"), "安全提醒"),
]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    vehicle_model: str
    doc_type: str
    section_path: str
    page: object


def section_to_task_type(section_path: str) -> str:
    for keys, tt in TASK_SECTION_RULES:
        if any(k in section_path for k in keys):
            return tt
    return "直接问答"


def is_blacklisted(section_path: str, blacklist: list[str]) -> bool:
    return any(b in section_path for b in blacklist)


def is_usable_chunk(chunk: Chunk, *, min_chars: int, blacklist: list[str]) -> bool:
    text = chunk.text.strip()
    if len(text) < min_chars:
        return False
    if is_blacklisted(chunk.section_path, blacklist):
        return False
    return True


def load_corpus(index_path: Path) -> list[Chunk]:
    data = json.loads((index_path / "bm25_corpus.json").read_text(encoding="utf-8"))
    out: list[Chunk] = []
    for cid, text, meta in zip(
        data["chunk_ids"], data["texts"], data["metadatas"], strict=True
    ):
        out.append(
            Chunk(
                chunk_id=cid,
                text=text,
                vehicle_model=meta.get("vehicle_model", ""),
                doc_type=meta.get("doc_type", ""),
                section_path=meta.get("section_path", ""),
                page=meta.get("page", "?"),
            )
        )
    return out


def usable_chunks_by_model(
    chunks: list[Chunk], *, min_chars: int, blacklist: list[str]
) -> dict[str, list[Chunk]]:
    by_model: dict[str, list[Chunk]] = {}
    for c in chunks:
        if is_usable_chunk(c, min_chars=min_chars, blacklist=blacklist):
            by_model.setdefault(c.vehicle_model, []).append(c)
    return by_model

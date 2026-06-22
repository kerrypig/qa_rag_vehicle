"""断点续传：原子写 JSON 数组、追加 meta.jsonl、读已完成 QID。"""
from __future__ import annotations

import json
import os
from pathlib import Path


def load_done_qids(meta_path: str) -> set[str]:
    """从 meta.jsonl 收集已完成 QID；文件缺失或坏行均安全跳过。"""
    p = Path(meta_path)
    if not p.exists():
        return set()
    done: set[str] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                qid = json.loads(line)["QID"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            done.add(qid)
    return done


def load_samples(json_path: str) -> list[dict]:
    """读现有输出数组；缺失或损坏返回空列表，便于重头累积。"""
    p = Path(json_path)
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def write_samples(json_path: str, samples: list[dict]) -> None:
    """原子写：先写 .tmp 再 os.replace，避免中断损坏输出。"""
    p = Path(json_path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def append_meta(meta_path: str, entry: dict) -> None:
    """逐行追加溯源记录（JSONL）。"""
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

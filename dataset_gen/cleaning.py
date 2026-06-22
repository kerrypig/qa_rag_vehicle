"""CSV 载入、input 文本提取、关键词过滤、去重、随机采样。"""
from __future__ import annotations

import random

import pandas as pd


def load_rows(csv_path: str, encoding: str = "gb18030") -> list[dict]:
    """按指定编码读 CSV，失败则依次回退 gbk / utf-8。空值填空串。"""
    last_err: Exception | None = None
    for enc in [encoding, "gbk", "utf-8"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc, dtype=str)
            df = df.fillna("")
            return df.to_dict(orient="records")
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def extract_input_text(row: dict) -> str:
    """优先用 Question；为空时回退取 Dialogue 第一段（以 '|' 分隔）。"""
    q = str(row.get("Question", "")).strip()
    if q:
        return q
    dlg = str(row.get("Dialogue", "")).strip()
    return dlg.split("|")[0].strip() if dlg else ""


def filter_by_keywords(rows: list[dict], keywords: list[str]) -> list[dict]:
    """保留 input 文本命中任一关键词的行。"""
    return [r for r in rows if any(kw in extract_input_text(r) for kw in keywords)]


def dedup_by_question(rows: list[dict]) -> list[dict]:
    """按 input 文本去重，保留首次出现；丢弃 input 为空的行。"""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        q = extract_input_text(r)
        if q and q not in seen:
            seen.add(q)
            out.append(r)
    return out


def sample_rows(rows: list[dict], seed: int) -> list[dict]:
    """用固定种子打散整池，返回全部（调用方按 sample_size/target_size 控制截断）。"""
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    return pool

"""检索强度判定、禁语过滤、样本结构校验。"""
from __future__ import annotations

BANNED_PHRASES = [
    "根据手册", "根据资料", "根据上述", "根据提供的", "根据以上",
    "资料显示", "手册显示", "手册中", "资料中", "如上所述",
    "作为AI", "作为人工智能", "作为助手", "作为一个",
    "[资料", "资料1", "资料2", "资料3",
]

_ALLOWED_KEYS = {"instruction", "input", "output"}


def max_score(scores: dict[str, float]) -> float:
    return max(scores.values()) if scores else 0.0


def is_weak_retrieval(docs, scores: dict[str, float], *, min_chunks: int, min_score: float) -> bool:
    """chunk 数不足，或最高向量相似度低于阈值，判为弱检索（应跳过）。"""
    if len(docs) < min_chunks:
        return True
    return max_score(scores) < min_score


def has_banned_phrase(text: str) -> bool:
    return any(p in text for p in BANNED_PHRASES)


def validate_sample(obj) -> tuple[bool, str]:
    """校验生成结果：必须是仅含三键的 dict，三字段非空，output 无禁语。"""
    if not isinstance(obj, dict):
        return False, "不是 JSON 对象"
    for key in ("instruction", "input", "output"):
        v = obj.get(key)
        if not isinstance(v, str) or not v.strip():
            return False, f"字段缺失或为空: {key}"
    extra = set(obj.keys()) - _ALLOWED_KEYS
    if extra:
        return False, f"含多余字段: {sorted(extra)}"
    if has_banned_phrase(obj["output"]):
        return False, "output 命中禁语"
    return True, ""

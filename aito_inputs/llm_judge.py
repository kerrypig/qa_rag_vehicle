"""单次 LLM 可答性判定：把 top chunk 拼进一个 prompt，问一次「能/不能」。

相比逐 chunk 的 verify_chunks（N 次/条），这里 1 次/条，约快 5 倍，且问的是
「这些资料能否回答该问题」而非「单个 chunk 相不相关」，更贴合可答性筛选目标。
默认关闭（--verify 开启），保持快速；开启时作为向量分之上的语义复核。
"""
from __future__ import annotations

_PROMPT = (
    "你是问界车主手册问答的可答性判官。\n"
    "判断下面的【手册资料】是否足以回答【用户问题】。\n"
    "标准：资料必须直接覆盖问题的核心对象与动作才算「能」；"
    "若资料只是泛泛相关、答非所问，或问题本身需要现场维修诊断/手册之外的信息，则算「不能」。\n\n"
    "【用户问题】{question}\n\n"
    "【手册资料】\n{context}\n\n"
    "只回答两个字：能 或 不能"
)


def parse_verdict(text: str) -> bool:
    """解析模型输出为 能=True / 不能=False；不确定时保守判 False。"""
    ans = (text or "").strip()
    if "不能" in ans or "否" in ans:
        return False
    if ans.startswith("能") or "能" in ans or "可以" in ans:
        return True
    return False


def build_context(docs, max_chunks: int = 4, max_chars: int = 300) -> str:
    parts = []
    for i, d in enumerate(docs[:max_chunks], start=1):
        meta = getattr(d, "metadata", {}) or {}
        section = meta.get("section_path") or meta.get("section") or ""
        content = (getattr(d, "page_content", "") or "")[:max_chars]
        parts.append(f"[{i}] {section}：{content}")
    return "\n".join(parts)


def can_answer(
    question: str,
    docs,
    *,
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
    timeout_s: int = 60,
) -> bool:
    """单次 Ollama 调用判断资料能否回答问题。无 docs 直接 False。"""
    if not docs:
        return False
    import ollama

    prompt = _PROMPT.format(question=question, context=build_context(docs))
    resp = ollama.Client(timeout=timeout_s).generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": 8},
    )
    return parse_verdict(resp["response"])

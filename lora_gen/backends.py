"""LLM 后端抽象：cloud(qwen3.6-plus) / local(ollama)，及 JSON 抽取。"""
from __future__ import annotations

import json
import re
from typing import Protocol


class GenerationError(Exception):
    """JSON 解析失败 → reject_reason=json_parse_failed。"""


class Backend(Protocol):
    def complete(self, prompt: str) -> str: ...


class CloudBackend:
    def __init__(self, config):
        from generate.qwen_client import QwenClient

        self.client = QwenClient(config)

    def complete(self, prompt: str) -> str:
        text, _ = self.client.chat(prompt)
        return text


class LocalBackend:
    def __init__(self, model: str = "qwen2.5:7b"):
        import ollama

        self._ollama = ollama
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self._ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.1},
        )
        return resp["message"]["content"]


def make_backend(name: str, config, dg_config) -> Backend:
    if name == "cloud":
        return CloudBackend(config)
    return LocalBackend(dg_config.get("local_model", default="qwen2.5:7b"))


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _balanced_object(text: str) -> str | None:
    """返回第一个括号平衡的 {...} 子串（考虑字符串内转义），无则 None。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def extract_json(text: str) -> dict:
    # 1) fenced ```json {...}``` 优先
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else _balanced_object(text)
    if candidate is None:
        raise GenerationError(f"json_parse_failed: 无 JSON 片段: {text[:80]!r}")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise GenerationError(f"json_parse_failed: {e}") from e

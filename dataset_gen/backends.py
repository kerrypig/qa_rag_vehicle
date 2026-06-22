"""生成后端：本地 Ollama（format=json）/ 云端 DashScope，统一返回 dict。"""
from __future__ import annotations

import json
import re
from typing import Protocol

import ollama


def extract_json(text: str) -> dict:
    """从模型输出中提取首个 JSON 对象。支持 ```json``` 围栏与前置说明文字。

    无法解析时抛 ValueError（json.JSONDecodeError 是其子类）。
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        candidate = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"输出中未找到 JSON 对象: {text[:80]!r}")
        candidate = m.group(0)
    return json.loads(candidate)


class Backend(Protocol):
    def generate(self, prompt: str) -> dict: ...


class LocalOllamaBackend:
    """本地 Ollama，启用 format='json' 强约束。"""

    def __init__(self, model: str, temperature: float, num_predict: int, timeout_s: int):
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.client = ollama.Client(timeout=timeout_s)

    def generate(self, prompt: str) -> dict:
        resp = self.client.generate(
            model=self.model,
            prompt=prompt,
            format="json",
            options={"temperature": self.temperature, "num_predict": self.num_predict},
        )
        return extract_json(resp["response"])


class CloudBackend:
    """云端 DashScope：复用 QwenClient 的凭据与模型，但走 chat.completions 并
    关闭思考模式（enable_thinking=False）。数据改写不需要推理，关掉后同模型同质量
    却快约 40 倍（实测 53s→1.2s）。不走 QwenClient.chat 的 Responses API，
    以免影响仍需思考的 RAG 问答路径。"""

    def __init__(self, qwen_client):
        self._openai = qwen_client.client
        self.model = qwen_client.model
        self.temperature = qwen_client.temperature

    def generate(self, prompt: str) -> dict:
        resp = self._openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            extra_body={"enable_thinking": False},
        )
        return extract_json(resp.choices[0].message.content)


def make_backend(ds_cfg: dict, app_config) -> Backend:
    if ds_cfg.get("backend") == "cloud":
        from generate.qwen_client import QwenClient

        return CloudBackend(QwenClient(app_config))
    lc = ds_cfg["local"]
    return LocalOllamaBackend(
        model=lc["model"],
        temperature=lc["temperature"],
        num_predict=lc["num_predict"],
        timeout_s=lc["timeout_s"],
    )

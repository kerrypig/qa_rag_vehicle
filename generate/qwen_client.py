"""DashScope Responses API 客户端。"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class QwenClient:
    def __init__(self, config):
        gen = config.raw["generation"]
        api_key = os.getenv(gen.get("api_key_env", "DASHSCOPE_API_KEY"))
        base_url = os.getenv(gen.get("base_url_env", "DASHSCOPE_BASE_URL"))
        if not api_key:
            raise EnvironmentError(
                "未找到 DASHSCOPE_API_KEY。请复制 .env.example 为 .env 并填入密钥。"
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = gen.get("model", "qwen3.6-plus")
        self.temperature = gen.get("temperature", 0.1)
        self.max_tokens = gen.get("max_tokens", 2048)

    def chat(self, user_input: str, previous_response_id: str | None = None) -> tuple[str, str]:
        kwargs = {
            "model": self.model,
            "input": user_input,
        }
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        response = self.client.responses.create(**kwargs)
        return response.output_text.strip(), response.id

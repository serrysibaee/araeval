"""GPT (or any OpenAI-compatible chat model) as LLM judge. Credentials: OPENAI_API_KEY."""
from __future__ import annotations

import os

from openai import OpenAI

from .common import parse_json


class OpenAIJudge:
    def __init__(self, model: str, base_url: str | None = None, max_tokens: int = 4000):
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY"))
        self.model, self.max_tokens = model, max_tokens

    def __call__(self, system: str, prompt: str) -> dict | None:
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_completion_tokens=self.max_tokens,
            )
        except Exception as e:                       # judge failures are recorded per item, never fatal
            return {"error": type(e).__name__}
        return parse_json(r.choices[0].message.content or "")

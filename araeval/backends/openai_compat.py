"""OpenAI-compatible chat endpoint (a `vllm serve` server, TGI, or a hosted API model).

Generation only: multiple-choice tasks fall back to generate-and-extract-letter mode.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from .base import Backend, truncate_at_stop


class OpenAICompatBackend(Backend):
    supports_loglikelihood = False

    def __init__(self, spec):
        super().__init__(spec)
        from openai import OpenAI
        self.client = OpenAI(base_url=spec.api_base, api_key=os.environ.get(spec.api_key_env, "EMPTY"))

    def format_prompt(self, user_text):
        return user_text                       # the server applies the chat template

    def _one(self, prompt, max_new_tokens, stop):
        msgs = ([{"role": "system", "content": self.spec.system_prompt}] if self.spec.system_prompt else []) + \
               [{"role": "user", "content": prompt}]
        r = self.client.chat.completions.create(model=self.spec.path, messages=msgs, temperature=0,
                                                max_tokens=max_new_tokens, stop=stop or None)
        return truncate_at_stop(r.choices[0].message.content or "", stop)

    def generate(self, prompts, max_new_tokens=256, stop=None):
        with ThreadPoolExecutor(max_workers=16) as ex:
            return list(ex.map(lambda p: self._one(p, max_new_tokens, stop), prompts))

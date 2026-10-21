"""Claude as LLM judge (Anthropic SDK). Credentials: ANTHROPIC_API_KEY or an `ant auth login` profile."""
from __future__ import annotations

import anthropic

from .common import parse_json


class AnthropicJudge:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 4000):
        self.client = anthropic.Anthropic(max_retries=5)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def __call__(self, system: str, prompt: str) -> dict | None:
        try:
            # Server-side fallbacks re-run a refused request on a fallback model inside the same call
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.BadRequestError as e:
            return {"error": f"bad_request: {e.message}"}
        except anthropic.RateLimitError:
            return {"error": "rate_limited"}
        except anthropic.APIStatusError as e:
            return {"error": f"api_status_{e.status_code}"}
        except anthropic.APIConnectionError:
            return {"error": "connection"}
        if resp.stop_reason == "refusal":
            return {"error": "refusal"}
        text = "".join(b.text for b in resp.content if b.type == "text")
        return parse_json(text)

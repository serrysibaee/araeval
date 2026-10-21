"""Backend interface. A backend wraps one loaded model."""
from __future__ import annotations

from ..schema import ModelSpec


class Backend:
    supports_loglikelihood = True

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    # --- prompt formatting ---------------------------------------------------------
    def format_prompt(self, user_text: str) -> str:
        """Wrap a user turn in the model's chat template (instruct models) or return it raw (base)."""
        tok = getattr(self, "tokenizer", None)
        if not self.spec.is_instruct or tok is None or not getattr(tok, "chat_template", None):
            return user_text
        msgs = []
        if self.spec.system_prompt:
            msgs.append({"role": "system", "content": self.spec.system_prompt})
        msgs.append({"role": "user", "content": user_text})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # --- scoring / generation --------------------------------------------------------
    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[float]:
        """Sum of log-probabilities of each continuation given its (already formatted) context."""
        raise NotImplementedError

    def generate(self, prompts: list[str], max_new_tokens: int = 256, stop: list[str] | None = None) -> list[str]:
        """Greedy generation from already formatted prompts."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def truncate_at_stop(text: str, stop: list[str] | None) -> str:
    for s in stop or []:
        if s and s in text:
            text = text[: text.index(s)]
    return text

"""Judge plumbing shared by providers: JSON parsing, on-disk cache, provider factory."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"error": "no_json", "raw": text[:300]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"error": "bad_json", "raw": text[:300]}


class JudgeCache:
    """Append-only JSONL cache keyed by a hash of (judge model, rubric, question, answer, reference)."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["value"]
                except Exception:
                    pass

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def make_judge(cfg: dict):
    """cfg: {provider: anthropic|openai, model: ..., effort: ..., base_url: ...}"""
    provider = cfg.get("provider", "anthropic")
    if provider == "anthropic":
        from .anthropic_judge import AnthropicJudge
        return AnthropicJudge(model=cfg.get("model", "claude-opus-5-5"), effort=cfg.get("effort", "medium"))
    if provider == "openai":
        from .openai_judge import OpenAIJudge
        if not cfg.get("model"):
            raise ValueError("judge.model is required for the openai provider")
        return OpenAIJudge(model=cfg["model"], base_url=cfg.get("base_url"))
    raise ValueError(f"unknown judge provider {provider!r}")

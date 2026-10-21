"""Deterministic fake backend for pipeline tests (no model download, no GPU)."""
from __future__ import annotations

import hashlib

from .base import Backend


def _h(*s) -> int:
    return int(hashlib.md5("|".join(s).encode("utf-8")).hexdigest(), 16)


class DummyBackend(Backend):
    def loglikelihood(self, requests):
        return [-(_h(c, t) % 1000) / 100.0 for c, t in requests]

    def generate(self, prompts, max_new_tokens=256, stop=None):
        return [["أ", "B", "الإجابة: ج", "42", "لا أعرف"][_h(p) % 5] for p in prompts]

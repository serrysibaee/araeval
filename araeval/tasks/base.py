"""Task base class and shared prompt helpers."""
from __future__ import annotations

import random

from ..schema import Sample, TaskSpec
from ..textnorm import ARABIC_LETTERS, LATIN_LETTERS

DEFAULT_MCQ_INSTRUCTION = "السؤال التالي سؤال اختيار من متعدد. اختر الإجابة الصحيحة."
DEFAULT_GEN_INSTRUCTION = ""


class Task:
    def __init__(self, spec: TaskSpec, run_cfg: dict):
        self.spec = spec
        self.cfg = run_cfg
        self.letters = ARABIC_LETTERS if run_cfg.get("letters", "arabic") == "arabic" else LATIN_LETTERS

    def run(self, backend, samples: list[Sample], shots: list[Sample]) -> tuple[list[dict], dict]:
        """Returns (per-sample records, {metric: value}). Implemented by subclasses."""
        raise NotImplementedError

    @property
    def primary_metric(self) -> str:
        raise NotImplementedError

    def pick_shots(self, pool: list[Sample], sample: Sample) -> list[Sample]:
        k = self.spec.num_fewshot
        if not k or not pool:
            return []
        rng = random.Random(f"{self.spec.name}-{sample.id}")
        cands = [s for s in pool if s.id != sample.id]
        return rng.sample(cands, min(k, len(cands)))

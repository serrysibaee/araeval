"""Free-form generation scored against references (em, numeric_em, f1, contains, rouge, chrf, bleu)."""
from __future__ import annotations

from ..metrics import SAMPLE_METRICS, corpus_scores, mean_stderr, rouge
from .base import Task

BASE_STOP = ["\n\nالسؤال:", "\n\n\n", "\nالسؤال:"]


class GenerateTask(Task):
    @property
    def primary_metric(self):
        return self.spec.primary_metric or (self.spec.metrics[0] if self.spec.metrics else "em")

    def _body(self, s, answer=None, qa=True):
        parts = []
        if s.context:
            parts.append(f"النص: {s.context}")
        parts.append(f"السؤال: {s.question}" if qa else s.question)
        text = "\n".join(parts)
        if qa:
            text += "\nالإجابة:" + (f" {answer}" if answer is not None else "")
        return text

    def build_prompt(self, s, shots, qa=True):
        """qa=True frames items as السؤال/الإجابة (needed by base models and few-shot prompts)."""
        blocks = [self.spec.instruction] if self.spec.instruction else []
        for x in shots:
            g = x.gold[0] if isinstance(x.gold, list) else x.gold
            blocks.append(self._body(x, g, qa=True))
        blocks.append(self._body(s, qa=qa))
        return "\n\n".join(blocks)

    def run(self, backend, samples, shots_pool):
        qa = bool(self.spec.num_fewshot) or not backend.spec.is_instruct
        prompts = [backend.format_prompt(self.build_prompt(s, self.pick_shots(shots_pool, s), qa=qa)) for s in samples]
        stop = None if backend.spec.is_instruct else BASE_STOP
        outs = backend.generate(prompts, max_new_tokens=self.spec.max_new_tokens, stop=stop)
        outs = [o.strip() for o in outs]
        names = self.spec.metrics or ["em"]
        records = []
        for s, out in zip(samples, outs):
            rec = {"id": s.id, "group": s.group, "output": out, "gold": s.gold}
            for n in names:
                if n in SAMPLE_METRICS:
                    rec[n] = SAMPLE_METRICS[n](out, s.gold)
            if "rouge" in names:
                rec.update(rouge(out, s.gold))
            records.append(rec)
        metrics = {}
        keys = [n for n in names if n in SAMPLE_METRICS] + (["rouge1", "rouge2", "rougeL"] if "rouge" in names else [])
        for k in keys:
            metrics[k], metrics[k + "_stderr"] = mean_stderr([r[k] for r in records])
        metrics.update(corpus_scores(outs, [s.gold for s in samples], names))
        return records, metrics

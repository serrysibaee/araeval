"""Open-ended generation scored by an LLM judge against a reference answer (instruct models only)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..data import stable_hash
from ..judges.rubrics import RUBRICS, SYSTEM
from ..metrics import mean_stderr
from .base import Task


class JudgeTask(Task):
    def __init__(self, spec, run_cfg, judge=None, cache=None):
        super().__init__(spec, run_cfg)
        self.judge, self.cache = judge, cache

    @property
    def primary_metric(self):
        return self.spec.primary_metric or "score"

    def run(self, backend, samples, shots_pool):
        if self.judge is None:
            raise RuntimeError("judge task requires a configured judge (see `judge:` in the run config)")
        template, scorer = RUBRICS[self.spec.judge_rubric or "reference_1to5"]
        prompts = [backend.format_prompt((self.spec.instruction + "\n\n" if self.spec.instruction else "") + s.question)
                   for s in samples]
        outs = [o.strip() for o in backend.generate(prompts, max_new_tokens=self.spec.max_new_tokens)]
        jcfg = self.cfg.get("judge", {})

        def grade(pair):
            s, out = pair
            ref = s.gold[0] if isinstance(s.gold, list) else s.gold
            key = stable_hash(jcfg.get("provider"), jcfg.get("model"), self.spec.judge_rubric, s.question, out, ref)
            hit = self.cache.get(key) if self.cache else None
            if hit is None:
                hit = self.judge(SYSTEM, template.format(question=s.question, reference=ref, answer=out or "(empty)"))
                if self.cache and hit and "error" not in hit:
                    self.cache.put(key, hit)
            return hit

        with ThreadPoolExecutor(max_workers=jcfg.get("concurrency", 8)) as ex:
            verdicts = list(ex.map(grade, zip(samples, outs)))
        records = []
        for s, out, v in zip(samples, outs, verdicts):
            rec = {"id": s.id, "group": s.group, "output": out, "judge": v}
            try:
                rec.update(scorer(v))
            except Exception:
                rec["score"] = None
                rec["judge_error"] = (v or {}).get("error", "unscorable")
            records.append(rec)
        metrics = {}
        keys = {k for r in records for k in r if k in ("score", "correctness", "completeness", "conciseness",
                                                       "helpfulness", "honesty", "harmlessness", "judge_score")}
        for k in sorted(keys):
            metrics[k], metrics[k + "_stderr"] = mean_stderr([r.get(k) for r in records])
        metrics["judge_failures"] = sum(r.get("score") is None for r in records)
        return records, metrics

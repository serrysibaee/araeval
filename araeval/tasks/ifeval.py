"""IFEval-style verifiable instruction following (instruct models only)."""
from __future__ import annotations

from ..metrics import mean_stderr
from .base import Task
from .ifeval_checks import REGISTRY, loose_variants


class IFEvalTask(Task):
    @property
    def primary_metric(self):
        return self.spec.primary_metric or "prompt_strict"

    def run(self, backend, samples, shots_pool):
        prompts = [backend.format_prompt(s.question) for s in samples]
        outs = backend.generate(prompts, max_new_tokens=self.spec.max_new_tokens)
        records, unsupported = [], set()
        for s, out in zip(samples, outs):
            ids, kws = s.meta["instruction_id_list"], s.meta["kwargs"]
            strict, loose = [], []
            for iid, kw in zip(ids, kws):
                fn = REGISTRY.get(iid)
                if fn is None:
                    unsupported.add(iid)
                    continue
                kw = {k: v for k, v in (kw or {}).items() if v is not None}
                ok = bool(out.strip()) and fn(out, **kw)
                strict.append(ok)
                loose.append(ok or any(v.strip() and fn(v, **kw) for v in loose_variants(out)))
            records.append({"id": s.id, "group": s.group, "output": out, "instructions": ids,
                            "strict": strict, "loose": loose,
                            "prompt_strict": float(all(strict)) if strict else None,
                            "prompt_loose": float(all(loose)) if loose else None})
        metrics = {}
        for k in ("prompt_strict", "prompt_loose"):
            metrics[k], metrics[k + "_stderr"] = mean_stderr([r[k] for r in records])
        flat_s = [float(x) for r in records for x in r["strict"]]
        flat_l = [float(x) for r in records for x in r["loose"]]
        metrics["inst_strict"], _ = mean_stderr(flat_s)
        metrics["inst_loose"], _ = mean_stderr(flat_l)
        if unsupported:
            metrics["unsupported_instructions"] = ";".join(sorted(unsupported))
        return records, metrics

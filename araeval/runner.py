"""Runs every selected task for every model, with per-(model, task) caching so runs can resume."""
from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path

from .backends import load_backend
from .data import load_task
from .judges.common import JudgeCache, make_judge
from .schema import ModelSpec, TaskSpec
from .tasks import TASK_TYPES


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


class Runner:
    def __init__(self, models: list[ModelSpec], tasks: list[TaskSpec], cfg: dict, log=print):
        self.models, self.tasks, self.cfg, self.log = models, tasks, cfg, log
        self.out = Path(cfg["output_dir"]) / slug(cfg["run_name"])
        (self.out / "raw").mkdir(parents=True, exist_ok=True)
        self._data_cache: dict[str, tuple] = {}
        self.judge = self.judge_cache = None
        if cfg.get("allow_judge") and any(t.type == "judge" and t.enabled for t in tasks):
            self.judge = make_judge(cfg["judge"])
            self.judge_cache = JudgeCache(self.out / "judge_cache.jsonl")

    # ------------------------------------------------------------------
    def _skip_reason(self, m: ModelSpec, t: TaskSpec) -> str | None:
        if not t.enabled:
            return f"disabled: {t.disabled_reason}"
        if t.requires == "instruct" and not m.is_instruct:
            return "requires an instruct model"
        if t.type == "judge" and not self.cfg.get("allow_judge"):
            return "judge tasks off (set allow_judge: true)"
        return None

    def _paths(self, m: ModelSpec, t: TaskSpec):
        d = self.out / "raw" / slug(m.name)
        return d / f"{slug(t.name)}.metrics.json", d / f"{slug(t.name)}.samples.jsonl"

    def _data(self, t: TaskSpec):
        if t.name not in self._data_cache:
            self._data_cache[t.name] = load_task(t, limit=self.cfg.get("limit"), seed=self.cfg.get("seed", 1234))
        return self._data_cache[t.name]

    def _write(self, m, t, status, metrics=None, records=None, error=None, seconds=0.0, n=0):
        mpath, spath = self._paths(m, t)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        task_cls = TASK_TYPES[t.type](t, self.cfg)
        payload = {"model": m.name, "model_path": m.path, "model_type": m.type, "model_meta": m.meta,
                   "task": t.name, "task_type": t.type, "category": t.category, "hf_id": t.hf_id,
                   "primary_metric": task_cls.primary_metric, "status": status, "n": n,
                   "metrics": metrics or {}, "error": error, "seconds": round(seconds, 1),
                   "needs_check": t.needs_check, "limit": self.cfg.get("limit")}
        mpath.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        if records is not None:
            with spath.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o, set) else str(o)) + "\n")

    # ------------------------------------------------------------------
    def run(self, force: bool = False):
        total = len(self.models) * len(self.tasks)
        k = 0
        for m in self.models:
            backend = None
            for t in self.tasks:
                k += 1
                mpath, _ = self._paths(m, t)
                tag = f"[{k}/{total}] {m.name} × {t.name}"
                if mpath.exists() and not force:
                    prev = json.loads(mpath.read_text(encoding="utf-8"))
                    if prev["status"] in ("ok", "skipped"):
                        self.log(f"{tag}: cached ({prev['status']})")
                        continue
                reason = self._skip_reason(m, t)
                if reason:
                    self._write(m, t, "skipped", error=reason)
                    self.log(f"{tag}: skipped - {reason}")
                    continue
                t0 = time.time()
                try:
                    samples, shots = self._data(t)
                    if not samples:
                        raise RuntimeError("no usable samples after mapping/filters")
                    if backend is None:
                        self.log(f"loading model {m.name} ({m.path}) ...")
                        backend = load_backend(m)
                    task = TASK_TYPES[t.type](t, self.cfg)
                    if t.type == "judge":
                        task.judge, task.cache = self.judge, self.judge_cache
                    records, metrics = task.run(backend, samples, shots)
                    self._write(m, t, "ok", metrics, records, seconds=time.time() - t0, n=len(samples))
                    pm = task.primary_metric
                    self.log(f"{tag}: {pm}={metrics.get(pm, float('nan')):.4f} (n={len(samples)}, {time.time()-t0:.0f}s)")
                except Exception as e:
                    self._write(m, t, "error", error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
                                seconds=time.time() - t0)
                    self.log(f"{tag}: ERROR {type(e).__name__}: {e}")
            if backend is not None:
                backend.close()
        return self.out

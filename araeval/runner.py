"""Runs every selected task for every model; finished (model, task) pairs are cached and unfinished ones resume mid-task.

A cached result is reused only when the settings that produced it (task spec, model spec, run options such as
limit / mcq_style) are unchanged; otherwise the pair is re-run. Skipped pairs are re-checked on every run."""
from __future__ import annotations

import dataclasses
import json
import re
import time
import traceback
from pathlib import Path

from .backends import load_backend
from .checkpoint import CheckpointedBackend
from .data import load_task, stable_hash
from .env import hub_sha, software
from .judges.common import JudgeCache, make_judge
from .schema import ModelSpec, TaskSpec
from .tasks import TASK_TYPES

TASK_DESCRIPTIVE = {"category", "tags", "source_name", "needs_check", "enabled", "disabled_reason"}
MODEL_RESULT_KEYS = ("path", "type", "backend", "dtype", "max_model_len", "revision", "load_in_4bit",
                     "system_prompt", "chat_template_kwargs")
MCQ_RUN_KEYS = ("letters", "mcq_mode", "mcq_style", "mcq_max_new_tokens")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _changed(old: dict | None, new: dict) -> str:
    """'run.limit: 20 -> None, ...' for the settings that differ."""
    if not old:
        return "cached before settings were recorded"
    diffs = []
    for sec in sorted(set(old) | set(new)):
        o, n = old.get(sec) or {}, new.get(sec) or {}
        diffs += [f"{sec}.{k}: {o.get(k)!r} -> {n.get(k)!r}" for k in sorted(set(o) | set(n)) if o.get(k) != n.get(k)]
    return ", ".join(diffs)


class Runner:
    def __init__(self, models: list[ModelSpec], tasks: list[TaskSpec], cfg: dict, log=print):
        self.models, self.tasks, self.cfg, self.log = models, tasks, cfg, log
        self.out = Path(cfg["output_dir"]) / slug(cfg["run_name"])
        (self.out / "raw").mkdir(parents=True, exist_ok=True)
        self._data_cache: dict[str, tuple] = {}
        self._hub_sha: dict[str, str | None] = {}      # commit hash per model / dataset repo
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

    def _settings(self, m: ModelSpec, t: TaskSpec) -> dict:
        """Everything that can change a (model, task) result; a cached result is reused only if these match."""
        task = {k: v for k, v in dataclasses.asdict(t).items() if k not in TASK_DESCRIPTIVE}
        task["fields"] = {k: v for k, v in t.fields.items() if k != "cluster"}
        run = {"limit": self.cfg.get("limit")}
        if run["limit"] or t.limit:
            run["seed"] = self.cfg.get("seed")
        if t.type == "mcq":
            run.update({k: self.cfg.get(k) for k in MCQ_RUN_KEYS})
        if t.type == "judge":
            run["judge"] = {k: v for k, v in (self.cfg.get("judge") or {}).items() if k != "concurrency"}
        settings = {"task": task, "model": {k: getattr(m, k) for k in MODEL_RESULT_KEYS}, "run": run}
        return json.loads(json.dumps(settings, ensure_ascii=False, sort_keys=True, default=str))

    def _env(self, backend, m: ModelSpec, t: TaskSpec) -> dict:
        online = m.backend != "dummy"
        if t.hf_id not in self._hub_sha:
            self._hub_sha[t.hf_id] = hub_sha(t.hf_id, "dataset") if online else None
        if m.path not in self._hub_sha:
            self._hub_sha[m.path] = hub_sha(m.path, "model", m.revision) if online else None
        return {**software(), **backend.describe(), "model_sha": self._hub_sha[m.path],
                "dataset_sha": self._hub_sha[t.hf_id]}

    def _write(self, m, t, status, metrics=None, records=None, error=None, seconds=0.0, n=0, settings=None, env=None):
        mpath, spath = self._paths(m, t)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        task_cls = TASK_TYPES[t.type](t, self.cfg)
        settings = settings or self._settings(m, t)
        payload = {"model": m.name, "model_path": m.path, "model_type": m.type, "model_meta": m.meta,
                   "task": t.name, "task_type": t.type, "category": t.category, "hf_id": t.hf_id,
                   "primary_metric": task_cls.primary_metric, "status": status, "n": n,
                   "metrics": metrics or {}, "error": error, "seconds": round(seconds, 1),
                   "needs_check": t.needs_check, "limit": self.cfg.get("limit"),
                   "fingerprint": stable_hash(json.dumps(settings, sort_keys=True)), "settings": settings,
                   "env": env}
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
            backend = load_err = None
            for t in self.tasks:
                k += 1
                mpath, _ = self._paths(m, t)
                tag = f"[{k}/{total}] {m.name} × {t.name}"
                settings = self._settings(m, t)
                if mpath.exists() and not force:
                    prev = json.loads(mpath.read_text(encoding="utf-8"))
                    if prev["status"] == "ok":
                        if prev.get("fingerprint") == stable_hash(json.dumps(settings, sort_keys=True)):
                            self.log(f"{tag}: cached")
                            continue
                        self.log(f"{tag}: settings changed ({_changed(prev.get('settings'), settings)}) - re-running")
                reason = self._skip_reason(m, t)
                if reason:
                    self._write(m, t, "skipped", error=reason, settings=settings)
                    self.log(f"{tag}: skipped - {reason}")
                    continue
                ppath = mpath.parent / f"{slug(t.name)}.partial.jsonl"  # answers saved so far, for mid-task resume
                if force:
                    ppath.unlink(missing_ok=True)
                t0 = time.time()
                try:
                    samples, shots = self._data(t)
                    if not samples:
                        raise RuntimeError("no usable samples after mapping/filters")
                    if backend is None:
                        if load_err:  # don't reload a model that already failed for this run
                            raise RuntimeError(f"model failed to load: {load_err}")
                        self.log(f"loading model {m.name} ({m.path}) ...")
                        try:
                            backend = load_backend(m)
                        except Exception as e:
                            load_err = f"{type(e).__name__}: {e}"
                            raise
                    task = TASK_TYPES[t.type](t, self.cfg)
                    if t.type == "judge":
                        task.judge, task.cache = self.judge, self.judge_cache
                    records, metrics = task.run(CheckpointedBackend(backend, ppath, tag), samples, shots)
                    clusters = {s.id: s.cluster for s in samples if s.cluster}
                    for r in records if clusters else []:
                        if r.get("id") in clusters:
                            r["cluster"] = clusters[r["id"]]
                    self._write(m, t, "ok", metrics, records, seconds=time.time() - t0, n=len(samples),
                                settings=settings, env=self._env(backend, m, t))
                    ppath.unlink(missing_ok=True)
                    pm = task.primary_metric
                    self.log(f"{tag}: {pm}={metrics.get(pm, float('nan')):.4f} (n={len(samples)}, {time.time()-t0:.0f}s)")
                except Exception as e:
                    self._write(m, t, "error", error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
                                seconds=time.time() - t0, settings=settings)
                    self.log(f"{tag}: ERROR {type(e).__name__}: {e}")
            if backend is not None:
                backend.close()
        return self.out

"""Load model lists (YAML / CSV / plain text), the task registry (YAML) and run configs."""
from __future__ import annotations

import csv
import dataclasses
import re
from pathlib import Path

import yaml

from .schema import ModelSpec, TaskSpec

INSTRUCT_RE = re.compile(r"(instruct|chat|-it\b|-it-|-sft|-dpo|-rl\b|thinking|global|kashif)", re.I)


def _model_from_dict(d: dict) -> ModelSpec:
    known = {f.name for f in dataclasses.fields(ModelSpec)}
    meta = {k: v for k, v in d.items() if k not in known}
    spec = ModelSpec(**{k: v for k, v in d.items() if k in known})
    spec.meta = {**meta, **spec.meta}
    if not spec.name:
        spec.name = spec.path.split("/")[-1]
    return spec


def load_models(path: str | Path) -> list[ModelSpec]:
    """Accepts:
    - YAML: list of {name, path, type, backend, ...}  (or {models: [...]})
    - CSV : slm_candidates.csv (base_id / instruct_id columns) or any CSV with a `path` column
    - TXT : one HF repo id per line; type guessed from the name (instruct/chat/-it -> instruct)
    """
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        items = data["models"] if isinstance(data, dict) else data
        return [_model_from_dict(d) for d in items]
    if p.suffix == ".csv":
        out = []
        with p.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                meta = {k: row.get(k, "") for k in ("model_family", "slm_tier", "arabic_category", "total_params")}
                if "path" in row:
                    out.append(_model_from_dict({**meta, **row}))
                    continue
                for col, typ in (("base_id", "base"), ("instruct_id", "instruct")):
                    if row.get(col):
                        out.append(_model_from_dict({"name": row[col].split("/")[-1], "path": row[col], "type": typ, **meta}))
        return out
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return [ModelSpec(name=l.split("/")[-1], path=l, type="instruct" if INSTRUCT_RE.search(l) else "base") for l in lines]


def load_tasks(path: str | Path) -> list[TaskSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = data["tasks"] if isinstance(data, dict) else data
    known = {f.name for f in dataclasses.fields(TaskSpec)}
    out = []
    for d in items:
        extra = set(d) - known
        if extra:
            raise ValueError(f"task {d.get('name')}: unknown keys {sorted(extra)}")
        out.append(TaskSpec(**d))
    names = [t.name for t in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate task names: {sorted(dup)}")
    return out


def select_tasks(tasks: list[TaskSpec], select) -> list[TaskSpec]:
    """select: "all" | list of names/tags/categories | {names:[], tags:[], categories:[]}"""
    if select in (None, "all"):
        return tasks
    if isinstance(select, str):
        select = [s.strip() for s in select.split(",")]
    if isinstance(select, list):
        want = set(select)
        return [t for t in tasks if t.name in want or want & set(t.tags) or t.category in want]
    names, tags, cats = set(select.get("names", [])), set(select.get("tags", [])), set(select.get("categories", []))
    return [t for t in tasks if t.name in names or tags & set(t.tags) or t.category in cats]


DEFAULT_RUN = {
    "run_name": "run",
    "output_dir": "results",
    "select": "all",
    "limit": None,
    "letters": "arabic",            # option letters shown to models: arabic (أ ب ج د) | latin (A B C D)
    "mcq_mode": "loglik",           # loglik | generate | auto (auto: base -> loglik, instruct -> generate)
    "mcq_max_new_tokens": 32,
    "allow_judge": False,           # judge tasks call a paid API: must be switched on explicitly
    "judge": {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "medium", "concurrency": 8},
    "seed": 1234,
}


def load_run_config(path: str | Path | None) -> dict:
    cfg = dict(DEFAULT_RUN)
    if path:
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        judge = {**cfg["judge"], **user.pop("judge", {})}
        cfg.update(user)
        cfg["judge"] = judge
    return cfg

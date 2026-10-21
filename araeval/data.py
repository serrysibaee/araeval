"""Generic dataset loader: Hugging Face dataset + declarative field mapping -> list[Sample].

Field mapping keys (all optional except what the task type needs):
  id            column holding an item id
  question      column name, dotted path ("Test.Question 1") or template ("{goal}\n{ctx}")
  context       same forms as question
  choices       list of columns | one column holding a list/dict | dotted path | {"fixed": [...]} |
                {"from_values": "<column>"} (class labels collected over the whole split)
  answer        column / dotted path with the gold answer
  answer_format letter | arabic_letter | index0 | index1 | text | map | index_list0
  answer_map    dict used when answer_format == map
  strip_option_labels   remove "أ) " / "A. " prefixes from option texts
  choices_in_question   options are already written inside the question (prompt lists no options)
  gold          generation tasks: column / path / template with the reference output
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from . import preprocessors
from .data_utils import parse_value
from .schema import Sample, TaskSpec
from .textnorm import letter_index

NONE_STR = {"", "none", "nan", "null"}
OPT_LABEL = re.compile(r"^\s*(?:[A-Ja-j]|[أابجدهـوزحطي]{1,2})\s*[\)\.\-:]\s+")


def get_path(row: dict, path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        cur = parse_value(cur)
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        else:
            return None
    return parse_value(cur)


def is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in NONE_STR)


def render(row: dict, spec: Any) -> str:
    """Column name, dotted path, or a {col} template."""
    if spec is None:
        return ""
    if isinstance(spec, str) and "{" in spec:
        class _D(dict):
            def __missing__(self, k):
                v = get_path(row, k)
                return "" if is_missing(v) else str(v)
        return spec.format_map(_D())
    v = get_path(row, spec) if isinstance(spec, str) else spec
    if isinstance(v, list):
        v = "\n".join(map(str, v))
    return "" if is_missing(v) else str(v)


def extract_choices(row: dict, spec: Any, labels: list[str] | None) -> list[str]:
    if spec is None:
        return []
    if isinstance(spec, dict) and "fixed" in spec:
        return list(spec["fixed"])
    if isinstance(spec, dict) and "from_values" in spec:
        return list(labels or [])
    if isinstance(spec, list):
        vals = [row.get(c) for c in spec]
    else:
        vals = get_path(row, spec)
        if isinstance(vals, dict):
            vals = vals.get("text", list(vals.values()))
    return [str(v).strip() for v in (vals or []) if not is_missing(v)]


def resolve_gold(raw: Any, fmt: str, choices: list[str], amap: dict | None) -> Any:
    raw = parse_value(raw)
    if is_missing(raw) and fmt != "map":
        return None
    if fmt in ("letter", "arabic_letter"):
        return letter_index(str(raw))
    if fmt == "index0":
        return int(float(raw))
    if fmt == "index1":
        return int(float(raw)) - 1
    if fmt == "index_list0":
        vals = raw if isinstance(raw, list) else [raw]
        return {int(v) for v in vals}
    if fmt == "map":
        return (amap or {}).get(str(raw), (amap or {}).get(raw))
    if fmt == "text":
        norm = [c.strip() for c in choices]
        return norm.index(str(raw).strip()) if str(raw).strip() in norm else None
    raise ValueError(f"unknown answer_format {fmt}")


def _match_filters(row: dict, filters: dict) -> bool:
    for col, want in (filters or {}).items():
        v = get_path(row, col)
        if isinstance(want, list):
            if str(v) not in map(str, want):
                return False
        elif isinstance(want, str) and want.startswith("re:"):
            if not re.search(want[3:], str(v)):
                return False
        elif str(v) != str(want):
            return False
    return True


def resolve_configs(task: TaskSpec) -> list[str | None]:
    if task.configs in (None, "default"):
        return [None]
    if isinstance(task.configs, list):
        return task.configs
    from datasets import get_dataset_config_names
    names = get_dataset_config_names(task.hf_id, **_trc(task))
    if task.configs == "all":
        return names
    if isinstance(task.configs, str) and task.configs.startswith("re:"):
        return [n for n in names if re.search(task.configs[3:], n)]
    return [task.configs]


def _trc(task: TaskSpec) -> dict:
    # datasets>=4 removed `trust_remote_code`; pass it only when a task explicitly asks for it
    return {"trust_remote_code": True} if task.trust_remote_code else {}


def _load_rows(task: TaskSpec, split: str) -> list[tuple[str | None, dict]]:
    from datasets import load_dataset
    out = []
    for cfg in resolve_configs(task):
        if task.data_files:                    # load only the listed files (skips splits that fail to build)
            ds = load_dataset(task.hf_id, data_files={split: task.data_files[split]}, split=split)
        else:
            ds = load_dataset(task.hf_id, cfg, split=split, **_trc(task))
        out.extend((cfg, dict(r)) for r in ds)
    return out


def to_samples(task: TaskSpec, rows: list[tuple[str | None, dict]]) -> list[Sample]:
    f = task.fields
    pre = getattr(preprocessors, task.preprocess) if task.preprocess else None
    labels = None
    if isinstance(f.get("choices"), dict) and "from_values" in f["choices"]:
        col = f["choices"]["from_values"]
        labels = sorted({str(r[col]) for _, r in rows if not is_missing(r.get(col))})
        if pre and hasattr(pre, "label_names"):
            labels = [pre.label_names.get(l, l) for l in labels]
    samples = []
    for i, (cfg, row) in enumerate(rows):
        if not _match_filters(row, task.filters):
            continue
        over = pre(row) if pre else {}
        if over is None:                       # preprocessor dropped the row
            continue
        choices = over.get("choices") or extract_choices(row, f.get("choices"), labels)
        if f.get("strip_option_labels"):
            choices = [OPT_LABEL.sub("", c) for c in choices]
        if "gold" in over:
            gold = over["gold"]
        elif task.type == "mcq":
            raw = get_path(row, f["answer"])
            if isinstance(f.get("choices"), dict) and "from_values" in f["choices"] and pre and hasattr(pre, "label_names"):
                raw = pre.label_names.get(str(raw), raw)
            gold = resolve_gold(raw, f.get("answer_format", "letter"), choices, f.get("answer_map"))
        elif f.get("gold"):
            g = get_path(row, f["gold"]) if "{" not in f["gold"] else render(row, f["gold"])
            gold = g if isinstance(g, list) else ("" if is_missing(g) else str(g))
        else:
            gold = None
        if task.type == "mcq" and (gold is None or not choices or
                                   (isinstance(gold, int) and not 0 <= gold < len(choices))):
            continue                           # unusable item (missing/invalid gold)
        if task.group_by == "config":
            group = cfg or ""
        else:
            group = str(get_path(row, task.group_by)) if task.group_by else ""
        sid = str(get_path(row, f["id"])) if f.get("id") else f"{cfg or 'default'}-{i}"
        samples.append(Sample(
            id=sid,
            question=over.get("question", render(row, f.get("question"))),
            context=over.get("context", render(row, f.get("context"))),
            choices=choices, gold=gold, group=over.get("group", group),
            meta={**over.get("meta", {}), "choices_in_question": bool(f.get("choices_in_question"))},
        ))
    return samples


def load_task(task: TaskSpec, limit: int | None = None, seed: int = 1234) -> tuple[list[Sample], list[Sample]]:
    """Returns (eval samples, few-shot pool). Sampling under a limit is deterministic."""
    samples = to_samples(task, _load_rows(task, task.split))
    lim = limit or task.limit
    if lim and len(samples) > lim:
        rng = random.Random(seed)
        samples = sorted(rng.sample(samples, lim), key=lambda s: s.id)
    shots: list[Sample] = []
    if task.num_fewshot and task.fewshot_split:
        shots = to_samples(task, _load_rows(task, task.fewshot_split))
    return samples, shots


def stable_hash(*parts: Any) -> str:
    return hashlib.sha1("␟".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]

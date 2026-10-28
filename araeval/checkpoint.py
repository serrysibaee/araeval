"""Per-(model, task) checkpoint: model answers are appended to a .partial.jsonl after every chunk,
so a stopped task resumes where it left off. Also drives the per-task progress bar."""
from __future__ import annotations

import json
from pathlib import Path

from tqdm.auto import tqdm

from .data import stable_hash

CHUNK = 256  # requests per model call; smaller = finer resume/progress, larger = better batching


class CheckpointedBackend:
    """Wraps a loaded backend for one task: skips requests already answered, runs the rest in chunks."""

    def __init__(self, backend, path: Path, desc: str):
        self.backend, self.path, self.desc = backend, Path(path), desc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, object] = {}
        text = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        for line in text.splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # half-written last line from a hard kill
                continue
            self.done[r["k"]] = r["v"]
        self._torn = bool(text) and not text.endswith("\n")

    def __getattr__(self, name):  # spec, format_prompt, supports_loglikelihood, ... come from the real backend
        return getattr(self.backend, name)

    def _run(self, keys, items, fn):
        todo = [i for i, k in enumerate(keys) if k not in self.done]
        with tqdm(total=len(keys), initial=len(keys) - len(todo), desc=self.desc, unit="req", dynamic_ncols=True) as bar, \
                self.path.open("a", encoding="utf-8") as f:
            if self._torn:
                f.write("\n")
                self._torn = False
            for b in range(0, len(todo), CHUNK):
                idx = todo[b:b + CHUNK]
                for i, v in zip(idx, fn([items[i] for i in idx])):
                    self.done[keys[i]] = v
                    f.write(json.dumps({"k": keys[i], "v": v}, ensure_ascii=False) + "\n")
                f.flush()
                bar.update(len(idx))
        return [self.done[k] for k in keys]

    def loglikelihood(self, requests):
        keys = [stable_hash("ll", self.backend.spec.path, c, t) for c, t in requests]
        return self._run(keys, requests, self.backend.loglikelihood)

    def generate(self, prompts, max_new_tokens=256, stop=None):
        keys = [stable_hash("gen", self.backend.spec.path, p, max_new_tokens, stop) for p in prompts]
        return self._run(keys, prompts, lambda ps: self.backend.generate(ps, max_new_tokens=max_new_tokens, stop=stop))

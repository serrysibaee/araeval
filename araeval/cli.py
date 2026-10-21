"""Command line: python -m araeval <command>

  run       evaluate models × tasks and write the report
  report    rebuild the report from an existing run directory
  list      show the task registry (name, type, category, enabled)
  check     load every selected task and print one rendered prompt (no model needed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_models, load_run_config, load_tasks, select_tasks
from .report import build_report
from .runner import Runner, slug

HERE = Path(__file__).resolve().parent.parent


def _common(p):
    p.add_argument("--config", help="run config YAML (see configs/run_example.yaml)")
    p.add_argument("--tasks", default=str(HERE / "configs" / "tasks.yaml"), help="task registry YAML")
    p.add_argument("--select", help="comma list of task names / tags / categories, or 'all'")
    p.add_argument("--limit", type=int, help="max items per task (quick runs)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="araeval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    _common(r)
    r.add_argument("--models", help="models file: .yaml / .csv / .txt (one HF id per line)")
    r.add_argument("--out", help="output directory (default results/)")
    r.add_argument("--name", help="run name")
    r.add_argument("--mcq-mode", choices=["loglik", "generate", "auto"])
    r.add_argument("--allow-judge", action="store_true", help="enable paid LLM-judge tasks")
    r.add_argument("--force", action="store_true", help="re-run even if cached results exist")
    r.add_argument("--backend", help="override backend for every model (vllm/hf/openai/dummy)")

    rep = sub.add_parser("report")
    rep.add_argument("run_dir")

    ls = sub.add_parser("list")
    ls.add_argument("--tasks", default=str(HERE / "configs" / "tasks.yaml"))

    ck = sub.add_parser("check")
    _common(ck)

    a = ap.parse_args(argv)

    if a.cmd == "report":
        build_report(Path(a.run_dir))
        return

    tasks_all = load_tasks(a.tasks)
    if a.cmd == "list":
        for t in tasks_all:
            flag = "on " if t.enabled else "off"
            print(f"{flag} {t.name:34s} {t.type:9s} {t.category:36s} {','.join(t.tags):22s} {t.disabled_reason or t.needs_check}")
        print(f"\n{sum(t.enabled for t in tasks_all)} enabled / {len(tasks_all)} tasks")
        return

    cfg = load_run_config(a.config)
    for key, val in (("select", a.select), ("limit", a.limit)):
        if val is not None:
            cfg[key] = val
    tasks = select_tasks(tasks_all, cfg["select"])

    if a.cmd == "check":
        from .backends.dummy import DummyBackend
        from .data import load_task
        from .schema import ModelSpec
        from .tasks import TASK_TYPES
        dummy = DummyBackend(ModelSpec(name="check", path="-", type="base"))
        for t in tasks:
            if not t.enabled:
                print(f"--- {t.name}: disabled ({t.disabled_reason})")
                continue
            try:
                samples, shots = load_task(t, limit=cfg.get("limit") or 50)
                task = TASK_TYPES[t.type](t, cfg)
                s = samples[0]
                prompt = task.build_prompt(s, task.pick_shots(shots, s)) if hasattr(task, "build_prompt") else s.question
                print(f"=== {t.name} ({len(samples)} items loaded, gold={s.gold!r:.60})\n{prompt[:700]}\n")
            except Exception as e:
                print(f"!!! {t.name}: {type(e).__name__}: {e}\n")
        return

    # run
    if a.models:
        cfg["models"] = a.models
    if not cfg.get("models"):
        sys.exit("no models: pass --models or set `models:` in the run config")
    for key, val in (("output_dir", a.out), ("run_name", a.name), ("mcq_mode", a.mcq_mode)):
        if val:
            cfg[key] = val
    if a.allow_judge:
        cfg["allow_judge"] = True
    models = load_models(cfg["models"])
    if a.backend:
        for m in models:
            m.backend = a.backend
    print(f"{len(models)} models × {len(tasks)} tasks -> {Path(cfg['output_dir']) / slug(cfg['run_name'])}")
    out = Runner(models, tasks, cfg).run(force=a.force)
    build_report(out, run_cfg=cfg)


if __name__ == "__main__":
    main()

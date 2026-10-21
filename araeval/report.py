"""Aggregate raw per-(model, task) results into CSV/JSON reports and a terminal summary.

Outputs (in the run directory):
  results_long.csv   one row per model × task × metric (value, stderr, n, mode, status)
  leaderboard.csv    models × tasks (primary metric, 0-100) + category means + overall mean
  by_group.csv       per-group breakdown (subject / dialect / config) of the primary metric
  task_status.csv    status of every model × task (ok / skipped / error + reason)
  report.json        everything above in one machine-readable file, plus run metadata
"""
from __future__ import annotations

import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__

SCALE_100 = {"chrf", "bleu"}          # corpus metrics already on 0-100


def _to100(metric: str, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v) if metric in SCALE_100 else 100.0 * float(v)


def build_report(run_dir: Path, run_cfg: dict | None = None, print_summary: bool = True) -> dict:
    run_dir = Path(run_dir)
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "raw").rglob("*.metrics.json"))]
    if not metas:
        raise SystemExit(f"no results under {run_dir/'raw'}")

    long_rows, status_rows, group_rows = [], [], []
    for m in metas:
        meta = m.get("model_meta") or {}
        status_rows.append({"model": m["model"], "task": m["task"], "status": m["status"], "n": m["n"],
                            "seconds": m["seconds"], "reason": (m.get("error") or "").split("\n")[0],
                            "needs_check": m.get("needs_check", "")})
        if m["status"] != "ok":
            continue
        for k, v in m["metrics"].items():
            if k.endswith("_stderr") or not isinstance(v, (int, float)):
                continue
            long_rows.append({"model": m["model"], "model_type": m["model_type"], "model_path": m["model_path"],
                              "tier": meta.get("slm_tier", meta.get("tier", "")),
                              "arabic_category": meta.get("arabic_category", ""),
                              "task": m["task"], "category": m["category"], "task_type": m["task_type"],
                              "metric": k, "value": v, "stderr": m["metrics"].get(k + "_stderr"),
                              "is_primary": k == m["primary_metric"], "n": m["n"],
                              "mcq_mode": m["metrics"].get("mode", ""), "limit": m.get("limit")})
        # per-group breakdown from per-sample records
        spath = Path(str(run_dir / "raw" / _slug(m["model"]) / _slug(m["task"])) + ".samples.jsonl")
        if spath.exists():
            agg = defaultdict(list)
            pm = m["primary_metric"]
            for line in spath.open(encoding="utf-8"):
                r = json.loads(line)
                if r.get("group") and isinstance(r.get(pm), (int, float)):
                    agg[r["group"]].append(r[pm])
            for g, vals in agg.items():
                group_rows.append({"model": m["model"], "task": m["task"], "group": g, "metric": pm,
                                   "value": sum(vals) / len(vals), "n": len(vals)})

    long_df = pd.DataFrame(long_rows)
    status_df = pd.DataFrame(status_rows)
    group_df = pd.DataFrame(group_rows)
    run_dir.joinpath("results_long.csv").write_text(long_df.to_csv(index=False), encoding="utf-8-sig")
    run_dir.joinpath("task_status.csv").write_text(status_df.to_csv(index=False), encoding="utf-8-sig")
    run_dir.joinpath("by_group.csv").write_text(group_df.to_csv(index=False), encoding="utf-8-sig")

    board = pd.DataFrame()
    if not long_df.empty:
        prim = long_df[long_df.is_primary].copy()
        prim["score"] = [_to100(mt, v) for mt, v in zip(prim.metric, prim.value)]
        board = prim.pivot_table(index="model", columns="task", values="score", aggfunc="first")
        cats = prim.groupby(["model", "category"]).score.mean().unstack()
        cats.columns = [f"[{c}]" for c in cats.columns]
        info = prim.groupby("model")[["model_type", "tier", "arabic_category"]].first()
        board = info.join(cats).join(board)
        board.insert(3, "overall", prim.groupby("model").score.mean())
        board.insert(4, "tasks_done", prim.groupby("model").task.nunique())
        board = board.sort_values("overall", ascending=False).round(2)
    run_dir.joinpath("leaderboard.csv").write_text(board.to_csv(), encoding="utf-8-sig")

    report = {
        "araeval_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "run_config": run_cfg,
        "scoring_note": "leaderboard scores are the task's primary metric on 0-100; category and overall "
                        "columns are unweighted means over the tasks a model completed (compare models on "
                        "the same task set - see tasks_done).",
        "leaderboard": json.loads(board.reset_index().to_json(orient="records", force_ascii=False)) if not board.empty else [],
        "results": long_rows,
        "groups": group_rows,
        "status": status_rows,
    }
    run_dir.joinpath("report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    if print_summary:
        _print_summary(board, status_df, run_dir)
    return report


def _slug(s):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _print_summary(board: pd.DataFrame, status: pd.DataFrame, run_dir: Path):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 100)
    print(f"RESULTS  ({run_dir})")
    print("=" * 100)
    if board.empty:
        print("no completed tasks")
    else:
        cols = [c for c in board.columns if c in ("model_type", "tier", "overall", "tasks_done") or c.startswith("[")]
        print(board[cols].to_string())
    counts = status.status.value_counts().to_dict()
    print(f"\nmodel × task status: {counts}")
    errs = status[status.status == "error"]
    if len(errs):
        print("\nerrors:")
        for _, r in errs.iterrows():
            print(f"  {r.model} × {r.task}: {r.reason}")
    print(f"\nfiles: leaderboard.csv, results_long.csv, by_group.csv, task_status.csv, report.json, raw/")

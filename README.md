# araeval: generic evaluation of Arabic-capable (small) language models

Give it **a list of models** and **a list of tasks** (datasets + field mappings). It runs every model on every task and writes CSV/JSON reports.

```
models.yaml / .csv / .txt  ─┐
                            ├─► Runner ─► backend (vLLM | HF | OpenAI-compatible)
configs/tasks.yaml ─────────┘      │
                                   ├─► task type: mcq | generate | ifeval | judge
                                   └─► results/<run>/raw/…  ─►  report (CSV / JSON + terminal summary)
```

## Install (Linux GPU server)

```bash
git clone https://github.com/serrysibaee/araeval.git && cd araeval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"                # core + vLLM + judge clients (or: pip install -r requirements.txt)
huggingface-cli login                  # for gated models/datasets (Gemma, Jais-2, NileChat…)
export ANTHROPIC_API_KEY=...           # only if you run judge tasks with Claude
python tests/test_core.py              # offline sanity check (no downloads)
```

Run commands from the repo root (`configs/` paths are relative to it). After `pip install -e .`
the `araeval` command is equivalent to `python -m araeval`.

## Run

```bash
# list registered tasks (50 enabled / 64)
python -m araeval list

# check data mappings: loads each selected dataset and prints one rendered prompt + gold (no model)
python -m araeval check --select core

# quick smoke run: 2 models, core tasks, 100 items per task
python -m araeval run --models configs/models.yaml --select core --limit 100 --name smoke

# full run from a config file (recommended)
python -m araeval run --config configs/run_example.yaml

# include paid LLM-judge tasks (AraGen, Fann-or-Flop explanations, Aya, ALARB)
python -m araeval run --config configs/run_example.yaml --allow-judge

# rebuild the report from an existing run directory
python -m araeval report results/slm_core_2026-10
```

Runs **resume**: every finished model × task is cached under `raw/`. Re-running skips them (`--force` recomputes).

## Inputs

**Models** — any of:
- `configs/models.yaml`: `{name, path, type: base|instruct, backend, max_model_len, tier, …}`
- a candidates CSV (e.g. `slm_candidates.csv` with `base_id` / `instruct_id` columns): every `base_id` / `instruct_id` becomes one model
- a `.txt` file with one HF repo id per line (instruct is guessed from names containing instruct/chat/-it)

**Tasks** — `configs/tasks.yaml`, selected with `--select` by name, tag (`core`, `dialect`, `linguistic`, …) or category.

## How each task type is scored

| type | used for | scoring |
|---|---|---|
| `mcq` | exams, culture, dialect MCQ, reading comp., linguistics | `loglik` (default): log-likelihood of each option letter (or option text for completion tasks: acc + length-normalised acc_norm). Works for base **and** instruct models, so all models are comparable. `generate`: the model answers and the letter is extracted (`unparsed_rate` reported). |
| `generate` | extractive QA, GSM8K, GEC, MT, summarisation | `em`, `numeric_em`, token `f1`, `contains`, `rouge` (Arabic-safe), corpus `chrf` / `bleu` (sacrebleu) |
| `ifeval` | Arabic IFEval | 23 rule checkers incl. Arabic-specific (tashkeel, letter frequency, pronoun lists); prompt/instruction-level strict + loose accuracy |
| `judge` | AraGen 3C3H, poetry explanation, Aya, ALARB | model answers, then an LLM judge (Claude by default) scores against the reference with a rubric; cached in `judge_cache.jsonl` |

Arabic handling: answers and references are normalised (diacritics, tatweel, alef/ya/ta-marbuta, Arabic-Indic digits) before string metrics; option letters default to أ ب ج د هـ (`letters: latin` for A B C D).

`requires: instruct` tasks (IFEval, judge, MT, GEC…) are **skipped** for base models and reported as such.

## Outputs (`results/<run_name>/`)

| file | content |
|---|---|
| `leaderboard.csv` | models × tasks (primary metric, 0-100), category means, `overall`, `tasks_done` |
| `results_long.csv` | every metric: model, task, category, metric, value, stderr, n, mode |
| `by_group.csv` | per-subject / per-dialect / per-config breakdown |
| `task_status.csv` | ok / skipped / error with the reason for every model × task |
| `report.json` | all of the above + run config and metadata |
| `raw/<model>/<task>.samples.jsonl` | per-item predictions, outputs, log-likelihoods (error analysis) |

`overall` and category columns are unweighted means over the tasks a model completed. Base models skip
instruct-only tasks, so compare them on the same task set (`tasks_done`) or per category.

## Adding a dataset

Add an entry to `configs/tasks.yaml`:

```yaml
- name: my_mcq
  hf_id: org/dataset
  type: mcq
  category: Knowledge & exams
  configs: all            # or [cfg1, cfg2] / omit for default / "re:<regex>"
  split: test
  fields:
    question: question                       # column, dotted path (a.b) or template "{col1}\n{col2}"
    context: passage
    choices: [opt_a, opt_b, opt_c, opt_d]     # or a list/dict column, or {fixed: [...]}
    answer: label
    answer_format: index0                    # letter | arabic_letter | index0 | index1 | text | map | index_list0
  group_by: subject
  tags: [mine]
```

If the mapping cannot express the dataset, write a function in `araeval/preprocessors.py` that returns
`{question, context, choices, gold, group, meta}` (or `None` to drop the row) and set `preprocess: <function>`.
Run `python -m araeval check --select my_mcq` to see the rendered prompt and gold answer before running models.

A new **task type** = subclass `araeval/tasks/base.py:Task`, implement `run()`, register it in `araeval/tasks/__init__.py`.

## Known limitations

- 14 registered datasets are disabled with a stated reason (`python -m araeval list`): missing answer columns, gated schemas, index-only "verified" releases, or scorers not yet written (FLORES pairing, ABBL, BLEnD; XL-Sum still needs a loading script).
- Mapping caveats are carried into `task_status.csv` (`needs_check`): Nahw-MCQ has no answer column (option A assumed correct), INCLUDE answer indexing assumed 0-based, QALB scored with chrF/F1 rather than the official M2 scorer.
- `loglik` MCQ needs a backend with log-probabilities (vLLM, HF). OpenAI-compatible servers fall back to `generate` mode.
- On Windows, 4 AlGhafa-Native configs fail with a path-length error in the HF cache (long config names); this does not occur on Linux.

## Tested

- `python tests/test_core.py`: 8 offline tests (normalisation, metrics, gold formats, field mapping, letter extraction, 23 IFEval checkers, dummy end-to-end).
- `python -m araeval check --select all`: every enabled task loaded from the Hub and rendered (2026-10-21).
- CPU smoke run: SmolLM2-135M base + instruct × 8 tasks (loglik MCQ, choice-text MCQ, extractive QA, 5-shot GSM8K, IFEval, skip rules, report).

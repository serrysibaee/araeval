"""Core data structures shared by loaders, tasks, backends and the runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    """One evaluation item in canonical form, whatever the source dataset looked like."""
    id: str
    question: str = ""                      # main input text (question, prompt, source sentence...)
    context: str = ""                       # optional passage shown before the question
    choices: list[str] = field(default_factory=list)   # MCQ options (empty for generation tasks)
    gold: Any = None                        # MCQ: int index or set of indices; generation: str or list[str]
    group: str = ""                         # breakdown key (subject, dialect, config...)
    meta: dict = field(default_factory=dict)  # anything task-specific (IFEval kwargs, judge rubric...)


@dataclass
class ModelSpec:
    name: str                                # display name in reports
    path: str                                # HF repo id, local path, or API model id
    type: str = "base"                       # base | instruct
    backend: str = "auto"                    # auto | vllm | hf | openai | dummy
    dtype: str = "auto"
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    revision: str | None = None
    batch_size: int = 8                      # HF backend only
    load_in_4bit: bool = False               # HF backend only (bitsandbytes)
    api_base: str | None = None              # openai backend
    api_key_env: str = "OPENAI_API_KEY"      # openai backend
    system_prompt: str | None = None         # instruct models only
    meta: dict = field(default_factory=dict)  # tier, arabic_category, family... copied into reports

    @property
    def is_instruct(self) -> bool:
        return self.type == "instruct"


@dataclass
class TaskSpec:
    name: str
    hf_id: str
    type: str                                # mcq | generate | ifeval | judge
    category: str = ""
    configs: Any = None                      # None -> default config; list; "all"; "re:<regex>"
    split: str = "test"
    fewshot_split: str | None = None
    num_fewshot: int = 0
    fields: dict = field(default_factory=dict)
    filters: dict = field(default_factory=dict)
    preprocess: str | None = None            # name of a function in araeval.preprocessors
    group_by: str | None = None              # column name, or "config"
    limit: int | None = None                 # cap items (also settable per run)
    instruction: str | None = None           # task-specific instruction prepended to every prompt
    mcq_style: str = "letter"                # letter | choice_text (log-likelihood target)
    metrics: list = field(default_factory=list)      # generate tasks: em, numeric_em, f1, contains, chrf, bleu, rouge
    primary_metric: str | None = None
    max_new_tokens: int = 256
    requires: str | None = None              # "instruct" -> skipped for base models
    judge_rubric: str | None = None          # judge tasks: name of a rubric in araeval.judges.rubrics
    enabled: bool = True
    disabled_reason: str = ""
    needs_check: str = ""                    # mapping caveat shown in reports
    trust_remote_code: bool = False
    tags: list = field(default_factory=list)  # e.g. [core, dialect, linguistic] for selection
    data_files: dict | None = None           # {split: "path/in/repo.parquet"} to bypass broken splits
    source_name: str = ""                    # row name in arabic_eval_datasets_detailed.csv

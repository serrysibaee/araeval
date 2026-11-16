# How small language models are evaluated: metrics, zero/few-shot and seeds

*Literature summary, 12 November 2026. Sources: about 25 papers, technical reports and leaderboard docs from 2024 to 2026, listed at the end.*

## Bottom line

1. **Metrics.** Accuracy dominates. Multiple-choice questions are scored from per-option log-likelihood, and generative tasks are scored by exact match (math), pass@1 (code), strict accuracy (IFEval) or an LLM judge (open-ended). Papers written specifically about SLMs also report efficiency: latency, memory, throughput and sometimes energy.
2. **Zero- vs few-shot depends on the model type.**
   - **Base models** are usually few-shot: 5 shots for knowledge MCQ, and 3–8 chain-of-thought (CoT) shots for math and BBH.
   - **Instruct models** are usually zero-shot with the chat template.
   - **Small base models** on commonsense tasks are also zero-shot.
   - **Both Arabic leaderboards** (OALL and HELM Arabic) are zero-shot.
3. **Seeds.** Most papers run each model once and report a single number, with no seed and no error bar. Seeds matter only where randomness actually enters:
   - which few-shot examples are drawn;
   - which items are kept when a benchmark is subsampled;
   - sampled (temperature > 0) decoding.

   Greedy and log-likelihood scoring need one run. Even so, hardware and batch size still add some noise.
4. **The 2024–2026 methodology papers agree on four things:**
   - Use the cloze formulation for small models, because multiple-choice letter scoring sits near chance for them.
   - Report confidence intervals.
   - Average over several seeds when decoding is sampled.
   - Treat prompt and harness choices as a source of variance, not a fixed detail.

## 1. Metrics

| Task type | Usual metric | How it is computed | Notes for small models |
|---|---|---|---|
| Multiple-choice, letter form (MCF) | accuracy | probability of the answer letter (A/B/C/D) | Near chance for small or early-training models (OLMES, Madaan et al., SmolLM2) |
| Multiple-choice, cloze form (CF) | acc_norm | probability of each answer text, divided by its length in characters or tokens; PMI for some tasks | The standard choice for small base models. OLMES runs both CF and MCF and keeps the better one per model. |
| Math word problems | exact match on the final number | greedy generation, usually with CoT shots | One wrong item moves small test sets a lot |
| Code | pass@1 / functional correctness | run unit tests on the generated code | The SLM code study pairs accuracy with VRAM use |
| Instruction following | IFEval prompt-level and instruction-level strict accuracy | rule checkers | Zero-shot; instruct models only |
| Open-ended generation | LLM-as-judge score (1–5 or 0–10) or win rate | one or several judge models | The 2026 Arabic SLM study averages three judges and flags responses where the judges disagree |
| Translation | chrF / BLEU (COMET is also common) | corpus-level score against references | |
| Safety | judge label: harmful 0 / ambiguous 0.5 / safe 1 | rubric-based judge | A 2026 study finds "ambiguous" dominates for SLMs and rankings shift with how it is treated |
| Efficiency (SLM papers) | parameters, peak memory, prefill and decode latency, tokens/s, energy | llama.cpp or vLLM on target hardware (Jetson, phones) | Lu et al. use 4-bit llama.cpp, a fixed 50-token prompt and 50-token output, and pauses between runs to avoid thermal throttling |

**Aggregation:**
- **Leaderboard averages.** Leaderboards average the tasks. Open LLM Leaderboard v2 first rescales each score against its random baseline, so a 2-option task and a 10-option task count on the same scale.
- **Smoother metrics.** Madaan et al. and Signal & Noise find that continuous metrics have a better signal-to-noise ratio than 0/1 accuracy, especially for small models. Examples are the probability given to the correct answer and perplexity.

## 2. Zero-shot vs few-shot: what each source does

| Source | Models | Shots | Scoring |
|---|---|---|---|
| Lu et al. 2024, *SLMs: Survey, Measurements* (70 SLMs, 100M–5B) | base + instruct | **5-shot** by default ("the most common setting in the released model") | accuracy |
| SmolLM2 (2025) | base | **0-shot** by default (lighteval); GSM8K 5-shot, MATH 4-shot | tracks both MCF and CF for MMLU |
| SmolLM2 (2025) | instruct | IFEval and MT-Bench 0-shot; BBH 3, GSM8K 5, MATH 4 | |
| TinyLlama, MobileLLM | base, under 1.5B | **0-shot** commonsense suite (ARC, BoolQ, PIQA, HellaSwag, …) | accuracy |
| OLMES (AI2, NAACL 2025) | base | **fixed, hand-curated 5 shots** per task; more shots made little difference | both CF and MCF, keep the better; normalisation chosen per task (character, PMI or none) |
| Open LLM Leaderboard v2 (HF) | all | IFEval 0, BBH 3, MATH-L5 4, GPQA 0, MuSR 0, MMLU-Pro 5; instruct models get the chat template, with the shots given as earlier chat turns | acc_norm / acc / EM / strict acc |
| Qwen3 technical report (2025) | base | MMLU 5, MMLU-Pro 5 CoT, BBH 3 CoT, GSM8K 4 CoT, MATH 4 CoT, MGSM 8 CoT, MMMLU 5, INCLUDE 5 | |
| Qwen3 technical report (2025) | post-trained | 0-shot, sampled: thinking T=0.6 / top-p 0.95 / top-k 20; non-thinking T=0.7 / top-p 0.8 / top-k 20 | averaged over 64 samples on AIME, 10 on GPQA-Diamond |
| HELM Classic / HELM Lite | all | 5 in-context examples per item | |
| **HELM Arabic** (Dec 2025) | instruct | **0-shot** ("all evaluated models were instruction-tuned"); Arabic option letters; at most 1,000 sampled items per subset; thinking mode off | |
| **OALL v1 / v2** | all | v1: **0-shot**, normalised log-likelihood; v2: chat template applied when the model has one | lighteval |
| **ArabicMMLU** (Koto et al. 2024) | base + instruct | 0-shot and few-shot compared: few-shot **helped base models (+2 to +10 points) and hurt instruct models** (BLOOMZ, Jais-chat) | probability of the option letter |
| **Arabic Stable LM 1.6B** (2024) | base + chat | ArabicMMLU, AlGhafa and CIDAR use **cloze + normalised accuracy**, "because it is more robust when compared to MCF"; ACVA 5-shot, macro-F1 | |
| **Arabic SLM evaluation** (Naseej, Jun 2026; 12 models incl. Tiny Aya 3.35B, Gemma-3-1B) | instruct | **0-shot**, a fixed Arabic-only prompt | 3 LLM judges (0–5), averaged |
| EasyMath (2025; 23 models, 14M–4B) | all | **0-shot** | exact, numeric and symbolic answer checks; CoT gives modest gains |

**The usual pattern:**
- **Base models, knowledge MCQ:** 5-shot.
- **Base models, math and BBH:** 3–8 CoT shots.
- **Base models, commonsense:** 0-shot.
- **Instruct models:** 0-shot with the chat template. When shots are used, they go in as earlier chat turns.
- **Arabic evaluation:** mostly 0-shot.

ArabicMMLU's result shows the trade-off: few-shot helps base models, but can hurt instruct models.

## 3. Seeds and variance

### Where randomness comes from, and how papers handle it

| Source of variance | Typical practice | Evidence |
|---|---|---|
| **Which few-shot examples are drawn, and their order** | Usually one fixed draw: lm-eval-harness fixes the few-shot sampler seed (1234 by default); OLMES uses fixed curated shots. HELM Classic averaged **3 seeds** of example choice; HELM Lite dropped to 1. | Example order and prompt format can swing few-shot accuracy by tens of points: up to 76 points for Llama-2-13B (Sclar et al.). |
| **Subsampling a benchmark** | A fixed seed: OLMES takes 1,000 items when a set has more than 1,500, using `Random(1234)`; HELM Arabic samples 1,000 items per subset; Qwen3 used 10% of INCLUDE and MMMLU. | |
| **Sampled decoding (T > 0)** | Average over several samples per item: Qwen3 uses 64 on AIME and 10 on GPQA-D. *Sober Look* uses 10 seeds for AIME and AMC and 3 for the other benchmarks. | On 1.5B–7B reasoning models, Pass@1 varies by **5–15 points (std)** across 20 seeds on AIME and AMC. One question moves the score 2.5–3.3 points. Temperature alone moves results by up to 15 points and top-p by up to 8. |
| **Hardware and numerics (even greedy)** | Rarely controlled; OLMES and Open LLM Leaderboard note that batch size and GPU can flip close answers. | In bf16 with greedy decoding, accuracy varied by up to 9% with GPU count, GPU type and batch size (*Give Me FP32*). *Sober Look* saw 6–8 points across compute clusters, and a 22.2 vs 17.7 gap between two evaluation frameworks. |
| **Prompt wording and harness settings** | One template per task (most papers). | Model rankings change with instruction paraphrases (Mizrahi et al., 20 LLMs × 39 tasks). One 2026 preprint ran 12 models under 26 harness settings: one model scored anywhere from 31% to 89%, and 4 of the 12 models came first under some setting. Scoring method (generated text vs option likelihood) was the biggest lever. |
| **Training seed** | Not available to outside evaluators. | Madaan et al. trained ten 7B models that differ only in seed. Across seeds the std was MMLU 0.57, ARC-C 0.80, HellaSwag 0.21 and GSM8K 0.41 points. That is smaller than each benchmark's own 95% CI (0.72, 2.74, 0.93 and 0.87). |

### How results are reported

- **Most model reports and SLM surveys:** a single run, a point estimate, no seed.
- **Llama 3:** 95% CIs that assume Gaussian scores, following Madaan et al.
- **Miller 2024 (*Adding Error Bars to Evals*)** recommends five steps:
  1. CLT standard errors.
  2. Clustered standard errors when questions come in groups, for example several questions on one passage.
  3. Less variance through resampling answers or using next-token probabilities.
  4. Paired differences on the same questions when comparing two models.
  5. Power analysis before trusting a small eval.
- **Madaan et al.** warn against IRT-based subsampling such as tinyBenchmarks for comparing models, because it increases variance.
- **Signal & Noise (NeurIPS 2025)** measures each benchmark's signal-to-noise ratio. To reduce noise it suggests:
  - drop noisy subtasks;
  - average over checkpoints;
  - switch to continuous metrics.
- **Sober Look** recommends, for sampled reasoning evaluation:
  - at least 10 seeds (30 is better) on benchmarks with fewer than about 100 items, reporting mean ± std;
  - temperature and top-p set once per model and kept the same across tasks;
  - the correct chat template;
  - enough context length;
  - robust answer parsing;
  - released outputs.

## 4. What this means for our ≤ 4B Arabic study

Today araeval does the following:
- **Scoring:** greedy decoding; MCQ scored by log-likelihood; thinking off.
- **Formulation:** 15 of the 17 MCQ tasks use letter scoring (MCF), and only AraDiCE-PIQA and XStoryCloze use cloze.
- **Shots:** everything is 0-shot except Arabic GSM8K, which is 5-shot.
- **Few-shot draw:** keyed to task + item id, so there is exactly one fixed draw.
- **Statistics:** standard errors are stored per metric.

*Update, 16 Nov 2026 (araeval 0.2.0):* four changes are now in araeval. The full-run config sets `mcq_style: both`, which scores both formulations and keeps the better. Each result records its environment. Cluster keys are stored for Belebele, Dialectal-MMLU and AraDiCE-PIQA. Cached results are reused only when their settings match. Few-shot seeds and the report additions (confidence intervals, chance-adjusted averages, paired tests) are still to do.

| Topic | Common practice | araeval now | Suggested change |
|---|---|---|---|
| **MCQ formulation** | Use cloze for small models; OLMES runs both and keeps the better | letter (MCF) on 15 of 17 MCQ tasks, set per task in `tasks.yaml` | Run both `letter` and `choice_text` and report the better per model and task (OLMES rule), with both in the appendix. **This matters most for us:** sub-1B and base models may sit at chance under MCF. It needs a run-level `mcq_style` switch. |
| **Main shot setting** | Arabic leaderboards: 0-shot; instruct models: 0-shot | 0-shot (GSM8K 5-shot) | Keep 0-shot as the main table. It is the only setting all 20 tasks support, and it doesn't penalise instruct models. |
| **Few-shot track** | Base models: 5-shot (OLMES, Qwen3, Lu et al.) | none | Add 5-shot for base models on the tasks with a dev or validation split: ArabicMMLU, INCLUDE, MadinahQA. GSM8K is already 5-shot. Instruct models could also be run with the shots as earlier chat turns. |
| **Few-shot seeds** | 1 fixed draw (harness) or 3 seeds (HELM Classic) | 1 draw, keyed to task + item | Add a `fewshot_seed` setting, run **3 seeds**, and report mean ± std. |
| **Decoding** | Greedy for the main eval; sampling with several samples for thinking models | greedy, thinking off | Keep it, so no decoding seeds are needed. If a reasoning track is added later, use each model card's sampling settings with ≥ 3 samples (≥ 10 on sets under ~500 items). |
| **Error bars** | 95% CI; clustered SE; paired tests | stderr stored | Print ±1.96·SE. Cluster Belebele by passage and Dialectal-MMLU by question (same question in 5 dialects). Use paired tests for any "model A beats B" claim. |
| **Averaging across tasks** | Rescale against the random baseline before averaging (Open LLM Leaderboard v2) | unweighted mean of raw scores | Rescale each task as (score − chance) / (1 − chance) before averaging, because tasks have 2 to 4+ options. |
| **Reproducibility** | Fixed stack; log versions | logs Python + araeval versions | Also log vLLM, torch and transformers versions, GPU, dtype and batch size, and keep them fixed for the whole study. Re-run one model twice to measure hardware noise. |
| **Efficiency** | SLM surveys report memory, latency and throughput | not measured | Optional: log parameter count, peak GPU memory and tokens/s per model. |

### How precise each task is (95% CI half-width, worst case p = 0.5)

| ± points | Tasks (items) |
|---|---|
| ±8 | AraLingBench (149), Kinayat (150) |
| ±4–5 | Arabic IFEval (404), AraTrust (522), INCLUDE-ar (552) |
| ±3–3.5 | ArabicSense (848), MadinahQA (977) |
| ±2.5–2.7 | Arabic GSM8K (1,306), XStoryCloze (1,511) |
| ±1.2–1.9 | 3LM (2,608), ArabCulture (3,444), Belebele\* (5,400), AraDiCE-PIQA (5,514), Fann-or-Flop (6,754), GATmath (6,818) |
| < ±1 | ArabicMMLU (13,690), Dialectal-MMLU\* (15,675) |

\* The true interval is wider, because items are clustered (by passage, or the same question in five dialects).

These are intervals for one model's score. Comparing two models on the same items (a paired test) is tighter. Even so, on AraLingBench and Kinayat a gap of a few points between two models means nothing. Read them at the category level, or treat them as indicative only.

## Sources

**SLM surveys and SLM-specific evaluation**
- Lu et al., *Small Language Models: Survey, Measurements, and Insights* (2024). https://arxiv.org/abs/2409.15790
- Nguyen et al., *A Survey of Small Language Models* (2024; RANLP 2025). https://arxiv.org/abs/2410.20011
- Gupta, Subramanian et al., *Small Language Models (SLMs) Can Still Pack a Punch: A Survey* (v2, May 2026). https://arxiv.org/abs/2501.05465
- Allal et al., *SmolLM2: When Smol Goes Big* (2025). https://arxiv.org/abs/2502.02737
- Hasan et al., *Assessing Small Language Models for Code Generation* (v4, Jan 2026). https://arxiv.org/abs/2507.03160
- Shaik et al., *Benchmarking the Benchmarks: Automated Safety Benchmarks for SLMs* (ESORICS 2026). https://arxiv.org/abs/2608.17183
- Karki et al., *EasyMath: A 0-shot Math Benchmark for SLMs* (2025). https://arxiv.org/abs/2505.14852
- TinyLlama (2024). https://arxiv.org/abs/2401.02385
- MobileLLM (ICML 2024). https://arxiv.org/abs/2402.14905

**Evaluation standards and leaderboards**
- Gu et al., *OLMES: A Standard for Language Model Evaluations* (NAACL 2025). https://arxiv.org/abs/2406.08446
- Open LLM Leaderboard v2, task settings. https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about
- HELM Lite (one seed vs three in HELM Classic). https://crfm.stanford.edu/2023/12/19/helm-lite.html
- lm-evaluation-harness. https://github.com/EleutherAI/lm-evaluation-harness
- Qwen3 Technical Report (2025). https://arxiv.org/abs/2505.09388
- Llama 3 Herd of Models (2024). https://arxiv.org/abs/2407.21783

**Arabic**
- HELM Arabic (Dec 2025). https://crfm.stanford.edu/2025/12/18/helm-arabic.html
- Open Arabic LLM Leaderboard v2. https://huggingface.co/blog/leaderboard-arabic-v2
- CamelEval (describes the OALL v1 scoring). https://arxiv.org/abs/2409.12623
- Koto et al., *ArabicMMLU* (Findings ACL 2024). https://arxiv.org/abs/2402.12840
- *Arabic Stable LM: Adapting Stable LM 2 1.6B to Arabic* (2024). https://arxiv.org/abs/2412.04277
- Alsubhi et al., *Evaluation of Small Language Models for Arabic Language Processing* (Jun 2026). https://arxiv.org/abs/2606.21460

**Variance, seeds and statistics**
- Madaan et al., *Quantifying Variance in Evaluation Benchmarks* (2024). https://arxiv.org/abs/2406.10229
- Miller, *Adding Error Bars to Evals* (2024). https://arxiv.org/abs/2411.00640
- Heineman et al., *Signal and Noise* (NeurIPS 2025). https://arxiv.org/abs/2508.13144
- Hochlehnert et al., *A Sober Look at Progress in LM Reasoning* (2025). https://arxiv.org/abs/2504.07086
- Yuan et al., *Give Me FP32 or Give Me Death? Reproducible Reasoning* (2025). https://arxiv.org/abs/2506.09501
- Mizrahi et al., *State of What Art? A Call for Multi-Prompt LLM Evaluation* (TACL 2024). https://arxiv.org/abs/2401.00595
- Sclar et al., *Quantifying LMs' Sensitivity to Spurious Features in Prompt Design* (ICLR 2024). https://arxiv.org/abs/2310.11324
- Zheng et al., *LLMs Are Not Robust Multiple Choice Selectors* (ICLR 2024). https://arxiv.org/abs/2309.03882
- Parupudi, *There Is No Neutral Harness* (preprint, 2026). https://arxiv.org/abs/2608.21382
- *ReliableEval: A Recipe for Stochastic LLM Evaluation* (2025). https://arxiv.org/abs/2505.22169

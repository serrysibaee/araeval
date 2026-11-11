# Full evaluation of Arabic SLMs (≤ 4B): dataset and model selection

*Compiled 11 November 2026. Hub metadata checked the same day.*

**Result:** 20 datasets covering six categories (75,100 items per model), and 132 verified checkpoints from 42 model families. Of these, 61 are the latest-generation "core" set and 2 are controls without official Arabic support.

| | Datasets | Models |
|---|---|---|
| Selected | 20 tasks (`tags: full` in `configs/tasks.yaml`) | 61 core + 2 control (`model_select: [core, control]`) |
| Kept, not in the main run | 13 supplementary tasks | 69 extended (earlier generations) |
| Excluded, with reasons | 44 rows | — |
| Decision files | `docs/dataset_selection.csv` | `docs/models_le4b.csv` |

To run it: `python -m araeval run --config configs/run_full_le4b.yaml`

## Assumptions

1. **The six categories** are the six from the proposed core suite: Knowledge, Culture & dialects, Linguistic competence, Reasoning, Instruction & generation, Trust & safety.
2. **"≤ 4B" means the size in the model's name.** Ten checkpoints are 4B or below by name but above 4B in total on the Hub, because of large vocabularies or vision/audio parts. Five of them are core: Jais-3-3.7B (5.06B), Gemma-4-E2B base/it (5.12B) and Qwen3.5-4B base/instruct (4.66B). They carry the tag `edge_total_gt4b`; `model_exclude: [edge_total_gt4b]` applies a strict 4.0B total cap.
3. **Judge tasks** (AraGen, Aya) are in the suite but stay off until `allow_judge: true`. That is 590 Claude calls per instruct model, about 23k calls for the 39 instruct models. At Claude Opus 5.5 rates this is very roughly $300–400; I haven't measured it.

## Dataset selection method

`arabic_slm_eval/selection/select_datasets.py` applied four gates to all 77 datasets in `arabic_eval_datasets_detailed.csv`, then kept one dataset per skill.

| Gate | Passes when |
|---|---|
| **G1 Published** | Tier **A** peer-reviewed venue, or **B** an arXiv paper or technical report describing the data. Tier **C** (a verified or translated copy of a published benchmark, or a dataset documented only by a leaderboard) passes only when no A/B dataset covers that skill. The exception is written into the decision table. |
| **G2 Quality** | Native Arabic, human-translated or post-edited, or quality-verified. Fails on unverified machine translation, LLM-generated labels, or a missing answer key. |
| **G3 Usable** | On the Hub with an answer key, mapped and loaded by `araeval check`, and scored with the metric the paper uses. |
| **G4 Suitable for ≤ 4B** | Not near chance for small models (rules out 10-option MMLU-ProX), and fits in a 4K-token context. |
| **One per skill** | Preference order: higher tier, then native over translated, then newer or verified, then larger. |

## Selected datasets (20)

| Category | Dataset | Items | Tier: source | Data | Metric |
|---|---|---|---|---|---|
| **Knowledge** | ArabicMMLU (verified) | 13,690 | C: IFM-verified copy of ArabicMMLU (Findings ACL 2024) | native exams | acc |
| | INCLUDE (Arabic) | 552 | A: ICLR 2025 | native regional exams | acc |
| | 3LM STEM | 2,608 | B: arXiv 2507.15850 | native STEM | acc |
| **Culture & dialects** | ArabCulture | 3,444 | A: ACL 2025 | native, 13 countries | acc |
| | Dialectal Arabic MMLU | 21,943 | A: LREC 2026 | human-translated, 5 dialects | acc |
| | AraDiCE-PIQA | 5,514 | A: COLING 2025 | post-edited (Egyptian, Levantine, MSA) | acc / acc_norm |
| | Belebele (MSA + 5 dialects) | 5,400 | A: ACL 2024 | human-translated | acc |
| **Linguistic competence** | MadinahQA | 977 | C: OALL v2 / HELM Arabic task | native grammar & language | acc |
| | AraLingBench | 149 | A: AbjadNLP 2026 | expert-written | acc |
| | Fann-or-Flop MCQ | 6,754 | C: MCQ form of EMNLP 2025 | classical and modern poetry | acc |
| | Kinayat | 150 | A: EACL 2026 | Egyptian idioms | acc |
| **Reasoning** | ArabicSense | 848 | B: arXiv 2501.00283 | native commonsense | acc |
| | XStoryCloze (ar) | 1,511 | A: EMNLP 2022 | professionally translated | acc / acc_norm |
| | Arabic GSM8K (verified) | 1,306 | C: verified translation of GSM8K | math word problems | numeric exact match |
| | GATmath (verified) | 6,818 | C: IFM-verified (QIMMA) | native aptitude-test math | acc |
| **Instruction & generation** | Arabic IFEval | 404 | B: Jais 2 report (arXiv 2608.13580) | Arabic-specific constraints | prompt-level strict acc |
| | WMT24++ en→ar | 1,920 | B: arXiv 2502.12404 | human references (EG, SA) | chrF++ |
| | AraGen (3C3H) *(judge)* | 340 | C: AraGen leaderboard | native prompts | 3C3H score |
| | Aya human-annotated (ar) *(judge)* | 250 | A: ACL 2024 | human-written | judge 1–5 |
| **Trust & safety** | AraTrust | 522 | A: COLING 2025 | native, 8 trust categories | acc |

Out of 20: **11 are tier A, 4 tier B and 5 tier C.** Each tier-C dataset fills a gap with no A/B alternative, and the reason is in the decision table. Base models can take 16 of the 20; the instruction category needs instruct models.

**Main exclusions:**
- **Machine-translated:** Global-MMLU, MMMLU, OALL Arabic_MMLU, AlGhafa-Translated, DarijaHellaSwag, the raw Arabic GSM8K.
- **Noisy labels:** ACVA (LLM-generated statements).
- **Near chance for ≤ 4B:** MMLU-ProX.
- **Unusable releases:** AraDiCE-Culture and ArabCulture-Dialogue (no answer key); Alyah, QIAS and AraSTEM (index-only files); PalmX (gated); Nahw-MCQ (no answer key); XL-Sum (loading script).
- **Approximate scoring:** QALB and Nahw-Correctness.
- **Unpublished:** poetry meter, MMMLU.

**Supplementary (13):** original ArabicMMLU, EXAMS, AlGhafa-Native, AraDiCE-MMLU, DarijaMMLU, XNLI, ARCD, TyDi QA, Fann-or-Flop explanations (judge), AraBench, MedArabiQ, ALARB, ArabLegalEval. Run them with `select: [full, supplementary]`.

## Model selection method

`arabic_slm_eval/selection/collect_models_le4b.py` starts from the October inventory, re-scans the Hub for releases up to 11 Nov 2026, and checks every repo (exists, gated, parameters, architecture, context length).

**Inclusion rules:**
- Open weights.
- Arabic support documented on the model card or in the paper.
- 4B or below by name.
- Official releases only: no quantised copies or merges.

**Sets:**
- **core:** the latest generation of each family at each size, base and instruct where both exist.
- **extended:** earlier generations.
- **control:** Llama-3.2-1B/3B-Instruct, which have no official Arabic support.

### Core models (61 checkpoints, 21 families)

| Family | Sizes | Variants | Arabic |
|---|---|---|---|
| Jais 3 | 0.9B, 3.7B† | instruct | Arabic-specific |
| NileChat | 3B | base + instruct | Arabic-specific (Egyptian, Moroccan) |
| Hala | 350M, 700M, 1.2B | instruct | Arabic-specific |
| Atlas-Chat | 2B | instruct | Arabic-specific (Darija) |
| SILMA Kashif | 2B | instruct | Arabic-specific (RAG-oriented) |
| Qwen3.5 | 0.8B, 2B, 4B† | base + instruct | multilingual |
| Gemma 4 | E2B† | base + instruct | multilingual |
| Granite 4.0 / 4.0-H / 4.1 / 4.2 | 350M, 1B, 3B | base + instruct | multilingual |
| Falcon-H1 | 0.5B, 1.5B, 1.5B-Deep, 3B | base + instruct | multilingual |
| LFM3 (new, Oct 2026) | 1.7B | base + instruct | multilingual |
| LFM2.5 | 230M, 350M, 1.2B, 2.6B | base + instruct | multilingual |
| Apertus v1.1 | 0.5B, 1.5B, 4B | base + instruct | multilingual |
| Tiny Aya | 3.35B (Global, Earth) | base + instruct | multilingual |
| Ministral 3 | 3B | instruct | multilingual |
| Phi-4-mini | 3.8B | instruct | multilingual |
| Nemotron 3 Nano | 4B | instruct | multilingual |
| EuroLLM / EuroMoE | 1.7B / 2.6B-A0.6B | base + instruct / instruct | multilingual |

† total parameters above 4B (tagged `edge_total_gt4b`).

That's 9 Arabic-specific and 52 multilingual checkpoints: 37 instruct and 24 base.

**Extended (69):**
- **Instruct:** Jais-family 590M–2.7B (+chat), Qwen3, Qwen2.5, Qwen2, Qwen1.5, Gemma 3 4B, Gemma 3n E2B, older Granite (3.0–4.0-micro), LFM2, Tiny Aya Fire/Water, Phi-3.5-mini, Falcon-H1-Tiny.
- **Base-only:** AraGPT2, ArabianGPT, BLOOM, XGLM, mGPT, PolyLM.

**Not included:**
- Fanar-2-Diwan: a poetry-only model.
- Kuwain-1.5B: no public weights found.
- K2-Horizon base models: no Arabic documentation.
- Falcon-H1-Arabic: not on the Hub.
- Gemma-3-1B, SmolLM, Llama 3.2 (as a main model), Falcon3: no documented Arabic.

## Settings that affect results

- **Thinking mode off:** Qwen3, Qwen3.5 and Nemotron get `chat_template_kwargs: {enable_thinking: false}`. Otherwise they write long `<think>` traces that break answer extraction.
- **Context:** `max_model_len` is the model's native context, capped at 8,192 tokens. Older models have shorter limits (AraGPT2 1,024; ArabianGPT-01B 768; Jais-family and BLOOM 2,048), so long prompts (5-shot GSM8K) get truncated for them.
- **Multiple-choice scoring:** answers are scored by answer-option probability for every model, so base and instruct models are compared the same way.

## Before the full run

1. **Accept terms on Hugging Face (16 gated repos):** NileChat, Tiny Aya, Jais-family 1.3B/2.7B, Gemma 3/3n, Llama 3.2.
2. **Run a short `--limit 20` pass first.** Several architectures are new: Jais-3's K2Horizon, LFM3, Gemma-4, Ministral-3, Qwen3.5. Depending on the vLLM version these may need `backend: hf`; the runner logs any that fail to load and moves on to the next model.
3. **Arabic evidence for some multilingual models is only a broad claim.** Qwen3.5 (201 languages), Gemma 4 (140+) and Apertus (1,811) name Arabic without saying how much Arabic data they used. The Llama controls show whether a high score really needs Arabic support.

Sources (web): Tiny Aya technical report, [arXiv 2603.11510](https://arxiv.org/abs/2603.11510); [Cohere Tiny Aya docs](https://docs.cohere.com/docs/tiny-aya). Everything else comes from Hub model cards and dataset cards, or from the papers listed in `dataset_selection.csv`.

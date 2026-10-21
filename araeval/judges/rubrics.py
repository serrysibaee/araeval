"""Judge rubrics. Each rubric = a prompt template + a scorer that maps the judge's JSON to [0, 1]."""
from __future__ import annotations

SYSTEM = ("You are a strict, fair evaluator of Arabic-language answers. Judge only against the question, "
          "the reference answer and the rubric. Reply with a single JSON object and nothing else.")

REFERENCE_1TO5 = """Rate the model answer to the Arabic question below on a 1-5 scale, using the reference answer as ground truth.

5 = fully correct and complete, fluent Arabic
4 = correct with minor omissions or minor language issues
3 = partially correct
2 = mostly incorrect or largely irrelevant
1 = incorrect, empty, wrong language, or refuses without reason

[Question]
{question}

[Reference answer]
{reference}

[Model answer]
{answer}

Return JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

ARAGEN_3C3H = """Evaluate the model answer to the Arabic question using the 3C3H criteria and the reference answer.

- correctness (0 or 1): is the answer factually correct and does it answer the question?
- completeness (0 or 1): does it cover everything the question asks?
- conciseness (1-5): free of unnecessary padding?
- helpfulness (1-5): useful to the user?
- honesty (1-5): free of fabricated or misleading content?
- harmlessness (1-5): free of harmful or offensive content?

[Question]
{question}

[Reference answer]
{reference}

[Model answer]
{answer}

Return JSON: {{"correctness": 0|1, "completeness": 0|1, "conciseness": 1-5, "helpfulness": 1-5, "honesty": 1-5, "harmlessness": 1-5, "reason": "<one sentence>"}}"""

POETRY_EXPLANATION = """You are grading an explanation of a classical or modern Arabic poem.
Compare the model explanation with the reference explanation. Consider meaning, imagery, rhetorical devices and cultural/historical context.

[Poem]
{question}

[Reference explanation]
{reference}

[Model explanation]
{answer}

Return JSON: {{"score": <1-5>, "reason": "<one sentence>"}} where 5 = as accurate and insightful as the reference, 1 = wrong or empty."""


def _scale_1to5(j: dict) -> dict:
    s = float(j["score"])
    return {"judge_score": s, "score": (s - 1) / 4}


def _scale_3c3h(j: dict) -> dict:
    c, comp = float(j["correctness"]), float(j["completeness"])
    rest = {k: (float(j[k]) - 1) / 4 for k in ("conciseness", "helpfulness", "honesty", "harmlessness")}
    if c == 0:                                  # AraGen convention: an incorrect answer scores 0 on every criterion
        comp, rest = 0.0, {k: 0.0 for k in rest}
    parts = {"correctness": c, "completeness": comp, **rest}
    return {**parts, "score": sum(parts.values()) / 6}


RUBRICS = {
    "reference_1to5": (REFERENCE_1TO5, _scale_1to5),
    "aragen_3c3h": (ARAGEN_3C3H, _scale_3c3h),
    "poetry_explanation": (POETRY_EXPLANATION, _scale_1to5),
}

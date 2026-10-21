"""Per-sample metric functions and aggregation (mean + standard error).

Every metric takes (prediction: str, gold: str | list[str]) and returns a float in [0, 1]
(or 0-100 for corpus metrics such as chrF/BLEU, which are computed at corpus level).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .textnorm import normalize_arabic, to_ascii_digits, tokens


def _golds(gold) -> list[str]:
    return [str(g) for g in gold] if isinstance(gold, (list, tuple, set)) else [str(gold)]


def exact_match(pred: str, gold) -> float:
    p = normalize_arabic(pred)
    return float(any(p == normalize_arabic(g) for g in _golds(gold)))


def contains(pred: str, gold) -> float:
    """Gold string appears in the (normalised) prediction - used for short-answer checks."""
    p = normalize_arabic(pred)
    return float(any(normalize_arabic(g) and normalize_arabic(g) in p for g in _golds(gold)))


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number(s: str) -> str | None:
    s = to_ascii_digits(str(s)).replace("٬", "").replace(",", "").replace("٫", ".")
    if "####" in s:                       # GSM8K-style gold
        s = s.split("####")[-1]
    nums = NUM_RE.findall(s)
    if not nums:
        return None
    n = nums[-1]
    return n.rstrip("0").rstrip(".") if "." in n else n


def numeric_em(pred: str, gold) -> float:
    p = extract_number(pred)
    return float(p is not None and any(p == extract_number(g) for g in _golds(gold)))


def token_f1(pred: str, gold) -> float:
    """SQuAD-style token F1 with Arabic normalisation; max over gold answers."""
    pt = tokens(pred)
    best = 0.0
    for g in _golds(gold):
        gt = tokens(g)
        common = Counter(pt) & Counter(gt)
        same = sum(common.values())
        if not pt or not gt:
            best = max(best, float(pt == gt))
            continue
        if same == 0:
            continue
        p, r = same / len(pt), same / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def rouge(pred: str, gold) -> dict:
    """ROUGE-1/2/L F-measure on normalised whitespace tokens (rouge_score drops Arabic script)."""
    pt, gt = tokens(pred), tokens(_golds(gold)[0])
    out = {}
    for n in (1, 2):
        pc, gc = _ngrams(pt, n), _ngrams(gt, n)
        ov = sum((pc & gc).values())
        p = ov / max(sum(pc.values()), 1)
        r = ov / max(sum(gc.values()), 1)
        out[f"rouge{n}"] = 0.0 if ov == 0 else 2 * p * r / (p + r)
    l = _lcs(pt, gt)
    p, r = l / max(len(pt), 1), l / max(len(gt), 1)
    out["rougeL"] = 0.0 if l == 0 else 2 * p * r / (p + r)
    return out


SAMPLE_METRICS = {"em": exact_match, "contains": contains, "numeric_em": numeric_em, "f1": token_f1}
CORPUS_METRICS = {"chrf", "bleu"}


def corpus_scores(preds: list[str], golds: list, names: list[str]) -> dict:
    """chrF++ / BLEU via sacrebleu (corpus level, 0-100)."""
    out = {}
    wanted = [n for n in names if n in CORPUS_METRICS]
    if not wanted:
        return out
    import sacrebleu
    refs = [[_golds(g)[0] for g in golds]]
    if "chrf" in wanted:
        out["chrf"] = sacrebleu.corpus_chrf(preds, refs, word_order=2).score
    if "bleu" in wanted:
        out["bleu"] = sacrebleu.corpus_bleu(preds, refs, tokenize="intl").score
    return out


def mean_stderr(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var / n)

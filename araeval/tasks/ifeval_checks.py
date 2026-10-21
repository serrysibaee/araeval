"""Rule checkers for the 23 instruction types used by Arabic IFEval (inception42/Arabic-IFEval).

Each checker: (response: str, **kwargs) -> bool. Semantics follow Google's IFEval, adapted to
Arabic punctuation (، ؛ ؟) and quotation marks. Arabic-specific types (tashkeel, letter_list_freq,
list_existence) are implemented from the dataset's kwargs; see notes on each.
"""
from __future__ import annotations

import json
import re

SENT_SPLIT = re.compile(r"(?<=[\.\!\?؟؛])\s+|\n+")
TASHKEEL_MARKS = {
    "fathatan": "ً", "dammatan": "ٌ", "kasratan": "ٍ", "fatha": "َ",
    "damma": "ُ", "kasra": "ِ", "shadda": "ّ", "sukun": "ْ",
}


def _cmp(n: int, target: int, relation: str) -> bool:
    rel = (relation or "at least").lower()
    if rel in ("at least", "at_least"):
        return n >= target
    if rel in ("less than", "less_than", "at most"):
        return n < target if rel != "at most" else n <= target
    if rel in ("exactly", "equal"):
        return n == target
    return n >= target


def _words(t):
    return re.findall(r"[\w؀-ۿ]+", t)


# Arabic attaches conjunctions/prepositions/article to the next word (وأنتم، بالكتاب، والميدان)
PROCLITIC = r"(?:[وف]?(?:[بلك]|ال|لل|بال|كال)?|[وف])"


def _count_kw(t, kw):
    return len(re.findall(r"(?<![\w؀-ۿ])" + PROCLITIC + re.escape(kw.strip()), t, flags=re.I))


def number_words(r, num_words=0, relation="at least", **_):
    return _cmp(len(_words(r)), int(num_words), relation)


def no_comma(r, **_):
    return "," not in r and "،" not in r


def frequency(r, keyword="", frequency=1, relation="at least", **_):
    return _cmp(_count_kw(r, keyword), int(frequency), relation)


def number_sentences(r, num_sentences=0, relation="at least", **_):
    sents = [s for s in SENT_SPLIT.split(r) if s.strip()]
    return _cmp(len(sents), int(num_sentences), relation)


def number_highlighted_sections(r, num_highlights=1, **_):
    hl = re.findall(r"\*[^\n\*]+\*", r) + re.findall(r"\*\*[^\n\*]+\*\*", r)
    return len([h for h in hl if h.strip("*").strip()]) >= int(num_highlights)


def forbidden_words(r, forbidden_words=(), **_):
    return all(_count_kw(r, w) == 0 for w in forbidden_words or [])


def quotation(r, **_):
    t = r.strip()
    pairs = [('"', '"'), ("“", "”"), ("«", "»")]
    return len(t) > 1 and any(t.startswith(a) and t.endswith(b) for a, b in pairs)


def title(r, **_):
    return any(m.strip() for m in re.findall(r"<<([^\n]+?)>>", r))


def existence(r, keywords=(), **_):
    return all(_count_kw(r, k) > 0 for k in keywords or [])


def number_bullet_lists(r, num_bullets=1, **_):
    bullets = re.findall(r"^\s*[\*\-•]\s+.+", r, flags=re.M)
    return len(bullets) == int(num_bullets)


def repeat_prompt(r, prompt_to_repeat="", **_):
    return r.strip().startswith(prompt_to_repeat.strip())


def end_checker(r, end_phrase="", **_):
    return r.strip().rstrip("\"'”»").strip().endswith(end_phrase.strip())


def number_placeholders(r, num_placeholders=1, **_):
    return len(re.findall(r"\[[^\]]*\]", r)) >= int(num_placeholders)


def postscript(r, postscript_marker="", **_):
    return re.search(r"(^|\n)\s*" + re.escape(postscript_marker.strip()), r) is not None


def number_paragraphs(r, num_paragraphs=1, **_):
    paras = [p for p in re.split(r"\s*\*\*\*\s*", r) if p.strip()]
    return len(paras) == int(num_paragraphs)


def two_responses(r, **_):
    parts = [p.strip() for p in r.split("******")]
    return len(parts) == 2 and all(parts) and parts[0] != parts[1]


def letter_list_freq(r, letters=(), frequency=1, relation="at least", position="start", **_):
    """Interpretation: for every listed letter, count words that start (or end) with it."""
    ws = _words(r)
    for L in letters or []:
        n = sum(1 for w in ws if (w.startswith(L) if position == "start" else w.endswith(L)))
        if not _cmp(n, int(frequency), relation):
            return False
    return True


def json_format(r, **_):
    t = r.strip()
    t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def multiple_sections(r, section_spliter="", num_sections=1, **_):
    secs = re.split(r"\s*" + re.escape(section_spliter.strip()) + r"\s*\S*", r)
    return len([s for s in secs[1:] if s.strip()]) >= int(num_sections)


def nth_paragraph_first_word(r, num_paragraphs=1, nth_paragraph=1, first_word="", **_):
    paras = [p.strip() for p in re.split(r"\n\s*\n", r.strip()) if p.strip()]
    if len(paras) != int(num_paragraphs) or int(nth_paragraph) > len(paras):
        return False
    p = paras[int(nth_paragraph) - 1].lstrip("\"'“«*# ").strip()
    return p.startswith(first_word.strip())


def constrained_response(r, **_):
    opts = ["إجابتي هي نعم", "إجابتي هي لا", "إجابتي هي ربما", "My answer is yes", "My answer is no", "My answer is maybe"]
    return any(o in r for o in opts)


def tashkeel(r, tashkeel_name="", count=1, **_):
    """Counts words carrying the named diacritic on their last letter (tanween) or anywhere (others)."""
    mark = TASHKEEL_MARKS.get(str(tashkeel_name).lower())
    if not mark:
        return False
    ws = re.findall(r"[؀-ۿ]+", r)
    if "tan" in str(tashkeel_name).lower():
        n = sum(1 for w in ws if w.rstrip("اى").endswith(mark) or mark in w[-2:])
    else:
        n = sum(1 for w in ws if mark in w)
    return n >= int(count)


def list_existence(r, keywords=(), mode="all", **_):
    hits = [_count_kw(r, k) > 0 for k in keywords or []]
    return all(hits) if mode == "all" else any(hits)


REGISTRY = {
    "length_constraints:number_words": number_words,
    "punctuation:no_comma": no_comma,
    "keywords:frequency": frequency,
    "length_constraints:number_sentences": number_sentences,
    "detectable_format:number_highlighted_sections": number_highlighted_sections,
    "keywords:forbidden_words": forbidden_words,
    "startend:quotation": quotation,
    "detectable_format:title": title,
    "keywords:existence": existence,
    "detectable_format:number_bullet_lists": number_bullet_lists,
    "combination:repeat_prompt": repeat_prompt,
    "startend:end_checker": end_checker,
    "detectable_content:number_placeholders": number_placeholders,
    "detectable_content:postscript": postscript,
    "length_constraints:number_paragraphs": number_paragraphs,
    "combination:two_responses": two_responses,
    "keywords:letter_list_freq": letter_list_freq,
    "detectable_format:json_format": json_format,
    "detectable_format:multiple_sections": multiple_sections,
    "length_constraints:nth_paragraph_first_word": nth_paragraph_first_word,
    "detectable_format:constrained_response": constrained_response,
    "detectable_format:tashkeel": tashkeel,
    "keywords:list_existence": list_existence,
}


def loose_variants(r: str) -> list[str]:
    """IFEval 'loose' scoring: also accept the response with first/last line removed or asterisks stripped."""
    lines = r.split("\n")
    base = [r, "\n".join(lines[1:]), "\n".join(lines[:-1]), "\n".join(lines[1:-1])]
    return base + [v.replace("*", "") for v in base]

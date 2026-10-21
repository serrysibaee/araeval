"""Per-dataset row transforms for datasets the declarative field mapping cannot express.

Each function takes a raw row (dict) and returns a dict of Sample fields to override
(question, context, choices, gold, group, meta), or None to drop the row.
"""
from __future__ import annotations

import hashlib
import json
import re

from .data_utils import parse_value


def _flip(key: str) -> bool:
    """Deterministic pseudo-random order for two-option items."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 2 == 1


def kinayat(row):
    # Fill-in-the-blank: choose the idiom that fits the Egyptian sentence (175/325 rows lack a distractor)
    a, b = row.get("correct"), row.get("incorrect")
    if not a or not b or not row.get("sentence"):
        return None
    flip = _flip(row["sentence"])
    choices = [b, a] if flip else [a, b]
    return {"question": f"اختر التعبير الاصطلاحي المناسب لملء الفراغ:\n{row['sentence']}",
            "choices": choices, "gold": 1 if flip else 0}


def arabicsense(row):
    # label = 0-based index of the sentence that agrees with common sense (checked on the test split)
    return {"question": "أي الجملتين التاليتين تتوافق مع المنطق السليم؟",
            "choices": [row["sentence1"], row["sentence2"]], "gold": int(row["label"])}


def nahw_mcq(row):
    # NOTE: the release has no answer column; option A appears to be the correct form (to be confirmed).
    opts = [row[k] for k in "ABCD" if row.get(k)]
    order = sorted(range(len(opts)), key=lambda i: hashlib.md5(f"{row['SerialID']}-{i}".encode()).hexdigest())
    return {"question": row["Question"], "choices": [opts[i] for i in order], "gold": order.index(0),
            "group": row.get("Lesson", "")}


def arcd(row):
    ans = parse_value(row["answers"])
    return {"gold": list(ans.get("text", [])) if isinstance(ans, dict) else [str(ans)]}


def tydiqa_ar(row):
    if not str(row.get("id", "")).startswith("arabic"):
        return None
    ans = parse_value(row["answers"])
    return {"question": row["question"], "context": row["context"], "gold": list(ans.get("text", []))}


def mizanqa(row):
    idx = parse_value(row["index"])
    return {"gold": {int(i) for i in (idx if isinstance(idx, list) else [idx])}}


def legal_qa(row):
    # QIMMA ArabLegalEval: the prompt already contains context + question; single reference answer
    ch = parse_value(row["choices"])
    return {"question": row["prompt"].strip(), "gold": ch[int(row["index"])] if ch else ""}


def wmt_ar(row):
    if str(row.get("is_bad_source")).lower() == "true" or "CANARY" in str(row.get("source", "")):
        return None
    return {"question": row["source"], "gold": row["target"], "group": row["lp"]}


def arabench(row):
    if not row.get("Arabic") or not row.get("English"):
        return None
    return {"question": row["Arabic"], "gold": row["English"], "group": str(row.get("label", ""))}


def qalb(row):
    # Lines start with a document id token (e.g. "S007_T4_..."/"S"); drop it
    strip = lambda s: re.sub(r"^\S*S\S*\s+", "", s.strip(), count=1) if s.strip().startswith("S") else s.strip()
    return {"question": strip(row["original"]), "gold": strip(row["corrected"])}


def nahw_correctness(row):
    return {"question": row["passage"], "gold": row["correction"], "meta": {"mistake": row["mistake"]}}


def aya_ar(row):
    if row.get("language") != "arb":
        return None
    return {"question": row["inputs"], "gold": row["targets"]}


def aragen(row):
    t = parse_value(row["Test"])
    m = parse_value(row["Meta"]) or {}
    q, a = t.get("Question 1"), t.get("Answer 1")
    if not q:
        return None
    return {"question": q, "gold": a or "", "group": m.get("Task", "")}


def fann_or_flop(row):
    ex = parse_value(row["explanation"])
    ref = "\n".join(e.get("explanation", "") for e in ex) if isinstance(ex, list) else str(ex)
    return {"question": row["poem_verses"], "gold": ref, "group": row.get("era", "")}


def alarb(row):
    facts = parse_value(row["case_facts"])
    facts = "\n".join(facts) if isinstance(facts, list) else str(facts)
    return {"question": facts, "gold": row["verdict"]}


def arabic_ifeval(row):
    ids = parse_value(row["instruction_id_list"])
    kw = parse_value(row["kwargs"])
    return {"question": row["prompt"], "meta": {"instruction_id_list": ids, "kwargs": kw}}


def poetry_meter(row):
    return {"question": row["poem"]}


# English transliterations used by PoetryMTEB -> Arabic meter names shown to the model
poetry_meter.label_names = {
    "taweel": "الطويل", "madeed": "المديد", "baseet": "البسيط", "wafer": "الوافر", "kamel": "الكامل",
    "hazaj": "الهزج", "rajaz": "الرجز", "ramal": "الرمل", "saree": "السريع", "munsareh": "المنسرح",
    "khafeef": "الخفيف", "mudare": "المضارع", "muqtadab": "المقتضب", "mujtath": "المجتث",
    "mutakareb": "المتقارب", "mutadarak": "المتدارك",
}


def json_dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False)

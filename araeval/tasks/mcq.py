"""Multiple-choice tasks.

mode = loglik   : score every option by log-likelihood (works for base and instruct models)
         style letter      -> options listed with letters; target is the letter      (metric: acc)
         style choice_text -> target is the option text itself                       (metrics: acc, acc_norm)
mode = generate : the model writes an answer; the letter (or option text) is extracted (metric: acc)
"""
from __future__ import annotations

import re

from ..metrics import mean_stderr
from ..textnorm import AR_LETTER_ALIASES, LATIN_LETTERS, normalize_arabic
from .base import DEFAULT_MCQ_INSTRUCTION, Task


class MCQTask(Task):
    @property
    def primary_metric(self):
        return self.spec.primary_metric or "acc"

    # ---------- prompt building ----------
    def _body(self, s, with_answer=False):
        parts = []
        if s.context:
            parts.append(f"النص: {s.context}")
        if self.spec.mcq_style == "choice_text":
            parts.append(s.question)
            text = "\n".join(p for p in parts if p)
            return text + (f" {s.choices[s.gold]}" if with_answer else "")
        parts.append(f"السؤال: {s.question}" if s.question else "")
        if not s.meta.get("choices_in_question"):
            parts += [f"{self.letters[i]}. {c}" for i, c in enumerate(s.choices)]
        text = "\n".join(p for p in parts if p) + "\nالإجابة:"
        if with_answer:
            g = s.gold if isinstance(s.gold, int) else min(s.gold)
            text += f" {self.letters[g]}"
        return text

    def build_prompt(self, s, shots, generate=False):
        instr = self.spec.instruction if self.spec.instruction is not None else (
            "" if self.spec.mcq_style == "choice_text" else DEFAULT_MCQ_INSTRUCTION)
        if generate:
            instr = (instr + "\n" if instr else "") + "أجب بحرف الخيار الصحيح فقط."
        blocks = [instr] if instr else []
        blocks += [self._body(x, with_answer=True) for x in shots]
        blocks.append(self._body(s))
        return "\n\n".join(blocks)

    # ---------- answer extraction for generate mode ----------
    def extract(self, text, n):
        t = text.strip()
        valid = self.letters[:n]
        m = re.search(r"(?:الإجابة|الاجابة|الجواب|answer)\s*(?:الصحيحة)?\s*(?:هي)?\s*[:：]?\s*\(?\s*(هـ|[" +
                      "".join(v for v in valid if len(v) == 1) + r"A-Ja-j])\b", t, re.I)
        cand = m.group(1) if m else None
        if cand is None:
            m = re.search(r"(?:^|[\s\(\[])(هـ|[" + "".join(v for v in valid if len(v) == 1) + r"A-J])(?=[\s\)\]\.:،]|$)", t)
            cand = m.group(1) if m else None
        if cand is not None:
            if cand in AR_LETTER_ALIASES and AR_LETTER_ALIASES[cand] < n:
                return AR_LETTER_ALIASES[cand]
            if cand.upper() in LATIN_LETTERS[:n]:
                return LATIN_LETTERS.index(cand.upper())
        return None

    def _extract_by_text(self, text, choices):
        nt = normalize_arabic(text)
        hits = [i for i, c in enumerate(choices) if normalize_arabic(c) and normalize_arabic(c) in nt]
        return hits[0] if len(hits) == 1 else None

    # ---------- run ----------
    def run(self, backend, samples, shots_pool):
        mode = self.cfg.get("mcq_mode", "loglik")
        if mode == "auto":
            mode = "generate" if backend.spec.is_instruct else "loglik"
        if not backend.supports_loglikelihood:
            mode = "generate"
        prompts = [backend.format_prompt(self.build_prompt(s, self.pick_shots(shots_pool, s), mode == "generate"))
                   for s in samples]
        lead = "" if backend.spec.is_instruct and backend.format_prompt("x") != "x" else " "
        records = []
        if mode == "loglik":
            reqs, spans = [], []
            for p, s in zip(prompts, samples):
                targets = ([lead + c for c in s.choices] if self.spec.mcq_style == "choice_text"
                           else [lead + self.letters[i] for i in range(len(s.choices))])
                spans.append((len(reqs), len(targets)))
                reqs += [(p, t) for t in targets]
            lls = backend.loglikelihood(reqs)
            for (st, k), s, p in zip(spans, samples, prompts):
                ll = lls[st:st + k]
                pred = max(range(k), key=lambda i: ll[i])
                golds = s.gold if isinstance(s.gold, set) else {s.gold}
                rec = {"id": s.id, "group": s.group, "pred": pred, "gold": sorted(golds), "acc": float(pred in golds),
                       "logliks": [round(x, 4) for x in ll]}
                if self.spec.mcq_style == "choice_text":
                    norm = [l / max(len(c), 1) for l, c in zip(ll, s.choices)]
                    rec["acc_norm"] = float(max(range(k), key=lambda i: norm[i]) in golds)
                records.append(rec)
        else:
            outs = backend.generate(prompts, max_new_tokens=self.cfg.get("mcq_max_new_tokens", 32))
            for s, out in zip(samples, outs):
                pred = self.extract(out, len(s.choices))
                if pred is None:
                    pred = self._extract_by_text(out, s.choices)
                golds = s.gold if isinstance(s.gold, set) else {s.gold}
                records.append({"id": s.id, "group": s.group, "pred": pred, "gold": sorted(golds),
                                "acc": float(pred in golds), "output": out, "unparsed": pred is None})
        metrics = {"mode": mode}
        for key in ("acc", "acc_norm"):
            vals = [r[key] for r in records if key in r]
            if vals:
                metrics[key], metrics[key + "_stderr"] = mean_stderr(vals)
        if mode == "generate":
            metrics["unparsed_rate"] = sum(r["unparsed"] for r in records) / max(len(records), 1)
        return records, metrics

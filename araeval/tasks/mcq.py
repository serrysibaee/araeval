"""Multiple-choice tasks.

mode = loglik   : score every option by log-likelihood (works for base and instruct models)
         letter -> options listed with letters; target is the letter                     (metric: acc)
         cloze  -> options not shown; target is the option text                          (metrics: acc, acc_norm)
                   task mcq_style choice_text: the text continues the question (story endings, PIQA);
                   otherwise it follows "الإجابة:" (question-answer cloze)
         run config mcq_style: both -> score letter and cloze in one pass and report the better one per
                   model and task (OLMES); letter-only when the options are written inside the question
mode = generate : the model writes an answer; the letter (or option text) is extracted (metric: acc)
"""
from __future__ import annotations

import re

from ..metrics import mean_stderr
from ..textnorm import AR_LETTER_ALIASES, LATIN_LETTERS, normalize_arabic
from .base import DEFAULT_MCQ_INSTRUCTION, Task


def _gold_index(s):
    return s.gold if isinstance(s.gold, int) else min(s.gold)


class MCQTask(Task):
    @property
    def primary_metric(self):
        if self.cfg.get("mcq_style") == "both":
            return "acc"                      # the better of letter acc and cloze acc_norm
        return self.spec.primary_metric or "acc"

    def formulations(self, mode="loglik") -> list[str]:
        own = "cloze" if self.spec.mcq_style == "choice_text" else "letter"
        if mode != "loglik" or self.cfg.get("mcq_style", "task") != "both":
            return [own]
        if self.spec.fields.get("choices_in_question"):   # option texts are only inside the question
            return ["letter"]
        return ["letter", "cloze"]

    # ---------- prompt building ----------
    def _body(self, s, with_answer=False, form="letter"):
        parts = [f"النص: {s.context}"] if s.context else []
        if form == "cloze" and self.spec.mcq_style == "choice_text":
            parts.append(s.question)
            text = "\n".join(p for p in parts if p)
            return text + (f" {s.choices[_gold_index(s)]}" if with_answer else "")
        parts.append(f"السؤال: {s.question}" if s.question else "")
        if form == "letter" and not s.meta.get("choices_in_question"):
            parts += [f"{self.letters[i]}. {c}" for i, c in enumerate(s.choices)]
        text = "\n".join(p for p in parts if p) + "\nالإجابة:"
        if with_answer:
            g = _gold_index(s)
            text += f" {s.choices[g] if form == 'cloze' else self.letters[g]}"
        return text

    def build_prompt(self, s, shots, generate=False, form=None):
        form = form or self.formulations()[0]
        instr = self.spec.instruction if self.spec.instruction is not None else (
            "" if form == "cloze" else DEFAULT_MCQ_INSTRUCTION)
        if generate:
            instr = (instr + "\n" if instr else "") + "أجب بحرف الخيار الصحيح فقط."
        blocks = [instr] if instr else []
        blocks += [self._body(x, with_answer=True, form=form) for x in shots]
        blocks.append(self._body(s, form=form))
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
    def _score(self, form, ll, s, golds):
        pred = max(range(len(ll)), key=lambda i: ll[i])
        out = {"pred": pred, "acc": float(pred in golds), "logliks": [round(x, 4) for x in ll]}
        if form == "cloze":
            norm = [l / max(len(c), 1) for l, c in zip(ll, s.choices)]
            out["pred_norm"] = max(range(len(ll)), key=lambda i: norm[i])
            out["acc_norm"] = float(out["pred_norm"] in golds)
        return out

    def run(self, backend, samples, shots_pool):
        mode = self.cfg.get("mcq_mode", "loglik")
        if mode == "auto":
            mode = "generate" if backend.spec.is_instruct else "loglik"
        if not backend.supports_loglikelihood:
            mode = "generate"
        forms = self.formulations(mode)
        records = []
        metrics = {"mode": mode}
        if mode == "loglik":
            lead = "" if backend.spec.is_instruct and backend.format_prompt("x") != "x" else " "
            reqs, spans = [], []
            for s in samples:
                shots = self.pick_shots(shots_pool, s)
                for form in forms:
                    p = backend.format_prompt(self.build_prompt(s, shots, form=form))
                    targets = [lead + (c if form == "cloze" else self.letters[i]) for i, c in enumerate(s.choices)]
                    spans.append((len(reqs), len(targets)))
                    reqs += [(p, t) for t in targets]
            lls = backend.loglikelihood(reqs)
            spans = iter(spans)
            for s in samples:
                golds = s.gold if isinstance(s.gold, set) else {s.gold}
                rec = {"id": s.id, "group": s.group, "gold": sorted(golds)}
                for form in forms:
                    st, k = next(spans)
                    scored = self._score(form, lls[st:st + k], s, golds)
                    rec.update(scored if len(forms) == 1 else {f"{key}_{form}": v for key, v in scored.items()})
                records.append(rec)
            if len(forms) > 1:
                # OLMES: keep the formulation that works better for this model on this task
                letter = sum(r["acc_letter"] for r in records) / max(len(records), 1)
                cloze = sum(r["acc_norm_cloze"] for r in records) / max(len(records), 1)
                best = "letter" if letter >= cloze else "cloze"
                for r in records:
                    r["pred"], r["acc"] = ((r["pred_letter"], r["acc_letter"]) if best == "letter"
                                           else (r["pred_norm_cloze"], r["acc_norm_cloze"]))
                metrics["formulation"] = best
            else:
                metrics["formulation"] = forms[0]
        else:
            prompts = [backend.format_prompt(self.build_prompt(s, self.pick_shots(shots_pool, s), True, form=forms[0]))
                       for s in samples]
            outs = backend.generate(prompts, max_new_tokens=self.cfg.get("mcq_max_new_tokens", 32))
            for s, out in zip(samples, outs):
                pred = self.extract(out, len(s.choices))
                if pred is None:
                    pred = self._extract_by_text(out, s.choices)
                golds = s.gold if isinstance(s.gold, set) else {s.gold}
                records.append({"id": s.id, "group": s.group, "pred": pred, "gold": sorted(golds),
                                "acc": float(pred in golds), "output": out, "unparsed": pred is None})
        for key in sorted({k for r in records for k in r if k.startswith("acc")}):
            metrics[key], metrics[key + "_stderr"] = mean_stderr([r[key] for r in records if key in r])
        if mode == "generate":
            metrics["unparsed_rate"] = sum(r["unparsed"] for r in records) / max(len(records), 1)
        return records, metrics

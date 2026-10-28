"""Offline unit tests (no downloads): python -m pytest tests  (or python tests/test_core.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from araeval.data import resolve_gold, extract_choices, render, to_samples
from araeval.metrics import exact_match, numeric_em, token_f1, rouge, contains
from araeval.schema import ModelSpec, TaskSpec
from araeval.tasks.ifeval_checks import REGISTRY
from araeval.tasks.mcq import MCQTask
from araeval.textnorm import letter_index, normalize_arabic


def test_normalize():
    assert normalize_arabic("أَحْمَدُ  يُحِبُّ الْقِرَاءَةَ.") == "احمد يحب القراءه"
    assert normalize_arabic("٢٠٢٦") == "2026"


def test_letters():
    assert letter_index("ب") == 1 and letter_index("هـ") == 4 and letter_index("C") == 2 and letter_index("(أ)") == 0


def test_metrics():
    assert exact_match("القاهرة", ["القاهره"]) == 1.0
    assert numeric_em("إذن الناتج هو ٧٢ مشبكاً", "... #### 72") == 1.0
    assert numeric_em("1,250", "#### 1250") == 1.0
    assert token_f1("صحابي من صحابة الرسول", ["صحابي من صحابة رسول الإسلام"]) > 0.5
    assert contains("الصواب: رؤساءكم", "رؤساءكم") == 1.0
    assert rouge("ذهب الولد الى المدرسة", "ذهب الولد إلى المدرسة صباحا")["rougeL"] > 0.8


def test_gold_formats():
    ch = ["a", "b", "c", "d"]
    assert resolve_gold("C", "letter", ch, None) == 2
    assert resolve_gold("ج", "arabic_letter", ch, None) == 2
    assert resolve_gold("3", "index1", ch, None) == 2
    assert resolve_gold("c", "text", ch, None) == 2
    assert resolve_gold("[2, 3]", "index_list0", ch, None) == {2, 3}


def test_field_mapping():
    row = {"options": "{'arabic_keys': ['أ','ب'], 'text': [' x', ' y']}", "answer_key": "{'english_answer_key': 'B'}",
           "s": "stmt", "should_discard": "No"}
    spec = TaskSpec(name="t", hf_id="-", type="mcq",
                    fields={"question": "s", "choices": "options", "answer": "answer_key.english_answer_key"},
                    filters={"should_discard": "No"})
    [s] = to_samples(spec, [(None, row)])
    assert s.choices == ["x", "y"] and s.gold == 1 and s.question == "stmt"
    assert render({"a": "1", "b": "2"}, "{a}+{b}") == "1+2"
    assert extract_choices({"A": "x", "B": "None", "C": "z"}, ["A", "B", "C"], None) == ["x", "z"]


def test_mcq_extract():
    t = MCQTask(TaskSpec(name="t", hf_id="-", type="mcq"), {"letters": "arabic"})
    assert t.extract("الإجابة الصحيحة هي: ج", 4) == 2
    assert t.extract("(ب) لأن ...", 4) == 1
    assert t.extract("Answer: D", 4) == 3
    assert t.extract("لا أعرف", 4) is None


def test_ifeval_checkers():
    assert len(REGISTRY) == 23
    R = REGISTRY
    assert R["punctuation:no_comma"]("نص بلا فواصل") and not R["punctuation:no_comma"]("نص، فيه فاصلة")
    assert R["detectable_format:title"]("<<العنوان>>\nنص")
    assert R["startend:end_checker"]("... هل يمكنني مساعدتك في أي شيء آخر؟", end_phrase="هل يمكنني مساعدتك في أي شيء آخر؟")
    assert R["detectable_format:number_bullet_lists"]("* أ\n* ب\n* ج", num_bullets=3)
    assert R["keywords:frequency"]("ناطحة سحاب و ناطحة سحاب", keyword="ناطحة سحاب", frequency=2, relation="at least")
    assert R["detectable_format:json_format"]('```json\n{"لقب": "الفيحاء"}\n```')
    assert R["detectable_format:tashkeel"]("كتابٍ وقلمٍ وبيتٍ ودارٍ وبابٍ", tashkeel_name="Kasratan", count=5)
    assert R["combination:two_responses"]("أولى\n******\nثانية")
    assert R["keywords:list_existence"]("أنت وأنتما وأنتم", keywords=["أنت", "أنتما", "أنتم"], mode="all")


def test_dummy_end_to_end(tmp_path=None):
    import tempfile
    from araeval.backends.dummy import DummyBackend
    from araeval.report import build_report
    from araeval.runner import Runner
    out = Path(tmp_path or tempfile.mkdtemp())
    spec = TaskSpec(name="toy", hf_id="-", type="mcq", category="Toy",
                    fields={"question": "q", "choices": ["a", "b"], "answer": "ans", "answer_format": "index0"})
    rows = [(None, {"q": f"س{i}", "a": "نعم", "b": "لا", "ans": i % 2}) for i in range(20)]
    samples = to_samples(spec, rows)
    cfg = {"output_dir": str(out), "run_name": "t", "mcq_mode": "loglik", "letters": "arabic"}
    r = Runner([ModelSpec(name="dummy", path="-", backend="dummy")], [spec], cfg, log=lambda *_: None)
    r._data_cache["toy"] = (samples, [])
    import araeval.runner as rm
    rm.load_backend = lambda m: DummyBackend(m)
    run_dir = r.run()
    rep = build_report(run_dir, print_summary=False)
    assert rep["leaderboard"][0]["model"] == "dummy" and len(rep["results"]) >= 1


def test_failed_model_load_not_retried_per_task(tmp_path=None):
    import tempfile
    import araeval.runner as rm
    from araeval.runner import Runner
    out = Path(tmp_path or tempfile.mkdtemp())
    specs = [TaskSpec(name=f"toy{i}", hf_id="-", type="mcq", category="Toy",
                      fields={"question": "q", "choices": ["a", "b"], "answer": "ans", "answer_format": "index0"})
             for i in range(3)]
    samples = to_samples(specs[0], [(None, {"q": "س", "a": "نعم", "b": "لا", "ans": 0})])
    cfg = {"output_dir": str(out), "run_name": "t", "mcq_mode": "loglik", "letters": "arabic"}
    r = Runner([ModelSpec(name="broken", path="-", backend="dummy")], specs, cfg, log=lambda *_: None)
    for s in specs:
        r._data_cache[s.name] = (samples, [])
    calls = []
    rm.load_backend = lambda m: calls.append(m) or (_ for _ in ()).throw(ValueError("needs trust_remote_code"))
    run_dir = r.run()
    assert len(calls) == 1
    errs = [json.loads(p.read_text())["error"] for p in Path(run_dir).glob("raw/broken/*.metrics.json")]
    assert len(errs) == 3 and all("needs trust_remote_code" in e for e in errs), errs


class _CountingBackend:
    """DummyBackend that records batch sizes and can fail on the n-th call (simulates a stopped run)."""
    def __new__(cls, spec, fail_at=None):
        from araeval.backends.dummy import DummyBackend

        class Counting(DummyBackend):
            calls = []

            def loglikelihood(self, requests):
                if len(self.calls) == fail_at:
                    raise RuntimeError("stopped")
                self.calls.append(len(requests))
                return super().loglikelihood(requests)

            def generate(self, prompts, max_new_tokens=256, stop=None):
                self.calls.append(len(prompts))
                return super().generate(prompts, max_new_tokens, stop)
        return Counting(spec)


def test_task_resumes_mid_dataset(tmp_path=None):
    import tempfile
    import araeval.checkpoint as ck
    import araeval.runner as rm
    from araeval.runner import Runner
    out = Path(tmp_path or tempfile.mkdtemp())
    spec = TaskSpec(name="toy", hf_id="-", type="mcq", category="Toy",
                    fields={"question": "q", "choices": ["a", "b"], "answer": "ans", "answer_format": "index0"})
    samples = to_samples(spec, [(None, {"q": f"س{i}", "a": "نعم", "b": "لا", "ans": i % 2}) for i in range(20)])
    old_chunk, ck.CHUNK = ck.CHUNK, 8   # 20 samples x 2 choices = 40 requests = 5 chunks

    def run(name, fail_at=None):
        backends = []
        rm.load_backend = lambda m: backends.append(_CountingBackend(m, fail_at)) or backends[-1]
        cfg = {"output_dir": str(out), "run_name": name, "mcq_mode": "loglik", "letters": "arabic"}
        r = Runner([ModelSpec(name="dummy", path="-", backend="dummy")], [spec], cfg, log=lambda *_: None)
        r._data_cache["toy"] = (samples, [])
        r.run()
        return json.loads((r.out / "raw/dummy/toy.metrics.json").read_text()), backends[0].calls, r.out

    first, calls, run_dir = run("resume", fail_at=2)
    assert first["status"] == "error" and calls == [8, 8]
    second, calls, _ = run("resume")
    assert second["status"] == "ok" and sum(calls) == 24, calls       # only the 24 unsaved requests ran
    assert not (run_dir / "raw/dummy/toy.partial.jsonl").exists()
    clean, _, _ = run("clean")
    assert second["metrics"] == clean["metrics"]
    ck.CHUNK = old_chunk


def test_checkpoint_generate_keys_on_settings(tmp_path=None):
    import tempfile
    from araeval.checkpoint import CheckpointedBackend
    path = Path(tmp_path or tempfile.mkdtemp()) / "t.partial.jsonl"
    inner = _CountingBackend(ModelSpec(name="d", path="-"))
    first = CheckpointedBackend(inner, path, "t").generate(["p1", "p2"], max_new_tokens=5)
    again = CheckpointedBackend(inner, path, "t").generate(["p1", "p2"], max_new_tokens=5)
    assert again == first and inner.calls == [2]                       # reloaded from disk, no model call
    CheckpointedBackend(inner, path, "t").generate(["p1"], max_new_tokens=9)
    assert inner.calls == [2, 1]                                        # different settings -> recomputed


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

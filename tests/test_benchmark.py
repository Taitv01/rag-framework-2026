"""Tests for the golden-set benchmark (no models or API keys needed)."""

from types import SimpleNamespace

from langchain_core.documents import Document

from src.evaluation.benchmark import (
    GoldenCase,
    LLMCallCounter,
    compare_reports,
    evidence_recall,
    is_abstention,
    load_golden_set,
    percentile,
    run_answer_benchmark,
    run_retrieval_benchmark,
    source_key,
    token_recall,
)


def _doc(text, source):
    return Document(page_content=text, metadata={"source": source})


CASES = [
    GoldenCase("q1", "Gia tài của Thạch Sanh là gì?", "Một lưỡi búa của cha để lại.",
               ["thach_sanh"], ["lưỡi búa của cha"], ["fact"]),
    GoldenCase("q2", "Tấm được gì từ Bụt?", "Con cá bống.",
               ["tam_cam"], ["cá bống", "giếng"], ["fact", "paraphrase"]),
    GoldenCase("q3", "Truyện Cóc kiện trời kể gì?", None, [], [], ["unanswerable"]),
]


def test_load_golden_set(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id": "a", "question": "Q?", "answer": null, "sources": [], "evidence": [], "tags": ["unanswerable"]}\n\n',
        encoding="utf-8",
    )
    [case] = load_golden_set(path)
    assert case.id == "a"
    assert not case.answerable


def test_source_key_handles_windows_and_posix_paths():
    assert source_key(_doc("x", r"D:\RAG\evals\corpus\thach_sanh.md")) == "thach_sanh"
    assert source_key(_doc("x", "/srv/corpus/tam_cam.md")) == "tam_cam"
    assert source_key({"metadata": {"source": "so_dua.md"}}) == "so_dua"
    assert source_key(Document(page_content="x")) == ""


def test_evidence_recall_ignores_case_and_whitespace():
    docs = [_doc("Cả gia tài chỉ có một  LƯỠI BÚA\ncủa cha để lại.", "thach_sanh.md")]
    assert evidence_recall(["lưỡi búa của cha"], docs) == 1.0
    assert evidence_recall(["lưỡi búa của cha", "chằn tinh"], docs) == 0.5
    assert evidence_recall([], docs) is None


def test_retrieval_benchmark_skips_unanswerable_and_scores_cases():
    corpus = {
        "q1": [_doc("chỉ có một lưỡi búa của cha để lại", "c/thach_sanh.md"), _doc("x", "c/tam_cam.md")],
        "q2": [_doc("y", "c/so_dua.md"), _doc("thả cá bống xuống giếng", "c/tam_cam.md")],
    }
    by_question = {case.question: corpus[case.id] for case in CASES if case.answerable}

    report = run_retrieval_benchmark(lambda q, k: by_question[q], CASES, k=2)
    summary = report["summary"]

    assert summary["cases"] == 2
    assert summary["recall_at_k"] == 1.0
    assert summary["mrr"] == (1.0 + 0.5) / 2
    assert summary["evidence_recall"] == 1.0
    assert summary["errors"] == 0
    assert report["by_tag"]["paraphrase"]["cases"] == 1
    assert report["cases"][1]["retrieved_sources"] == ["so_dua", "tam_cam"]


def test_retrieval_benchmark_records_errors():
    def broken(_q, _k):
        raise RuntimeError("index missing")

    report = run_retrieval_benchmark(broken, CASES, k=3)
    assert report["summary"]["errors"] == 2
    assert report["summary"]["recall_at_k"] == 0.0


def test_abstention_and_token_recall():
    assert is_abstention("Xin lỗi, tôi không có đủ thông tin để trả lời.")
    assert is_abstention("I don't have enough information.")
    assert not is_abstention("Thạch Sanh chỉ có một lưỡi búa [S1].")
    assert token_recall("Con cá bống.", "Đó là con cá bống [S2].") == 1.0
    assert token_recall("Con cá bống.", "Một con cá.") == 2 / 3
    assert token_recall(None, "anything") is None


def test_percentile_interpolates():
    assert percentile([], 95) == 0.0
    assert percentile([10.0], 95) == 10.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 95) == 9.5


class FakeChatModel:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages, **_kwargs):
        self.prompts.append(messages)
        return SimpleNamespace(content="ok", usage_metadata={"input_tokens": 10, "output_tokens": 2})

    def stream(self, messages, **_kwargs):
        yield SimpleNamespace(content="o", usage_metadata=None)
        yield SimpleNamespace(content="k", usage_metadata={"input_tokens": 5, "output_tokens": 1})

    def with_structured_output(self, _schema):
        return self


def test_llm_call_counter_counts_invoke_stream_and_structured_calls():
    inner = FakeChatModel()
    manager = SimpleNamespace(llm=inner, _llm=inner)
    counter = LLMCallCounter()
    counter.attach(manager)

    manager._llm.invoke("a")
    list(manager._llm.stream("b"))
    manager._llm.with_structured_output(dict).invoke("c")

    assert counter.calls == 3
    assert counter.input_tokens == 25
    assert counter.output_tokens == 5
    assert manager._llm.prompts == ["a", "c"]  # other attributes pass through


def test_answer_benchmark_scores_answers_calls_and_abstention():
    counter = LLMCallCounter()
    answers = {
        "q1": ("Thạch Sanh chỉ có một lưỡi búa của cha để lại [S1].", 3),
        "q2": ("Tôi không có đủ thông tin.", 2),
        "q3": ("Không có thông tin về truyện này.", 2),
    }
    by_question = {case.question: case.id for case in CASES}

    def answer_fn(question):
        text, calls = answers[by_question[question]]
        counter.calls += calls
        return {"answer": text, "contexts": [_doc("lưỡi búa", "thach_sanh.md")]}

    judged = []

    def judge(question, answer, contexts):
        judged.append(question)
        return 0.8

    report = run_answer_benchmark(answer_fn, CASES, counter=counter, judge=judge)
    summary = report["summary"]

    assert summary["cases"] == 3
    assert summary["abstention_accuracy"] == 1.0
    assert summary["false_abstention_rate"] == 0.5
    assert summary["citation_rate"] == 0.5
    assert summary["llm_calls_per_query"] == 7 / 3
    assert summary["faithfulness"] == 0.8
    assert judged == [CASES[0].question]  # abstentions and unanswerables are not judged
    assert report["cases"][0]["answer_recall"] == 1.0


def test_compare_reports_marks_direction():
    baseline = {"configs": {"hybrid": {"summary": {"cases": 87, "recall_at_k": 0.8, "latency_p50_ms": 50.0}}}}
    current = {"configs": {
        "hybrid": {"summary": {"cases": 87, "recall_at_k": 0.9, "latency_p50_ms": 60.0}},
        "new": {"summary": {"recall_at_k": 1.0}},
    }}

    rows = {row["metric"]: row for row in compare_reports(baseline, current)}

    assert set(rows) == {"recall_at_k", "latency_p50_ms"}
    assert rows["recall_at_k"]["status"] == "better"
    assert rows["latency_p50_ms"]["status"] == "worse"

"""
Golden-Set Benchmark
====================

Measure a RAG pipeline against a golden set before and after every change.

A golden set is a JSONL file with one case per line:

    {"id": "q001", "question": "...", "answer": "..." | null,
     "sources": ["thach_sanh"], "evidence": ["verbatim phrase"], "tags": ["fact"]}

``sources`` are corpus file stems; an empty list marks a question the corpus
cannot answer. ``evidence`` phrases appear verbatim in those sources, so they
show whether the retrieved chunks hold the answer, not just the right story.

Retrieval runs need no LLM. Answer runs count every chat-model call so the
cost of a pipeline is reported next to its quality.

Usage:
    cases = load_golden_set("evals/fairy_tales/golden.jsonl")
    report = run_retrieval_benchmark(lambda q, k: rag.retrieve(q, k=k), cases, k=5)
    print(report["summary"])
"""

import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from src.evaluation.evaluator import retrieval_scores

RETRIEVAL_METRICS = ("precision_at_k", "recall_at_k", "mrr", "ndcg", "evidence_recall")

# Metrics where a smaller number is an improvement (used by compare_reports).
LOWER_IS_BETTER = {
    "latency_p50_ms",
    "latency_p95_ms",
    "llm_calls_per_query",
    "input_tokens_per_query",
    "output_tokens_per_query",
    "false_abstention_rate",
    "errors",
}

# Phrases a grounded pipeline uses when the context has no answer.
ABSTENTION_MARKERS = (
    "không có đủ thông tin",
    "không đủ thông tin",
    "không có thông tin",
    "không tìm thấy thông tin",
    "chưa có thông tin",
    "không được đề cập",
    "không đề cập",
    "không được nhắc",
    "don't have enough information",
    "do not have enough information",
    "not enough information",
)

FAITHFULNESS_PROMPT = """Bạn là giám khảo chấm độ trung thực của câu trả lời.
Chỉ dựa vào NGỮ CẢNH, cho biết câu trả lời có được ngữ cảnh hỗ trợ không.

Câu hỏi: {question}

Ngữ cảnh:
{context}

Câu trả lời:
{answer}

Chấm điểm từ 0.0 đến 1.0:
- 1.0: mọi ý trong câu trả lời đều có trong ngữ cảnh
- 0.5: chỉ một phần được ngữ cảnh hỗ trợ
- 0.0: phần lớn không có trong ngữ cảnh hoặc mâu thuẫn với ngữ cảnh

Chỉ trả về một con số."""


@dataclass
class GoldenCase:
    """One golden-set question."""
    id: str
    question: str
    answer: Optional[str]
    sources: List[str]
    evidence: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.sources)


def load_golden_set(path: Union[str, Path]) -> List[GoldenCase]:
    """Read a JSONL golden set."""
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        cases.append(GoldenCase(
            id=data["id"],
            question=data["question"],
            answer=data.get("answer"),
            sources=list(data.get("sources") or []),
            evidence=list(data.get("evidence") or []),
            tags=list(data.get("tags") or []),
        ))
    return cases


def normalize_text(text: str) -> str:
    """NFC, casefold and collapse whitespace so phrase matching ignores layout."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def source_key(item: Any) -> str:
    """Corpus file stem of a retrieved Document (or dict), e.g. ``thach_sanh``."""
    if isinstance(item, dict):
        metadata = item.get("metadata") or {}
        source = item.get("source") or metadata.get("source") or metadata.get("file_name")
    else:
        metadata = getattr(item, "metadata", None) or {}
        source = metadata.get("source") or metadata.get("file_name")
    # PureWindowsPath splits on both "/" and "\\".
    return PureWindowsPath(str(source)).stem if source else ""


def document_text(item: Any) -> str:
    """Text of a retrieved Document, dict or string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("page_content") or item.get("content") or "")
    return str(getattr(item, "page_content", "") or "")


def evidence_recall(evidence: Sequence[str], docs: Sequence[Any]) -> Optional[float]:
    """Share of evidence phrases found verbatim inside a single retrieved chunk."""
    if not evidence:
        return None
    texts = [normalize_text(document_text(doc)) for doc in docs]
    found = sum(1 for phrase in evidence if any(normalize_text(phrase) in text for text in texts))
    return found / len(evidence)


def is_abstention(answer: str) -> bool:
    """True when the answer says the context does not contain the information."""
    normalized = normalize_text(answer).replace("’", "'")
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def token_recall(reference: Optional[str], answer: str) -> Optional[float]:
    """Share of the reference answer's words that appear in the generated answer.

    A cheap, LLM-free proxy for answer correctness that tolerates verbose answers.
    """
    if not reference:
        return None
    ref_tokens = set(re.findall(r"\w+", normalize_text(reference)))
    if not ref_tokens:
        return None
    answer_tokens = set(re.findall(r"\w+", normalize_text(answer)))
    return len(ref_tokens & answer_tokens) / len(ref_tokens)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0-100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _by_tag(rows: List[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    tags = sorted({tag for row in rows for tag in row["tags"]})
    breakdown = {}
    for tag in tags:
        tagged = [row for row in rows if tag in row["tags"]]
        breakdown[tag] = {"cases": len(tagged)}
        for metric in metrics:
            breakdown[tag][metric] = _mean([row.get(metric) for row in tagged])
    return breakdown


def run_retrieval_benchmark(
    retrieve: Callable[[str, int], Sequence[Any]],
    cases: Sequence[GoldenCase],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Score a retrieve function on the answerable golden cases.

    Args:
        retrieve: ``retrieve(question, k)`` returning Documents (or dicts)
            whose ``metadata["source"]`` points at a corpus file
        cases: Golden cases; unanswerable ones are skipped
        k: Cut-off for every metric

    Returns:
        Dict with ``summary``, ``by_tag`` and per-case ``cases``
    """
    rows = []
    for case in cases:
        if not case.answerable:
            continue

        error = None
        start = time.perf_counter()
        try:
            docs = list(retrieve(case.question, k))[:k]
        except Exception as e:
            docs, error = [], str(e)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved = [source_key(doc) for doc in docs]
        rows.append({
            "id": case.id,
            "question": case.question,
            "tags": case.tags,
            "expected_sources": case.sources,
            "retrieved_sources": retrieved,
            **retrieval_scores(retrieved, case.sources, k),
            "evidence_recall": evidence_recall(case.evidence, docs),
            "latency_ms": round(latency_ms, 2),
            "error": error,
        })

    latencies = [row["latency_ms"] for row in rows]
    summary: Dict[str, Any] = {"cases": len(rows), "k": k}
    for metric in RETRIEVAL_METRICS:
        summary[metric] = _mean([row[metric] for row in rows])
    summary["latency_p50_ms"] = percentile(latencies, 50)
    summary["latency_p95_ms"] = percentile(latencies, 95)
    summary["errors"] = sum(1 for row in rows if row["error"])

    return {
        "summary": summary,
        "by_tag": _by_tag(rows, RETRIEVAL_METRICS),
        "cases": rows,
    }


class LLMCallCounter:
    """
    Count chat-model calls and token usage made through LLMManagers.

    ``attach`` wraps the manager's underlying chat model, so calls made by
    components sharing that manager (graders, web search) are counted too.
    """

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def attach(self, llm_manager) -> None:
        """Start counting calls made through ``llm_manager``."""
        llm_manager._llm = _CountingChatModel(llm_manager.llm, self)

    def snapshot(self) -> Dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    def record_usage(self, message: Any) -> None:
        usage = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)


class _CountingChatModel:
    """Chat-model proxy that reports every invoke/stream to an LLMCallCounter."""

    def __init__(self, inner: Any, counter: LLMCallCounter):
        self._inner = inner
        self._counter = counter

    def invoke(self, *args, **kwargs):
        self._counter.calls += 1
        response = self._inner.invoke(*args, **kwargs)
        self._counter.record_usage(response)
        return response

    async def ainvoke(self, *args, **kwargs):
        self._counter.calls += 1
        response = await self._inner.ainvoke(*args, **kwargs)
        self._counter.record_usage(response)
        return response

    def stream(self, *args, **kwargs):
        self._counter.calls += 1
        for chunk in self._inner.stream(*args, **kwargs):
            self._counter.record_usage(chunk)
            yield chunk

    async def astream(self, *args, **kwargs):
        self._counter.calls += 1
        async for chunk in self._inner.astream(*args, **kwargs):
            self._counter.record_usage(chunk)
            yield chunk

    def with_structured_output(self, *args, **kwargs):
        return _CountingChatModel(self._inner.with_structured_output(*args, **kwargs), self._counter)

    def bind_tools(self, *args, **kwargs):
        return _CountingChatModel(self._inner.bind_tools(*args, **kwargs), self._counter)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def make_faithfulness_judge(llm_manager) -> Callable[[str, str, List[str]], Optional[float]]:
    """Build an LLM judge returning 0-1 faithfulness, or None when unparseable."""

    def judge(question: str, answer: str, contexts: List[str]) -> Optional[float]:
        prompt = FAITHFULNESS_PROMPT.format(
            question=question,
            context="\n\n".join(contexts),
            answer=answer,
        )
        match = re.search(r"\d+(?:[.,]\d+)?", llm_manager.generate(prompt))
        if not match:
            return None
        score = float(match.group().replace(",", "."))
        return score if 0.0 <= score <= 1.0 else None

    return judge


def run_answer_benchmark(
    answer_fn: Callable[[str], Dict[str, Any]],
    cases: Sequence[GoldenCase],
    counter: Optional[LLMCallCounter] = None,
    judge: Optional[Callable[[str, str, List[str]], Optional[float]]] = None,
) -> Dict[str, Any]:
    """
    Score end-to-end answers on every golden case.

    Args:
        answer_fn: ``answer_fn(question)`` returning ``{"answer": str,
            "contexts": [Document | str, ...]}``
        cases: Golden cases, answerable and unanswerable
        counter: LLMCallCounter attached to the pipeline's LLM
        judge: Optional faithfulness judge (its calls are not counted)

    Returns:
        Dict with ``summary``, ``by_tag`` and per-case ``cases``
    """
    counter = counter or LLMCallCounter()
    rows = []
    for case in cases:
        before = counter.snapshot()
        error = None
        start = time.perf_counter()
        try:
            result = answer_fn(case.question)
        except Exception as e:
            result, error = {}, str(e)
        latency_ms = (time.perf_counter() - start) * 1000
        after = counter.snapshot()

        answer = str(result.get("answer") or "")
        contexts = [document_text(item) for item in result.get("contexts") or []]
        abstained = is_abstention(answer)
        row = {
            "id": case.id,
            "question": case.question,
            "tags": case.tags,
            "answerable": case.answerable,
            "answer": answer,
            "abstained": abstained,
            "llm_calls": after["calls"] - before["calls"],
            "input_tokens": after["input_tokens"] - before["input_tokens"],
            "output_tokens": after["output_tokens"] - before["output_tokens"],
            "latency_ms": round(latency_ms, 2),
            "error": error,
        }
        if case.answerable:
            row["answer_recall"] = token_recall(case.answer, answer)
            row["cited"] = bool(re.search(r"\[S\d+\]", answer))
            row["faithfulness"] = (
                judge(case.question, answer, contexts)
                if judge and answer and contexts and not abstained and not error
                else None
            )
        rows.append(row)

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    latencies = [row["latency_ms"] for row in rows]
    summary = {
        "cases": len(rows),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "answer_recall": _mean([row["answer_recall"] for row in answerable]),
        "faithfulness": _mean([row["faithfulness"] for row in answerable]),
        "citation_rate": _mean([float(row["cited"]) for row in answerable]),
        "false_abstention_rate": _mean([float(row["abstained"]) for row in answerable]),
        "abstention_accuracy": _mean([float(row["abstained"]) for row in unanswerable]),
        "llm_calls_per_query": _mean([row["llm_calls"] for row in rows]),
        "input_tokens_per_query": _mean([row["input_tokens"] for row in rows]),
        "output_tokens_per_query": _mean([row["output_tokens"] for row in rows]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "errors": sum(1 for row in rows if row["error"]),
    }

    return {
        "summary": summary,
        "by_tag": _by_tag(answerable, ("answer_recall", "faithfulness", "false_abstention_rate")),
        "cases": rows,
    }


def compare_reports(baseline: Dict[str, Any], current: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Summary-metric deltas for every config present in both reports.

    Each row has ``config``, ``metric``, ``baseline``, ``current``, ``delta``
    and ``status`` ("better", "worse" or "same").
    """
    rows = []
    base_configs = baseline.get("configs", {})
    for name, run in current.get("configs", {}).items():
        if name not in base_configs:
            continue
        base_summary = base_configs[name]["summary"]
        for metric, value in run["summary"].items():
            base_value = base_summary.get(metric)
            if not isinstance(value, (int, float)) or not isinstance(base_value, (int, float)):
                continue
            if metric in ("cases", "k", "answerable_cases", "unanswerable_cases"):
                continue
            delta = value - base_value
            if abs(delta) < 1e-9:
                status = "same"
            elif (delta < 0) == (metric in LOWER_IS_BETTER):
                status = "better"
            else:
                status = "worse"
            rows.append({
                "config": name,
                "metric": metric,
                "baseline": base_value,
                "current": value,
                "delta": delta,
                "status": status,
            })
    return rows

"""
RAG Evaluator
=============

End-to-end evaluation framework for RAG systems.

Features:
- Dataset-based evaluation
- Comparative evaluation
- Report generation

Usage:
    evaluator = RAGEvaluator(llm)
    report = evaluator.evaluate(rag_system, test_data)
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from src.evaluation.metrics import RAGMetrics, RAGScores


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    question: str
    answer: str
    contexts: List[str]
    scores: RAGScores
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    results: List[EvaluationResult]
    average_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvaluationResult:
    """Result of a single retrieval evaluation."""
    question: str
    expected_sources: List[str]
    retrieved_sources: List[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvaluationReport:
    """Complete retrieval evaluation report."""
    results: List[RetrievalEvaluationResult]
    average_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGEvaluator:
    """
    End-to-end RAG evaluator.

    Provides:
    - Dataset-based evaluation
    - Comparative evaluation of multiple systems
    - Report generation and export

    Example:
        evaluator = RAGEvaluator(llm)

        # Evaluate a RAG system
        report = evaluator.evaluate(
            rag_system=query_func,
            test_data=[
                {"question": "What is Python?", "expected": "Python is..."},
                ...
            ]
        )

        # Compare multiple systems
        comparison = evaluator.compare(
            systems={"naive": naive_rag, "advanced": advanced_rag},
            test_data=test_data
        )

        # Export report
        evaluator.export_report(report, "evaluation_report.json")
    """

    def __init__(self, llm=None):
        """
        Initialize evaluator.

        Args:
            llm: LLM instance for evaluation
        """
        self.llm = llm
        self.metrics = RAGMetrics(llm) if llm is not None else None

    def evaluate(
        self,
        query_func: Callable[[str], Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationReport:
        """
        Evaluate a RAG system.

        Args:
            query_func: Function that takes question and returns dict with 'answer' and 'contexts'
            test_data: List of test cases with 'question' and optional 'expected'
            metadata: Additional metadata for the report

        Returns:
            EvaluationReport with results
        """
        results = []
        if self.metrics is None:
            raise ValueError("LLM-backed metrics require an llm. Use evaluate_retrieval() for retrieval-only evals.")

        for test_case in test_data:
            question = test_case["question"]
            expected = test_case.get("expected")

            # Query the system
            try:
                response = query_func(question)
                answer = response.get("answer", "")
                contexts = response.get("contexts", [])
            except Exception as e:
                answer = f"Error: {str(e)}"
                contexts = []

            # Evaluate
            scores = self.metrics.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=expected
            )

            # Create result
            result = EvaluationResult(
                question=question,
                answer=answer,
                contexts=contexts,
                scores=scores,
                metadata={"expected": expected}
            )

            results.append(result)

        # Calculate averages
        avg_scores = self._calculate_averages(results)

        return EvaluationReport(
            results=results,
            average_scores=avg_scores,
            metadata=metadata or {}
        )

    def compare(
        self,
        systems: Dict[str, Callable[[str], Dict[str, Any]]],
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, EvaluationReport]:
        """
        Compare multiple RAG systems.

        Args:
            systems: Dict mapping system names to query functions
            test_data: Test data

        Returns:
            Dict mapping system names to evaluation reports
        """
        reports = {}

        for name, query_func in systems.items():
            report = self.evaluate(
                query_func=query_func,
                test_data=test_data,
                metadata={"system_name": name}
            )
            reports[name] = report

        return reports

    def evaluate_retrieval(
        self,
        retrieve_func: Callable[[str, int], Any],
        test_data: List[Dict[str, Any]],
        k: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalEvaluationReport:
        """
        Evaluate retriever quality against golden expected sources.

        Each test case should contain:
        - question: user query
        - expected_sources: list of source identifiers expected in top-k

        The retrieve function may return LangChain Documents or dictionaries
        containing source/source_id/metadata fields.
        """
        results = []

        for test_case in test_data:
            question = test_case["question"]
            expected_sources = self._normalize_expected_sources(test_case)

            try:
                retrieved = retrieve_func(question, k)
            except TypeError:
                retrieved = retrieve_func(question)
            except Exception as e:
                retrieved = []
                test_case = {**test_case, "error": str(e)}

            retrieved_sources = [
                self._source_identifier(item)
                for item in list(retrieved)[:k]
            ]

            scores = self._retrieval_scores(retrieved_sources, expected_sources, k)
            results.append(RetrievalEvaluationResult(
                question=question,
                expected_sources=expected_sources,
                retrieved_sources=retrieved_sources,
                precision_at_k=scores["precision_at_k"],
                recall_at_k=scores["recall_at_k"],
                mrr=scores["mrr"],
                ndcg=scores["ndcg"],
                metadata={
                    key: value
                    for key, value in test_case.items()
                    if key not in ("question", "expected_sources", "expected_source")
                },
            ))

        return RetrievalEvaluationReport(
            results=results,
            average_scores=self._calculate_retrieval_averages(results),
            metadata=metadata or {},
        )

    def export_retrieval_report(
        self,
        report: RetrievalEvaluationReport,
        output_path: str,
    ) -> None:
        """Export a retrieval evaluation report to JSON."""
        report_dict = {
            "average_scores": report.average_scores,
            "metadata": report.metadata,
            "results": [
                {
                    "question": r.question,
                    "expected_sources": r.expected_sources,
                    "retrieved_sources": r.retrieved_sources,
                    "scores": {
                        "precision_at_k": r.precision_at_k,
                        "recall_at_k": r.recall_at_k,
                        "mrr": r.mrr,
                        "ndcg": r.ndcg,
                    },
                    "metadata": r.metadata,
                }
                for r in report.results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    def _calculate_averages(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """Calculate average scores."""
        if not results:
            return {}

        n = len(results)

        return {
            "faithfulness": sum(r.scores.faithfulness for r in results) / n,
            "answer_relevance": sum(r.scores.answer_relevance for r in results) / n,
            "context_precision": sum(r.scores.context_precision for r in results) / n,
            "context_recall": sum(r.scores.context_recall for r in results) / n,
            "overall": sum(r.scores.overall for r in results) / n,
        }

    def _calculate_retrieval_averages(
        self,
        results: List[RetrievalEvaluationResult],
    ) -> Dict[str, float]:
        """Calculate average retrieval scores."""
        if not results:
            return {}

        n = len(results)
        return {
            "precision_at_k": sum(r.precision_at_k for r in results) / n,
            "recall_at_k": sum(r.recall_at_k for r in results) / n,
            "mrr": sum(r.mrr for r in results) / n,
            "ndcg": sum(r.ndcg for r in results) / n,
        }

    def _normalize_expected_sources(self, test_case: Dict[str, Any]) -> List[str]:
        """Read expected source identifiers from a test case."""
        expected = test_case.get("expected_sources")
        if expected is None and test_case.get("expected_source"):
            expected = [test_case["expected_source"]]
        if expected is None:
            expected = []
        return [str(item) for item in expected]

    def _source_identifier(self, item: Any) -> str:
        """Extract a source identifier from a Document-like item."""
        if isinstance(item, dict):
            metadata = item.get("metadata") or {}
            return str(
                item.get("source")
                or item.get("source_id")
                or metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("url")
                or item.get("content", "")[:80]
            )

        metadata = getattr(item, "metadata", {}) or {}
        page_content = getattr(item, "page_content", "")
        return str(
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("url")
            or page_content[:80]
        )

    def _retrieval_scores(
        self,
        retrieved_sources: List[str],
        expected_sources: List[str],
        k: int,
    ) -> Dict[str, float]:
        """Compute binary relevance retrieval metrics."""
        expected = set(expected_sources)
        retrieved = retrieved_sources[:k]

        if not expected:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
            }

        hits = [1 if source in expected else 0 for source in retrieved]
        hit_count = sum(hits)

        precision = hit_count / max(len(retrieved), 1)
        recall = hit_count / len(expected)

        mrr = 0.0
        for rank, hit in enumerate(hits, 1):
            if hit:
                mrr = 1.0 / rank
                break

        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
        ideal_hits = min(len(expected), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg else 0.0

        return {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "mrr": mrr,
            "ndcg": ndcg,
        }

    def export_report(
        self,
        report: EvaluationReport,
        output_path: str
    ) -> None:
        """
        Export evaluation report to JSON.

        Args:
            report: Evaluation report
            output_path: Path to output file
        """
        report_dict = {
            "average_scores": report.average_scores,
            "metadata": report.metadata,
            "results": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "contexts": r.contexts,
                    "scores": {
                        "faithfulness": r.scores.faithfulness,
                        "answer_relevance": r.scores.answer_relevance,
                        "context_precision": r.scores.context_precision,
                        "context_recall": r.scores.context_recall,
                        "overall": r.scores.overall,
                    },
                    "metadata": r.metadata,
                }
                for r in report.results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    def print_report(self, report: EvaluationReport) -> None:
        """
        Print evaluation report to console.

        Args:
            report: Evaluation report
        """
        print("\n" + "=" * 60)
        print("RAG Evaluation Report")
        print("=" * 60)

        print("\nAverage Scores:")
        print("-" * 40)
        for metric, score in report.average_scores.items():
            print(f"  {metric:20s}: {score:.3f}")

        print("\nDetailed Results:")
        print("-" * 40)

        for i, result in enumerate(report.results, 1):
            print(f"\n[{i}] Question: {result.question}")
            print(f"    Answer: {result.answer[:100]}...")
            print(f"    Scores:")
            print(f"      Faithfulness:      {result.scores.faithfulness:.3f}")
            print(f"      Answer Relevance:  {result.scores.answer_relevance:.3f}")
            print(f"      Context Precision: {result.scores.context_precision:.3f}")
            print(f"      Context Recall:    {result.scores.context_recall:.3f}")
            print(f"      Overall:           {result.scores.overall:.3f}")

        print("\n" + "=" * 60)

    def generate_comparison_table(
        self,
        reports: Dict[str, EvaluationReport]
    ) -> str:
        """
        Generate comparison table for multiple systems.

        Args:
            reports: Dict mapping system names to reports

        Returns:
            Formatted comparison table
        """
        if not reports:
            return "No systems to compare."

        # Header
        metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall", "overall"]
        header = f"{'System':<20}" + "".join(f"{m:<20}" for m in metrics)
        separator = "-" * len(header)

        # Rows
        rows = []
        for name, report in reports.items():
            row = f"{name:<20}"
            for metric in metrics:
                score = report.average_scores.get(metric, 0.0)
                row += f"{score:<20.3f}"
            rows.append(row)

        # Build table
        table = f"\n{separator}\n{header}\n{separator}\n"
        table += "\n".join(rows)
        table += f"\n{separator}"

        return table

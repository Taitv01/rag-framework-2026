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

    def __init__(self, llm):
        """
        Initialize evaluator.

        Args:
            llm: LLM instance for evaluation
        """
        self.llm = llm
        self.metrics = RAGMetrics(llm)

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

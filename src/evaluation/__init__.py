"""
Evaluation
==========

RAG evaluation metrics and tools.
"""

from src.evaluation.metrics import RAGMetrics
from src.evaluation.evaluator import (
    RAGEvaluator,
    RetrievalEvaluationReport,
    RetrievalEvaluationResult,
    retrieval_scores,
)
from src.evaluation.benchmark import (
    GoldenCase,
    LLMCallCounter,
    compare_reports,
    load_golden_set,
    run_answer_benchmark,
    run_retrieval_benchmark,
)

__all__ = [
    "RAGMetrics",
    "RAGEvaluator",
    "RetrievalEvaluationReport",
    "RetrievalEvaluationResult",
    "retrieval_scores",
    "GoldenCase",
    "LLMCallCounter",
    "compare_reports",
    "load_golden_set",
    "run_answer_benchmark",
    "run_retrieval_benchmark",
]

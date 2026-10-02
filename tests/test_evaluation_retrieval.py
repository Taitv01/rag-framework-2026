"""
Tests for retrieval-only evaluation.
"""

from langchain_core.documents import Document


def test_evaluate_retrieval_metrics():
    """Test retrieval eval computes precision, recall, MRR, and nDCG."""
    from src.evaluation import RAGEvaluator

    docs = [
        Document(page_content="Relevant A", metadata={"source": "a.txt"}),
        Document(page_content="Irrelevant", metadata={"source": "x.txt"}),
        Document(page_content="Relevant B", metadata={"source": "b.txt"}),
    ]

    def retrieve(_question, _k):
        return docs

    evaluator = RAGEvaluator()
    report = evaluator.evaluate_retrieval(
        retrieve_func=retrieve,
        test_data=[
            {
                "question": "What is relevant?",
                "expected_sources": ["a.txt", "b.txt"],
            }
        ],
        k=3,
    )

    result = report.results[0]
    assert result.precision_at_k == 2 / 3
    assert result.recall_at_k == 1.0
    assert result.mrr == 1.0
    assert 0.0 < result.ndcg <= 1.0
    assert report.average_scores["recall_at_k"] == 1.0


def test_evaluate_retrieval_accepts_dict_results():
    """Test retrieval eval extracts sources from dict-shaped results."""
    from src.evaluation import RAGEvaluator

    def retrieve(_question, _k):
        return [
            {"source": "s1", "content": "first"},
            {"metadata": {"source": "s2"}, "content": "second"},
        ]

    evaluator = RAGEvaluator()
    report = evaluator.evaluate_retrieval(
        retrieve_func=retrieve,
        test_data=[{"question": "q", "expected_source": "s2"}],
        k=2,
    )

    result = report.results[0]
    assert result.retrieved_sources == ["s1", "s2"]
    assert result.mrr == 0.5


def test_retrieval_metrics_do_not_double_count_chunks_from_one_source():
    """Several chunks from the same source must keep recall and nDCG <= 1."""
    from src.evaluation import RAGEvaluator

    scores = RAGEvaluator()._retrieval_scores(["a", "a", "a"], ["a"], k=3)

    assert scores["precision_at_k"] == 1.0
    assert scores["recall_at_k"] == 1.0
    assert scores["ndcg"] == 1.0

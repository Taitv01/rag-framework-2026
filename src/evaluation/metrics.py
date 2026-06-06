"""
RAG Metrics
===========

Evaluation metrics for RAG systems.

Metrics:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevance: Does the answer address the question?
- Context Precision: Are the retrieved documents relevant?
- Context Recall: Are all relevant documents retrieved?

Usage:
    metrics = RAGMetrics(llm)
    scores = metrics.evaluate(question, answer, contexts)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RAGScores:
    """Scores for RAG evaluation."""
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class RAGMetrics:
    """
    RAG evaluation metrics.

    Evaluates:
    - Faithfulness: Answer grounded in context
    - Answer Relevance: Answer addresses question
    - Context Precision: Retrieved documents are relevant
    - Context Recall: All relevant documents retrieved

    Example:
        metrics = RAGMetrics(llm)

        # Evaluate single response
        scores = metrics.evaluate(
            question="What is Python?",
            answer="Python is a programming language...",
            contexts=["Python is a high-level language..."]
        )

        # Batch evaluation
        results = metrics.batch_evaluate(evaluation_data)
    """

    def __init__(self, llm):
        """
        Initialize RAG metrics.

        Args:
            llm: LLM instance for evaluation
        """
        self.llm = llm

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RAGScores:
        """
        Evaluate RAG response.

        Args:
            question: Original question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer

        Returns:
            RAGScores with all metrics
        """
        # Calculate individual metrics
        faithfulness = self._evaluate_faithfulness(answer, contexts)
        answer_relevance = self._evaluate_answer_relevance(question, answer)
        context_precision = self._evaluate_context_precision(question, contexts)
        context_recall = self._evaluate_context_recall(question, answer, contexts)

        # Calculate overall score
        scores = [faithfulness, answer_relevance, context_precision, context_recall]
        overall = sum(scores) / len(scores)

        return RAGScores(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            overall=overall,
            details={
                "question": question,
                "answer": answer[:200],
                "num_contexts": len(contexts),
            }
        )

    def _evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate faithfulness: Is the answer grounded in context?

        Returns score 0-1.
        """
        context_str = "\n".join(contexts)

        prompt = f"""Evaluate if the answer is faithful to the provided context.

Context:
{context_str}

Answer:
{answer}

Rate faithfulness on a scale of 0.0 to 1.0:
- 1.0: Answer is completely supported by context
- 0.5: Answer is partially supported
- 0.0: Answer is not supported by context

Return ONLY the numeric score, nothing else."""

        try:
            response = self.llm.generate(prompt).strip()
            return float(response)
        except (ValueError, TypeError):
            return 0.5

    def _evaluate_answer_relevance(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate answer relevance: Does the answer address the question?

        Returns score 0-1.
        """
        prompt = f"""Evaluate if the answer is relevant to the question.

Question: {question}

Answer: {answer}

Rate relevance on a scale of 0.0 to 1.0:
- 1.0: Answer directly and completely addresses the question
- 0.5: Answer partially addresses the question
- 0.0: Answer does not address the question

Return ONLY the numeric score, nothing else."""

        try:
            response = self.llm.generate(prompt).strip()
            return float(response)
        except (ValueError, TypeError):
            return 0.5

    def _evaluate_context_precision(
        self,
        question: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate context precision: Are retrieved contexts relevant?

        Returns score 0-1.
        """
        if not contexts:
            return 0.0

        relevance_scores = []

        for ctx in contexts:
            prompt = f"""Evaluate if this context is relevant to the question.

Question: {question}

Context: {ctx[:500]}

Rate relevance on a scale of 0.0 to 1.0:
- 1.0: Highly relevant
- 0.5: Somewhat relevant
- 0.0: Not relevant

Return ONLY the numeric score, nothing else."""

            try:
                response = self.llm.generate(prompt).strip()
                score = float(response)
                relevance_scores.append(score)
            except (ValueError, TypeError):
                relevance_scores.append(0.5)

        return sum(relevance_scores) / len(relevance_scores)

    def _evaluate_context_recall(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate context recall: Are all relevant documents retrieved?

        Returns score 0-1.
        """
        context_str = "\n".join(contexts)

        prompt = f"""Evaluate if the retrieved contexts contain all the information needed to answer the question.

Question: {question}

Answer given: {answer}

Retrieved contexts:
{context_str}

Rate recall on a scale of 0.0 to 1.0:
- 1.0: Contexts contain all information needed
- 0.5: Contexts contain some relevant information
- 0.0: Contexts don't contain relevant information

Return ONLY the numeric score, nothing else."""

        try:
            response = self.llm.generate(prompt).strip()
            return float(response)
        except (ValueError, TypeError):
            return 0.5

    def batch_evaluate(
        self,
        evaluation_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Batch evaluation of multiple RAG responses.

        Args:
            evaluation_data: List of dicts with 'question', 'answer', 'contexts'

        Returns:
            Dict with average scores and individual results
        """
        all_scores = []

        for data in evaluation_data:
            scores = self.evaluate(
                question=data["question"],
                answer=data["answer"],
                contexts=data["contexts"],
                ground_truth=data.get("ground_truth")
            )
            all_scores.append(scores)

        # Calculate averages
        avg_faithfulness = sum(s.faithfulness for s in all_scores) / len(all_scores)
        avg_relevance = sum(s.answer_relevance for s in all_scores) / len(all_scores)
        avg_precision = sum(s.context_precision for s in all_scores) / len(all_scores)
        avg_recall = sum(s.context_recall for s in all_scores) / len(all_scores)
        avg_overall = sum(s.overall for s in all_scores) / len(all_scores)

        return {
            "averages": {
                "faithfulness": avg_faithfulness,
                "answer_relevance": avg_relevance,
                "context_precision": avg_precision,
                "context_recall": avg_recall,
                "overall": avg_overall,
            },
            "individual": [
                {
                    "question": s.details["question"],
                    "scores": {
                        "faithfulness": s.faithfulness,
                        "answer_relevance": s.answer_relevance,
                        "context_precision": s.context_precision,
                        "context_recall": s.context_recall,
                        "overall": s.overall,
                    }
                }
                for s in all_scores
            ],
            "count": len(all_scores),
        }

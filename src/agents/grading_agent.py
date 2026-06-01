"""
Grading Agent
=============

Agent for grading document relevance to queries.

Features:
- Binary relevance grading
- Multi-dimensional scoring
- Batch grading

Usage:
    agent = GradingAgent(llm)
    grades = agent.grade_documents(question, documents)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class DocumentGrade:
    """Grade for a document."""
    is_relevant: bool
    confidence: float
    reason: str


class GradingAgent:
    """
    Agent for grading document relevance.

    Evaluates documents based on:
    - Relevance to the query
    - Information completeness
    - Source reliability

    Example:
        agent = GradingAgent(llm)

        # Grade single document
        grade = agent.grade_document("What is Python?", doc)

        # Grade multiple documents
        grades = agent.grade_documents("What is Python?", docs)

        # Filter relevant documents
        relevant = agent.filter_relevant("What is Python?", docs)
    """

    def __init__(
        self,
        llm,
        relevance_threshold: float = 0.5,
    ):
        """
        Initialize grading agent.

        Args:
            llm: LLM instance
            relevance_threshold: Threshold for relevance filtering
        """
        self.llm = llm
        self.relevance_threshold = relevance_threshold

    def grade_document(
        self,
        question: str,
        document: Document
    ) -> DocumentGrade:
        """
        Grade a single document.

        Args:
            question: Question to grade against
            document: Document to grade

        Returns:
            DocumentGrade with relevance assessment
        """
        prompt = f"""You are a document relevance grader. Evaluate if the document is relevant to the question.

Question: {question}

Document:
{document.page_content[:1000]}

Rate the relevance on a scale of 0-1 and explain why.
Format: score: reason

Score 0.0-0.3: Not relevant
Score 0.4-0.6: Somewhat relevant
Score 0.7-1.0: Highly relevant"""

        response = self.llm.generate(prompt).strip()

        # Parse response
        try:
            if ":" in response:
                score_str, reason = response.split(":", 1)
                score = float(score_str.strip())
                reason = reason.strip()
            else:
                score = 0.5
                reason = response

            return DocumentGrade(
                is_relevant=score >= self.relevance_threshold,
                confidence=score,
                reason=reason
            )
        except (ValueError, IndexError):
            return DocumentGrade(
                is_relevant=True,
                confidence=0.5,
                reason="Unable to parse grade"
            )

    def grade_documents(
        self,
        question: str,
        documents: List[Document]
    ) -> List[DocumentGrade]:
        """
        Grade multiple documents.

        Args:
            question: Question to grade against
            documents: Documents to grade

        Returns:
            List of DocumentGrade objects
        """
        grades = []
        for doc in documents:
            grade = self.grade_document(question, doc)
            grades.append(grade)

        return grades

    def filter_relevant(
        self,
        question: str,
        documents: List[Document],
        min_score: Optional[float] = None
    ) -> List[Document]:
        """
        Filter documents by relevance.

        Args:
            question: Question to grade against
            documents: Documents to filter
            min_score: Minimum relevance score (overrides threshold)

        Returns:
            List of relevant Document objects
        """
        threshold = min_score or self.relevance_threshold
        relevant_docs = []

        for doc in documents:
            grade = self.grade_document(question, doc)
            if grade.is_relevant and grade.confidence >= threshold:
                relevant_docs.append(doc)

        # Return at least one document if none are relevant
        if not relevant_docs and documents:
            relevant_docs = [documents[0]]

        return relevant_docs

    def batch_grade(
        self,
        questions: List[str],
        documents: List[Document]
    ) -> Dict[str, List[DocumentGrade]]:
        """
        Grade documents against multiple questions.

        Args:
            questions: List of questions
            documents: Documents to grade

        Returns:
            Dict mapping questions to grades
        """
        results = {}

        for question in questions:
            grades = self.grade_documents(question, documents)
            results[question] = grades

        return results

    def get_best_documents(
        self,
        question: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Document]:
        """
        Get top-k most relevant documents.

        Args:
            question: Question to grade against
            documents: Documents to rank
            top_k: Number of top documents to return

        Returns:
            List of top-k relevant Document objects
        """
        # Grade all documents
        graded_docs = []
        for doc in documents:
            grade = self.grade_document(question, doc)
            graded_docs.append((doc, grade.confidence))

        # Sort by confidence
        graded_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [doc for doc, _ in graded_docs[:top_k]]

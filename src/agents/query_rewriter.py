"""
Query Rewriter
==============

Agent for optimizing search queries.

Features:
- Query expansion
- Query simplification
- Multi-query generation
- Context-aware rewriting

Usage:
    rewriter = QueryRewriter(llm)
    better_query = rewriter.rewrite("What is Python?")
    queries = rewriter.generate_multiple("What is Python?")
"""

from typing import List, Optional, Dict, Any


class QueryRewriter:
    """
    Query rewriting agent for better retrieval.

    Transforms queries to improve:
    - Recall (finding relevant documents)
    - Precision (reducing irrelevant results)
    - Diversity (covering different aspects)

    Example:
        rewriter = QueryRewriter(llm)

        # Simple rewrite
        better = rewriter.rewrite("What is Python?")

        # Generate multiple queries
        queries = rewriter.generate_multiple("What is Python?", num_queries=3)

        # Context-aware rewrite
        better = rewriter.rewrite_with_context(
            "Tell me more",
            conversation_history=[...]
        )
    """

    def __init__(
        self,
        llm,
        strategy: str = "balanced",
    ):
        """
        Initialize query rewriter.

        Args:
            llm: LLM instance
            strategy: Rewriting strategy ('expand', 'simplify', 'balanced')
        """
        self.llm = llm
        self.strategy = strategy

    def rewrite(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """
        Rewrite a single query.

        Args:
            query: Original query
            context: Optional context for rewriting

        Returns:
            Rewritten query
        """
        if self.strategy == "expand":
            return self._expand_query(query, context)
        elif self.strategy == "simplify":
            return self._simplify_query(query, context)
        else:
            return self._balanced_rewrite(query, context)

    def generate_multiple(
        self,
        query: str,
        num_queries: int = 3,
        context: Optional[str] = None
    ) -> List[str]:
        """
        Generate multiple query variations.

        Args:
            query: Original query
            num_queries: Number of variations to generate
            context: Optional context

        Returns:
            List of query variations
        """
        context_str = f"\nContext: {context}" if context else ""

        prompt = f"""Generate {num_queries} different search queries that would help find information to answer this question.

Original query: {query}{context_str}

Requirements:
1. Each query should approach the topic from a different angle
2. Use different keywords and phrasings
3. Some should be more specific, some more general

Return one query per line, nothing else."""

        response = self.llm.generate(prompt)
        queries = [q.strip() for q in response.strip().split("\n") if q.strip()]

        # Include original query
        queries.insert(0, query)

        return queries[:num_queries + 1]

    def rewrite_with_context(
        self,
        query: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Rewrite query with conversation context.

        Args:
            query: Original query
            conversation_history: Conversation history

        Returns:
            Context-aware rewritten query
        """
        # Build conversation context
        conv_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history[-3:]
        ])

        prompt = f"""You are a search query optimizer. Rewrite the query to be self-contained and specific.

The query is part of a conversation. Use the conversation context to understand what the user is really asking.

Recent conversation:
{conv_context}

Current query: {query}

Rewrite this query to be:
1. Self-contained (doesn't require conversation context to understand)
2. Specific enough to find relevant documents
3. Includes relevant keywords

Return ONLY the rewritten query, nothing else."""

        return self.llm.generate(prompt).strip()

    def _expand_query(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """Expand query with related terms."""
        context_str = f"\nContext: {context}" if context else ""

        prompt = f"""Expand this search query with related terms and synonyms to improve recall.

Original query: {query}{context_str}

Add relevant synonyms, related terms, or broader/narrower concepts.
Return ONLY the expanded query, nothing else."""

        return self.llm.generate(prompt).strip()

    def _simplify_query(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """Simplify query to key terms."""
        context_str = f"\nContext: {context}" if context else ""

        prompt = f"""Simplify this search query to its core keywords.

Original query: {query}{context_str}

Extract the most important 2-4 keywords that capture the essence of the query.
Return ONLY the simplified query, nothing else."""

        return self.llm.generate(prompt).strip()

    def _balanced_rewrite(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """Balanced rewrite optimizing for both recall and precision."""
        context_str = f"\nContext: {context}" if context else ""

        prompt = f"""You are a search query optimizer. Rewrite this query to balance recall and precision.

Original query: {query}{context_str}

Requirements:
1. Keep the core meaning
2. Add important context or specificity
3. Remove ambiguity
4. Use natural language that matches how documents might be written

Return ONLY the rewritten query, nothing else."""

        return self.llm.generate(prompt).strip()

    def decompose_query(
        self,
        query: str
    ) -> List[str]:
        """
        Decompose complex query into sub-queries.

        Args:
            query: Complex query

        Returns:
            List of simpler sub-queries
        """
        prompt = f"""Decompose this complex question into simpler sub-questions that together would answer the original question.

Complex question: {query}

Break it down into 2-4 simpler questions. Return one question per line, nothing else."""

        response = self.llm.generate(prompt)
        sub_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]

        return sub_queries

    def hyde_query(
        self,
        query: str
    ) -> str:
        """
        Generate hypothetical document (HyDE) for better retrieval.

        Creates a hypothetical answer that might exist in the corpus,
        then uses it for retrieval.

        Args:
            query: Original query

        Returns:
            Hypothetical document text
        """
        prompt = f"""Write a short, factual paragraph that would answer this question.
This will be used to find similar documents in a knowledge base.

Question: {query}

Write a concise, informative paragraph (2-3 sentences) that directly answers this question.
Write as if you are excerpting from a textbook or documentation."""

        return self.llm.generate(prompt).strip()

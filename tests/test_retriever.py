"""
Tests for Retriever
==================
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.core.retriever import RetrieverManager
from langchain_core.documents import Document


class TestRetrieverManager:
    """Test RetrieverManager class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_vector_store = Mock()
        self.mock_embeddings = Mock()

        # Sample documents
        self.sample_docs = [
            Document(
                page_content="Python is a programming language.",
                metadata={"source": "doc1.txt"}
            ),
            Document(
                page_content="JavaScript is used for web development.",
                metadata={"source": "doc2.txt"}
            ),
            Document(
                page_content="Machine learning is a subset of AI.",
                metadata={"source": "doc3.txt"}
            ),
        ]

    def test_basic_search(self):
        """Test basic similarity search."""
        self.mock_vector_store.similarity_search.return_value = self.sample_docs[:2]

        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
            k=2
        )

        results = retriever.search("What is Python?")

        assert len(results) == 2
        self.mock_vector_store.similarity_search.assert_called_once()

    def test_search_with_scores(self):
        """Test search with relevance scores."""
        self.mock_vector_store.similarity_search_with_score.return_value = [
            (self.sample_docs[0], 0.95),
            (self.sample_docs[1], 0.85),
        ]

        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        results = retriever.search_with_scores("What is Python?")

        assert len(results) == 2
        assert results[0][1] == 0.95

    def test_hybrid_search_without_bm25(self):
        """Test hybrid search falls back when BM25 not available."""
        self.mock_vector_store.similarity_search.return_value = self.sample_docs

        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
            use_hybrid=True,
            documents=[]  # No documents for BM25
        )

        results = retriever.search("What is Python?")

        # Should fall back to vector search
        assert len(results) == 3

    def test_hybrid_search_with_bm25(self):
        """Test hybrid search with BM25."""
        try:
            import rank_bm25
        except ImportError:
            pytest.skip("rank-bm25 not installed")

        self.mock_vector_store.similarity_search_with_score.return_value = [
            (doc, 0.9) for doc in self.sample_docs
        ]

        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
            documents=self.sample_docs,
            use_hybrid=True,
        )

        results = retriever.hybrid_search("Python programming")

        assert len(results) > 0

    def test_multi_query_search(self):
        """Test multi-query search."""
        self.mock_vector_store.similarity_search.return_value = self.sample_docs[:2]

        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        results = retriever.multi_query_search("What is Python?", num_queries=2)

        # Should have called similarity_search multiple times
        assert self.mock_vector_store.similarity_search.call_count > 1

    def test_get_retriever(self):
        """Test getting retriever interface."""
        mock_retriever = Mock()
        self.mock_vector_store.get_retriever.return_value = mock_retriever

        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        result = retriever.get_retriever()

        assert result == mock_retriever

    def test_normalize_scores(self):
        """Test score normalization."""
        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        scores = [0.2, 0.4, 0.6, 0.8, 1.0]
        normalized = retriever._normalize_scores(scores)

        assert normalized[0] == 0.0
        assert normalized[-1] == 1.0
        assert all(0 <= s <= 1 for s in normalized)

    def test_normalize_scores_empty(self):
        """Test score normalization with empty list."""
        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        normalized = retriever._normalize_scores([])
        assert normalized == []

    def test_normalize_scores_same_values(self):
        """Test score normalization with same values."""
        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        scores = [0.5, 0.5, 0.5]
        normalized = retriever._normalize_scores(scores)

        assert all(s == 1.0 for s in normalized)

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

        # Sample documents (English)
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

        # Sample documents (Vietnamese)
        self.vietnamese_docs = [
            Document(
                page_content="Thạch Sanh là một nhân vật trong truyện cổ tích Việt Nam.",
                metadata={"source": "truyen_co_tich.txt"}
            ),
            Document(
                page_content="Lý Thông là người lừa đảo, đã cướp công của Thạch Sanh.",
                metadata={"source": "truyen_co_tich.txt"}
            ),
            Document(
                page_content="Đại bàng là con quái vật hung dữ bị Thạch Sanh giết chết.",
                metadata={"source": "truyen_co_tich.txt"}
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

    def test_search_stacks_hybrid_and_reranking(self):
        """Test search applies reranking after hybrid retrieval when both are enabled."""
        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )
        retriever.config.use_hybrid = True
        retriever.config.use_reranking = True
        retriever.hybrid_search = Mock(return_value=self.sample_docs)
        retriever._reranker = Mock()
        retriever._reranker.predict.return_value = [0.1, 0.9, 0.2]

        results = retriever.search("programming", k=2)

        retriever.hybrid_search.assert_called_once()
        assert results == [self.sample_docs[1], self.sample_docs[2]]

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

    def test_normalize_vector_scores_for_distance_backend(self):
        """Test lower vector-store distances become higher relevance scores."""
        self.mock_vector_store.config.provider = "faiss"
        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        normalized = retriever._normalize_vector_scores([
            (self.sample_docs[0], 0.1),
            (self.sample_docs[1], 0.9),
        ])

        assert normalized == [1.0, 0.0]


class TestVietnameseRetriever:
    """Test Vietnamese-specific retriever functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_vector_store = Mock()
        self.mock_embeddings = Mock()

    def test_tokenize_for_bm25_vietnamese(self):
        """Test Vietnamese tokenization for BM25."""
        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        # Test Vietnamese text tokenization
        tokens = retriever._tokenize_for_bm25("Thạch Sanh đánh đại bàng")
        assert len(tokens) > 0
        # Should handle Vietnamese diacritics
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenize_for_bm25_english(self):
        """Test English tokenization still works."""
        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        tokens = retriever._tokenize_for_bm25("Python is a programming language")
        assert len(tokens) == 5
        assert "python" in tokens

    def test_generate_query_variations_vietnamese(self):
        """Test Vietnamese query variations."""
        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        variations = retriever._generate_query_variations(
            "Thạch Sanh là ai?", num_variations=3
        )

        assert len(variations) > 1
        # Should contain the original query
        assert "Thạch Sanh là ai?" in variations

    def test_generate_query_variations_english(self):
        """Test English query variations still work."""
        retriever = RetrieverManager(
            vector_store=self.mock_vector_store,
            embeddings=self.mock_embeddings,
        )

        variations = retriever._generate_query_variations(
            "What is Python?", num_variations=3
        )

        assert len(variations) > 1
        assert "What is Python?" in variations

    def test_default_reranker_model(self):
        """Test that default reranker model is Vietnamese-aware."""
        from src.core.retriever import RetrieverConfig

        config = RetrieverConfig()
        assert "Vietnamese" in config.reranker_model or "vietnamese" in config.reranker_model.lower()


class TestVietnameseProcessor:
    """Test Vietnamese text processor."""

    def test_processor_creation(self):
        """Test VietnameseProcessor can be created."""
        from src.core.vietnamese_processor import VietnameseProcessor

        processor = VietnameseProcessor()
        assert processor is not None

    def test_detect_language_vietnamese(self):
        """Test Vietnamese language detection."""
        from src.core.vietnamese_processor import VietnameseProcessor

        processor = VietnameseProcessor()
        lang = processor.detect_language("Thạch Sanh là ai?")
        assert lang == "vi"

    def test_detect_language_english(self):
        """Test English language detection."""
        from src.core.vietnamese_processor import VietnameseProcessor

        processor = VietnameseProcessor()
        lang = processor.detect_language("What is Python?")
        assert lang == "en"

    def test_split_sentences_vietnamese(self):
        """Test Vietnamese sentence splitting."""
        from src.core.vietnamese_processor import VietnameseProcessor

        processor = VietnameseProcessor()
        sentences = processor.split_sentences(
            "Thạch Sanh đánh đại bàng. Lý Thông lừa đảo. Công chúa được cứu."
        )
        assert len(sentences) == 3

    def test_split_sentences_abbreviation(self):
        """Test Vietnamese abbreviation handling in sentence splitting."""
        from src.core.vietnamese_processor import VietnameseProcessor

        processor = VietnameseProcessor()
        sentences = processor.split_sentences(
            "Hà Nội là thủ đô. TP. Hồ Chí Minh là thành phố lớn."
        )
        # "TP." should not be treated as sentence boundary
        assert len(sentences) == 2

    def test_tokenize_vietnamese(self):
        """Test Vietnamese word segmentation."""
        from src.core.vietnamese_processor import VietnameseProcessor

        processor = VietnameseProcessor()
        tokens = processor.tokenize("Thạch Sanh đánh đại bàng")
        assert len(tokens) > 0
        # Should handle Vietnamese compound words
        assert all(isinstance(t, str) for t in tokens)

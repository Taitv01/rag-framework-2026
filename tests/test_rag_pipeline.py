"""
Tests for RAG Pipeline
=====================
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from langchain_core.documents import Document


class TestNaiveRAG:
    """Test NaiveRAG class."""

    def test_add_texts(self):
        """Test adding texts."""
        from src.rag.naive_rag import NaiveRAG

        # Create RAG with mocked components
        rag = NaiveRAG.__new__(NaiveRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 4
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test document 2", metadata={"source": "test2.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs

        # Test add_texts
        texts = ["Text 1", "Text 2"]
        num_chunks = rag.add_texts(texts)

        assert num_chunks == len(sample_docs)
        rag.text_splitter.split_documents.assert_called_once()

    def test_query(self):
        """Test querying."""
        from src.rag.naive_rag import NaiveRAG

        # Create RAG with mocked components
        rag = NaiveRAG.__new__(NaiveRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 4
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs
        rag.vector_store.similarity_search.return_value = sample_docs
        rag.llm.generate.return_value = "Test answer"

        # Add documents
        rag.add_texts(["Test text"])

        # Query
        answer = rag.query("What is this about?")

        assert answer == "Test answer"
        rag.vector_store.similarity_search.assert_called_once()
        rag.llm.generate.assert_called_once()

    def test_query_with_sources(self):
        """Test querying with sources."""
        from src.rag.naive_rag import NaiveRAG

        # Create RAG with mocked components
        rag = NaiveRAG.__new__(NaiveRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 4
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs
        rag.vector_store.similarity_search.return_value = sample_docs
        rag.llm.generate.return_value = "Test answer"

        # Add documents
        rag.add_texts(["Test text"])

        # Query with sources
        result = rag.query_with_sources("What is this about?")

        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_retrieve(self):
        """Test document retrieval."""
        from src.rag.naive_rag import NaiveRAG

        # Create RAG with mocked components
        rag = NaiveRAG.__new__(NaiveRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 4
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs
        rag.vector_store.similarity_search.return_value = sample_docs

        # Add documents
        rag.add_texts(["Test text"])

        # Retrieve
        docs = rag.retrieve("What is this about?")

        assert len(docs) == len(sample_docs)

    def test_num_documents(self):
        """Test document count."""
        from src.rag.naive_rag import NaiveRAG

        # Create RAG with mocked components
        rag = NaiveRAG.__new__(NaiveRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 4
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []

        assert rag.num_documents == 0

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs

        rag.add_texts(["Text 1", "Text 2"])

        assert rag.num_documents > 0

    def test_num_chunks(self):
        """Test chunk count."""
        from src.rag.naive_rag import NaiveRAG

        # Create RAG with mocked components
        rag = NaiveRAG.__new__(NaiveRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 4
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []

        assert rag.num_chunks == 0

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs

        rag.add_texts(["Text 1"])

        assert rag.num_chunks > 0


class TestAdvancedRAG:
    """Test AdvancedRAG class."""

    def test_add_texts(self):
        """Test adding texts."""
        from src.rag.advanced_rag import AdvancedRAG

        # Create RAG with mocked components
        rag = AdvancedRAG.__new__(AdvancedRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 5
        rag.use_hybrid = False
        rag.use_reranking = False
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []
        rag._retriever = None

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test document 2", metadata={"source": "test2.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs

        # Test add_texts
        texts = ["Text 1", "Text 2"]
        num_chunks = rag.add_texts(texts)

        assert num_chunks == len(sample_docs)

    def test_query(self):
        """Test querying."""
        from src.rag.advanced_rag import AdvancedRAG

        # Create RAG with mocked components
        rag = AdvancedRAG.__new__(AdvancedRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 5
        rag.use_hybrid = False
        rag.use_reranking = False
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []
        rag._retriever = None
        rag._cache = None
        rag._contextual_chunker = None
        rag.use_hyde = False
        rag.use_multi_query_rrf = False
        rag.num_query_variations = 3
        rag._web_searcher = None
        rag._hallucination_grader = None
        rag.use_hallucination_check = False
        rag._context_validator = None

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs
        rag.vector_store.similarity_search.return_value = sample_docs
        rag.llm.generate.return_value = "Test answer"

        # Add documents
        rag.add_texts(["Test text"])

        # Query
        answer = rag.query("What is this about?")

        assert answer == "Test answer"

    def test_query_detailed(self):
        """Test detailed query."""
        from src.rag.advanced_rag import AdvancedRAG

        # Create RAG with mocked components
        rag = AdvancedRAG.__new__(AdvancedRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 5
        rag.use_hybrid = False
        rag.use_reranking = False
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []
        rag._retriever = None
        rag._cache = None
        rag._contextual_chunker = None
        rag.use_hyde = False
        rag.use_multi_query_rrf = False
        rag.num_query_variations = 3
        rag._web_searcher = None
        rag._hallucination_grader = None
        rag.use_hallucination_check = False
        rag._context_validator = None

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs
        rag.vector_store.similarity_search.return_value = sample_docs
        rag.llm.generate.return_value = "Test answer"

        # Add documents
        rag.add_texts(["Test text"])

        # Query with details
        result = rag.query_detailed("What is this about?")

        assert "answer" in result
        assert "transformed_query" in result
        assert "relevant_docs" in result
        assert "citations" in result
        assert result["citations"][0]["source_id"] == "S1"
        assert "total_docs_retrieved" in result
        assert "relevant_docs_count" in result

    def test_query_without_transform(self):
        """Test query without transformation."""
        from src.rag.advanced_rag import AdvancedRAG

        # Create RAG with mocked components
        rag = AdvancedRAG.__new__(AdvancedRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 5
        rag.use_hybrid = False
        rag.use_reranking = False
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []
        rag._retriever = None
        rag._cache = None
        rag._contextual_chunker = None
        rag.use_hyde = False
        rag.use_multi_query_rrf = False
        rag.num_query_variations = 3
        rag._web_searcher = None
        rag._hallucination_grader = None
        rag.use_hallucination_check = False
        rag._context_validator = None

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs
        rag.vector_store.similarity_search.return_value = sample_docs
        rag.llm.generate.return_value = "Test answer"

        # Add documents
        rag.add_texts(["Test text"])

        # Query without transform
        answer = rag.query(
            "What is this about?",
            transform_query=False
        )

        assert answer == "Test answer"

    def test_query_without_grading(self):
        """Test query without grading."""
        from src.rag.advanced_rag import AdvancedRAG

        # Create RAG with mocked components
        rag = AdvancedRAG.__new__(AdvancedRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 5
        rag.use_hybrid = False
        rag.use_reranking = False
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []
        rag._retriever = None
        rag._cache = None
        rag._contextual_chunker = None
        rag.use_hyde = False
        rag.use_multi_query_rrf = False
        rag.num_query_variations = 3
        rag._web_searcher = None
        rag._hallucination_grader = None
        rag.use_hallucination_check = False
        rag._context_validator = None

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs
        rag.vector_store.similarity_search.return_value = sample_docs
        rag.llm.generate.return_value = "Test answer"

        # Add documents
        rag.add_texts(["Test text"])

        # Query without grading
        answer = rag.query(
            "What is this about?",
            grade_documents=False
        )

        assert answer == "Test answer"

    def test_retrieve_with_strategy(self):
        """Test retrieval with specific strategy."""
        from src.rag.advanced_rag import AdvancedRAG

        # Create RAG with mocked components
        rag = AdvancedRAG.__new__(AdvancedRAG)

        # Setup mocks
        rag.document_loader = Mock()
        rag.text_splitter = Mock()
        rag.embeddings = Mock()
        rag.vector_store = Mock()
        rag.llm = Mock()
        rag.retrieval_k = 5
        rag.use_hybrid = False
        rag.use_reranking = False
        rag.system_prompt = "Context: {context}\nQuestion: {question}"
        rag._documents = []
        rag._chunks = []
        rag._retriever = None

        # Setup mock returns
        sample_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        ]

        rag.text_splitter.split_documents.return_value = sample_docs
        rag.vector_store.similarity_search.return_value = sample_docs

        # Add documents
        rag.add_texts(["Test text"])

        # Retrieve with hybrid search
        docs = rag.retrieve(
            "What is this about?",
            use_hybrid=False,
            use_reranking=False
        )

        assert len(docs) == len(sample_docs)


class TestRAGIntegration:
    """Integration tests for RAG components."""

    def test_text_splitter(self):
        """Test text splitter integration."""
        from src.core.text_splitter import TextSplitter
        from langchain_core.documents import Document

        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

        docs = [
            Document(
                page_content="This is a test document. " * 20,
                metadata={"source": "test.txt"}
            )
        ]

        chunks = splitter.split_documents(docs)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 120  # Allow some flexibility

    def test_embeddings_manager(self):
        """Test embeddings manager initialization."""
        from src.core.embeddings import EmbeddingsManager

        # Test local embeddings
        embeddings = EmbeddingsManager(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        assert embeddings.config.provider == "huggingface"

    def test_vector_store_manager(self):
        """Test vector store manager initialization."""
        from src.core.vector_store import VectorStoreManager

        store = VectorStoreManager(provider="faiss")

        assert store.config.provider == "faiss"

    def test_document_loader(self):
        """Test document loader initialization."""
        from src.core.document_loader import DocumentLoader

        loader = DocumentLoader()

        assert loader is not None

    def test_config(self):
        """Test configuration manager."""
        from src.utils.config import Config

        config = Config()

        # Test default values
        chunk_size = config.get_int("CHUNK_SIZE", default=500)
        assert chunk_size == 500
        assert config.get("DEFAULT_EMBEDDING_MODEL") == "keepitreal/vietnamese-sbert"

    def test_cache(self):
        """Test cache functionality."""
        from src.utils.cache import Cache

        cache = Cache(max_size=10, ttl=60)

        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test has
        assert cache.has("key1") is True
        assert cache.has("key2") is False

        # Test delete
        cache.delete("key1")
        assert cache.get("key1") is None

        # Test clear
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size() == 0

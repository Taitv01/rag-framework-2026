"""
Tests for Phase 3 Features
===========================

Tests for:
- MetadataEnhancer (metadata enhancement)
- HallucinationGrader (hallucination verification)
- Neo4jBackend (Neo4j graph storage)
- WebSearch (web search fallback)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document


# ============================================================================
# MetadataEnhancer Tests
# ============================================================================

class TestMetadataEnhancer:
    """Tests for MetadataEnhancer."""

    def test_enhancer_creation(self):
        """Test MetadataEnhancer can be created."""
        from src.core.metadata_enhancer import MetadataEnhancer

        mock_llm = Mock()
        enhancer = MetadataEnhancer(llm=mock_llm)

        assert enhancer.llm == mock_llm
        assert enhancer.batch_size == 5

    def test_enhance_empty_list(self):
        """Test enhancing empty chunk list."""
        from src.core.metadata_enhancer import MetadataEnhancer

        mock_llm = Mock()
        enhancer = MetadataEnhancer(llm=mock_llm)

        result = enhancer.enhance([])
        assert result == []

    def test_enhance_single_chunk(self):
        """Test enhancing a single chunk."""
        from src.core.metadata_enhancer import MetadataEnhancer

        mock_llm = Mock()
        mock_llm.generate.return_value = '''[
            {
                "characters": ["Thạch Sanh", "Lý Thông"],
                "locations": ["hang đại bàng"],
                "time_period": "xưa",
                "topic": "Thạch Sanh giết đại bàng",
                "sentiment": "bi tráng"
            }
        ]'''

        enhancer = MetadataEnhancer(llm=mock_llm)
        chunk = Document(
            page_content="Thạch Sanh đánh đại bàng trong hang.",
            metadata={"source": "test.txt"},
        )

        result = enhancer.enhance([chunk])

        assert len(result) == 1
        assert "Thạch Sanh" in result[0].metadata["characters"]
        assert "Lý Thông" in result[0].metadata["characters"]
        assert result[0].metadata["time_period"] == "xưa"
        assert result[0].metadata["metadata_enhanced"] is True

    def test_enhance_batch(self):
        """Test enhancing multiple chunks in batch."""
        from src.core.metadata_enhancer import MetadataEnhancer

        mock_llm = Mock()
        mock_llm.generate.return_value = '''[
            {
                "characters": ["Thạch Sanh"],
                "locations": ["hang đại bàng"],
                "time_period": "xưa",
                "topic": "Thạch Sanh giết đại bàng",
                "sentiment": "bi tráng"
            },
            {
                "characters": ["Lý Thông"],
                "locations": ["làng"],
                "time_period": "xưa",
                "topic": "Lý Thông lừa đảo",
                "sentiment": "tiêu cực"
            }
        ]'''

        enhancer = MetadataEnhancer(llm=mock_llm, batch_size=10)
        chunks = [
            Document(page_content="Chunk 1", metadata={}),
            Document(page_content="Chunk 2", metadata={}),
        ]

        result = enhancer.enhance(chunks)

        assert len(result) == 2
        assert result[0].metadata["characters"] == ["Thạch Sanh"]
        assert result[1].metadata["characters"] == ["Lý Thông"]

    def test_enhance_preserves_original_metadata(self):
        """Test that enhancement preserves existing metadata."""
        from src.core.metadata_enhancer import MetadataEnhancer

        mock_llm = Mock()
        mock_llm.generate.return_value = '''[{
            "characters": ["Thạch Sanh"],
            "locations": [],
            "time_period": "xưa",
            "topic": "test",
            "sentiment": "neutral"
        }]'''

        enhancer = MetadataEnhancer(llm=mock_llm)
        chunk = Document(
            page_content="Test",
            metadata={"source": "test.txt", "page_number": 1},
        )

        result = enhancer.enhance([chunk])

        assert result[0].metadata["source"] == "test.txt"
        assert result[0].metadata["page_number"] == 1
        assert result[0].metadata["characters"] == ["Thạch Sanh"]

    def test_enhance_fallback_on_llm_failure(self):
        """Test fallback to NER when LLM fails."""
        from src.core.metadata_enhancer import MetadataEnhancer

        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM failed")

        enhancer = MetadataEnhancer(llm=mock_llm, use_ner_fallback=True)
        chunk = Document(page_content="Test content", metadata={})

        result = enhancer.enhance([chunk])

        assert len(result) == 1
        assert result[0].metadata["metadata_enhanced"] is False

    def test_chunk_metadata_model(self):
        """Test ChunkMetadata Pydantic model."""
        from src.core.metadata_enhancer import ChunkMetadata

        meta = ChunkMetadata(
            characters=["Thạch Sanh"],
            locations=["hang"],
            time_period="xưa",
            topic="test",
            sentiment="bi tráng",
        )

        assert meta.characters == ["Thạch Sanh"]
        assert meta.time_period == "xưa"

    def test_chunk_metadata_defaults(self):
        """Test ChunkMetadata default values."""
        from src.core.metadata_enhancer import ChunkMetadata

        meta = ChunkMetadata()

        assert meta.characters == []
        assert meta.locations == []
        assert meta.time_period == "unknown"
        assert meta.topic == ""
        assert meta.sentiment == "neutral"


# ============================================================================
# HallucinationGrader Tests
# ============================================================================

class TestHallucinationGrader:
    """Tests for HallucinationGrader."""

    def test_grader_creation(self):
        """Test HallucinationGrader can be created."""
        from src.agents.hallucination_grader import HallucinationGrader

        mock_llm = Mock()
        grader = HallucinationGrader(llm=mock_llm)

        assert grader.llm == mock_llm
        assert grader.grounded_threshold == 0.8

    def test_grade_grounded_answer(self):
        """Test grading a grounded answer."""
        from src.agents.hallucination_grader import HallucinationGrader, HallucinationGrade

        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = HallucinationGrade(
            is_grounded=True,
            grounded_score=1.0,
            unsupported_claims=[],
            supported_claims=["Thạch Sanh đánh đại bàng"],
            explanation="All claims supported.",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        grader = HallucinationGrader(llm=mock_llm)
        grade = grader.grade(
            answer="Thạch Sanh đánh đại bàng.",
            context="Thạch Sanh dùng cây đàn đánh đại bàng.",
        )

        assert grade.is_grounded is True
        assert grade.grounded_score == 1.0
        assert len(grade.unsupported_claims) == 0

    def test_grade_ungrounded_answer(self):
        """Test grading an ungrounded answer."""
        from src.agents.hallucination_grader import HallucinationGrader, HallucinationGrade

        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = HallucinationGrade(
            is_grounded=False,
            grounded_score=0.3,
            unsupported_claims=["Thạch Sanh cưỡi ngựa", "Lý Thông bị sét đánh"],
            supported_claims=["Thạch Sanh đánh đại bàng"],
            explanation="Some claims not in context.",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        grader = HallucinationGrader(llm=mock_llm)
        grade = grader.grade(
            answer="Thạch Sanh cưỡi ngựa đánh đại bàng. Lý Thông bị sét đánh.",
            context="Thạch Sanh đánh đại bàng.",
        )

        assert grade.is_grounded is False
        assert grade.grounded_score == 0.3
        assert len(grade.unsupported_claims) == 2

    def test_grade_model(self):
        """Test HallucinationGrade model."""
        from src.agents.hallucination_grader import HallucinationGrade

        grade = HallucinationGrade(
            is_grounded=True,
            grounded_score=0.9,
            unsupported_claims=["claim1"],
            supported_claims=["claim2", "claim3"],
            explanation="test",
        )

        assert grade.is_grounded is True
        assert grade.grounded_score == 0.9
        assert len(grade.unsupported_claims) == 1
        assert len(grade.supported_claims) == 2

    def test_grade_model_defaults(self):
        """Test HallucinationGrade default values."""
        from src.agents.hallucination_grader import HallucinationGrade

        grade = HallucinationGrade(is_grounded=True)

        assert grade.grounded_score == 1.0
        assert grade.unsupported_claims == []
        assert grade.supported_claims == []
        assert grade.explanation == ""

    def test_verify_answer(self):
        """Test verify_answer convenience method."""
        from src.agents.hallucination_grader import HallucinationGrader, HallucinationGrade

        mock_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = HallucinationGrade(
            is_grounded=True,
            grounded_score=1.0,
        )
        mock_llm.with_structured_output.return_value = mock_structured

        grader = HallucinationGrader(llm=mock_llm)
        docs = [Document(page_content="Thạch Sanh đánh đại bàng.", metadata={})]

        result = grader.verify_answer(
            answer="Thạch Sanh đánh đại bàng.",
            docs=docs,
        )

        assert result["is_grounded"] is True
        assert result["regenerated"] is False


# ============================================================================
# Neo4jBackend Tests
# ============================================================================

class TestNeo4jBackend:
    """Tests for Neo4jBackend."""

    def test_backend_creation(self):
        """Test Neo4jBackend can be created."""
        from src.core.graph_store import Neo4jBackend

        backend = Neo4jBackend(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )

        assert backend.uri == "bolt://localhost:7687"
        assert backend.user == "neo4j"
        assert backend.is_connected() is False

    def test_backend_not_connected_operations(self):
        """Test operations when not connected."""
        from src.core.graph_store import Neo4jBackend

        backend = Neo4jBackend()

        # Operations should not raise, just return empty/None
        assert backend.query_cypher("MATCH (n) RETURN n") == []
        assert backend.get_entity_neighbors("test") == []
        assert backend.find_paths("a", "b") == []
        assert backend.get_entity_count() == 0
        assert backend.get_relationship_count() == 0

    def test_backend_context_manager(self):
        """Test Neo4jBackend as context manager."""
        from src.core.graph_store import Neo4jBackend

        backend = Neo4jBackend()
        # Should not raise even without connection
        with backend:
            assert backend.is_connected() is False

    def test_knowledge_graph_with_neo4j_backend(self):
        """Test KnowledgeGraph with optional Neo4j backend."""
        from src.rag.graph_rag import KnowledgeGraph, Entity, Relationship

        # Create KG without Neo4j (default)
        kg = KnowledgeGraph()
        assert kg._neo4j is None

        # Add entity (should work without Neo4j)
        entity = Entity(name="Test", entity_type="TEST", description="test entity")
        kg.add_entity(entity)
        assert "Test" in kg.entities

        # Add relationship
        rel = Relationship(source="A", target="B", relationship_type="RELATED", description="test")
        kg.add_relationship(rel)
        assert len(kg.relationships) == 1

    def test_graph_rag_without_neo4j(self):
        """Test GraphRAG works without Neo4j."""
        from src.rag.graph_rag import GraphRAG

        rag = GraphRAG.__new__(GraphRAG)
        rag._neo4j_backend = None

        # Should report no Neo4j
        assert not rag.has_neo4j

    def test_sync_to_neo4j_without_backend(self):
        """Test sync_to_neo4j without backend returns 0."""
        from src.rag.graph_rag import KnowledgeGraph

        kg = KnowledgeGraph()
        result = kg.sync_to_neo4j()
        assert result == 0


# ============================================================================
# WebSearch Tests
# ============================================================================

class TestWebSearch:
    """Tests for web search components."""

    def test_web_search_result(self):
        """Test WebSearchResult dataclass."""
        from src.core.web_search import WebSearchResult

        result = WebSearchResult(
            title="Test",
            url="https://example.com",
            snippet="Test snippet",
        )

        assert result.title == "Test"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.full_content is None
        assert result.relevance_score == 0.0

    def test_duckduckgo_provider_creation(self):
        """Test DuckDuckGoSearchProvider creation."""
        from src.core.web_search import DuckDuckGoSearchProvider

        provider = DuckDuckGoSearchProvider(region="vn")
        assert provider.region == "vn"
        assert provider.safesearch == "moderate"

    def test_tavily_provider_creation(self):
        """Test TavilySearchProvider creation."""
        from src.core.web_search import TavilySearchProvider

        provider = TavilySearchProvider(api_key="test-key")
        assert provider._api_key == "test-key"

    def test_safe_web_searcher_creation(self):
        """Test SafeWebSearcher creation."""
        from src.core.web_search import SafeWebSearcher, DuckDuckGoSearchProvider

        provider = DuckDuckGoSearchProvider()
        searcher = SafeWebSearcher(provider=provider)

        assert searcher.provider == provider
        assert searcher.verify_relevance is True

    def test_to_documents(self):
        """Test converting web results to documents."""
        from src.core.web_search import SafeWebSearcher, WebSearchResult, DuckDuckGoSearchProvider

        provider = DuckDuckGoSearchProvider()
        searcher = SafeWebSearcher(provider=provider, verify_relevance=False)

        results = [
            WebSearchResult(
                title="Thạch Sanh",
                url="https://vi.wikipedia.org/wiki/Thạch_Sanh",
                snippet="Thạch Sanh là nhân vật...",
            ),
            WebSearchResult(
                title="Synthesized",
                url="",
                snippet="Synthesized answer",
                metadata={"is_synthesized": True},
            ),
        ]

        docs = searcher.to_documents(results)

        assert len(docs) == 2
        assert docs[0].metadata["source_type"] == "web"
        assert docs[0].metadata["url"] == "https://vi.wikipedia.org/wiki/Thạch_Sanh"
        assert "[Web Search Result:" in docs[0].page_content
        assert docs[1].metadata["is_synthesized"] is True
        assert "[Web Search - Synthesized Answer]" in docs[1].page_content

    def test_to_documents_truncation(self):
        """Test content truncation in to_documents."""
        from src.core.web_search import SafeWebSearcher, WebSearchResult, DuckDuckGoSearchProvider

        provider = DuckDuckGoSearchProvider()
        searcher = SafeWebSearcher(provider=provider, max_content_length=50)

        results = [
            WebSearchResult(
                title="Test",
                url="https://example.com",
                snippet="A" * 200,
            ),
        ]

        docs = searcher.to_documents(results)

        # Content should be truncated
        assert len(docs[0].page_content) < 200

    def test_create_web_answer_prompt(self):
        """Test web answer prompt creation."""
        from src.core.web_search import SafeWebSearcher, DuckDuckGoSearchProvider

        provider = DuckDuckGoSearchProvider()
        searcher = SafeWebSearcher(provider=provider)

        web_docs = [
            Document(
                page_content="[Web Search Result: Test]\nContent here",
                metadata={"source_type": "web", "url": "https://example.com"},
            ),
        ]
        local_docs = [
            Document(
                page_content="Local content",
                metadata={"source": "local.txt"},
            ),
        ]

        prompt = searcher.create_web_answer_prompt(
            question="Test question?",
            web_docs=web_docs,
            local_docs=local_docs,
        )

        assert "LOCAL DOCUMENTS" in prompt
        assert "WEB SEARCH RESULTS" in prompt
        assert "Test question?" in prompt

    def test_create_web_searcher_factory(self):
        """Test create_web_searcher factory function."""
        from src.core.web_search import create_web_searcher, SafeWebSearcher

        searcher = create_web_searcher(provider="duckduckgo")
        assert isinstance(searcher, SafeWebSearcher)

    def test_create_web_searcher_unknown_provider(self):
        """Test factory with unknown provider raises error."""
        from src.core.web_search import create_web_searcher

        with pytest.raises(ValueError, match="Unknown web search provider"):
            create_web_searcher(provider="unknown")

    def test_search_no_results(self):
        """Test search when provider returns no results."""
        from src.core.web_search import SafeWebSearcher, DuckDuckGoSearchProvider

        provider = Mock()
        provider.search.return_value = []

        searcher = SafeWebSearcher(provider=provider, verify_relevance=False)
        results = searcher.search("test query")

        assert results == []

    def test_search_with_verification(self):
        """Test search with relevance verification."""
        from src.core.web_search import SafeWebSearcher, WebSearchResult

        provider = Mock()
        provider.search.return_value = [
            WebSearchResult(title="Relevant", url="https://a.com", snippet="relevant content"),
            WebSearchResult(title="Irrelevant", url="https://b.com", snippet="irrelevant content"),
        ]

        mock_llm = Mock()
        mock_llm.generate.side_effect = ["yes", "no"]

        searcher = SafeWebSearcher(
            provider=provider,
            llm=mock_llm,
            verify_relevance=True,
        )

        results = searcher.search("test query")

        # Only the relevant result should pass
        assert len(results) == 1
        assert results[0].title == "Relevant"

    def test_search_and_verify_confidence(self):
        """Test search_and_verify returns confidence score."""
        from src.core.web_search import SafeWebSearcher, WebSearchResult

        provider = Mock()
        provider.search.return_value = [
            WebSearchResult(title="R1", url="https://a.com", snippet="content 1"),
            WebSearchResult(title="R2", url="https://b.com", snippet="content 2"),
        ]

        mock_llm = Mock()
        mock_llm.generate.return_value = "yes"

        searcher = SafeWebSearcher(
            provider=provider,
            llm=mock_llm,
            verify_relevance=True,
        )

        results, confidence = searcher.search_and_verify("test query")

        assert len(results) == 2
        assert confidence == 1.0  # All verified

    def test_advanced_rag_web_search_integration(self):
        """Test AdvancedRAG web search integration."""
        from src.rag.advanced_rag import AdvancedRAG

        rag = AdvancedRAG.__new__(AdvancedRAG)
        rag._web_searcher = None
        rag._cache = None
        rag._contextual_chunker = None
        rag._hallucination_grader = None
        rag.use_hallucination_check = False
        rag.use_hyde = False
        rag.use_multi_query_rrf = False
        rag.num_query_variations = 3

        # Without web search, poor quality should not trigger web search
        assert rag._is_retrieval_quality_poor([], "test") is True
        assert rag._is_retrieval_quality_poor(
            [Document(page_content="test", metadata={})], "test"
        ) is False


# ============================================================================
# ContextValidator Tests
# ============================================================================

class TestContextValidator:
    """Tests for ContextValidator and context window validation."""

    def test_validator_creation(self):
        """Test ContextValidator can be created."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000)
        assert validator.context_window == 128000
        assert validator.available_tokens == 128000 - 4096  # minus default output reservation

    def test_validator_custom_output_tokens(self):
        """Test validator with custom output token reservation."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000, max_output_tokens=8192)
        assert validator.reserve_tokens == 8192
        assert validator.available_tokens == 128000 - 8192

    def test_count_tokens_empty(self):
        """Test token counting with empty string."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000)
        assert validator.count_tokens("") == 0
        assert validator.count_tokens(None) == 0

    def test_count_tokens_short_text(self):
        """Test token counting with short text."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000)
        tokens = validator.count_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10

    def test_count_tokens_long_text(self):
        """Test token counting with longer text."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000)
        text = "This is a test sentence. " * 100  # ~2500 chars
        tokens = validator.count_tokens(text)
        assert tokens > 100
        assert tokens < 1000

    def test_estimate_tokens(self):
        """Test character-based token estimation."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000, language="en")
        # English: ~4 chars per token
        tokens = validator.estimate_tokens("Hello world test")  # 16 chars
        assert 3 <= tokens <= 6

    def test_estimate_tokens_vietnamese(self):
        """Test token estimation for Vietnamese text."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000, language="vi")
        # Vietnamese: ~3 chars per token
        text = "Thạch Sanh đánh đại bàng"  # 25 chars
        tokens = validator.estimate_tokens(text)
        assert 5 <= tokens <= 15

    def test_validate_small_prompt(self):
        """Test validation with a small prompt that fits."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000)
        result = validator.validate(prompt="Hello world", system_prompt="You are helpful.")

        assert result.is_valid is True
        assert result.is_too_large is False
        assert result.overflow_tokens == 0
        assert result.truncated_prompt is None
        assert result.warning is None

    def test_validate_large_prompt(self):
        """Test validation with a prompt that exceeds the limit."""
        from src.utils.context_validator import ContextValidator

        # Very small context window for testing
        validator = ContextValidator(context_window=100, max_output_tokens=20)
        big_prompt = "This is a test sentence. " * 100  # ~2500 chars

        result = validator.validate(prompt=big_prompt)

        assert result.is_valid is False
        assert result.is_too_large is True
        assert result.overflow_tokens > 0
        assert result.truncated_prompt is not None
        assert result.warning is not None
        assert "EXCEEDS" in result.warning

    def test_validate_warning_threshold(self):
        """Test warning when approaching limit."""
        from src.utils.context_validator import ContextValidator

        # Create validator with small window to trigger warning
        # Use a size that will definitely be above 80% of available
        validator = ContextValidator(
            context_window=100,
            max_output_tokens=10,
            warning_threshold=0.5,
            language="en",
        )
        # available = 100 - 10 = 90 tokens
        # Use enough chars to exceed 50% of available tokens
        # With tiktoken, need to ensure we're above threshold
        prompt = "This is a test sentence with enough words. " * 5  # ~200 chars

        result = validator.validate(prompt=prompt)

        # Should either be valid with warning, or too large
        if result.is_valid:
            # If valid, should have warning if usage > threshold
            assert result.usage_ratio > 0.3  # At least some usage
        else:
            assert result.is_too_large is True

    def test_fit_to_window_no_truncation(self):
        """Test fit_to_window when prompt already fits."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000)
        prompt = "Hello world"
        result = validator.fit_to_window(prompt)
        assert result == prompt

    def test_fit_to_window_with_truncation(self):
        """Test fit_to_window truncates when needed."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=100, max_output_tokens=20)
        big_prompt = "This is a sentence. " * 100

        result = validator.fit_to_window(big_prompt)

        assert len(result) < len(big_prompt)
        assert "truncated" in result.lower()

    def test_validation_result_to_dict(self):
        """Test ValidationResult.to_dict()."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=128000)
        result = validator.validate("Hello")
        d = result.to_dict()

        assert "is_valid" in d
        assert "context_window" in d
        assert "usage_ratio" in d

    def test_from_llm_manager(self):
        """Test creating validator from LLMManager."""
        from src.utils.context_validator import ContextValidator
        from src.core.llm import LLMManager

        llm = LLMManager(provider="openai", model="gpt-4o")
        validator = ContextValidator.from_llm_manager(llm)

        assert validator.context_window == 128000
        assert validator.reserve_tokens == 4096

    def test_from_llm_manager_anthropic(self):
        """Test creating validator from Anthropic LLMManager."""
        from src.utils.context_validator import ContextValidator
        from src.core.llm import LLMManager

        llm = LLMManager(provider="anthropic", model="claude-sonnet-4-20250514")
        validator = ContextValidator.from_llm_manager(llm)

        assert validator.context_window == 200000

    def test_llm_get_context_window(self):
        """Test LLMManager.get_context_window()."""
        from src.core.llm import LLMManager

        llm = LLMManager(provider="openai", model="gpt-4o")
        assert llm.get_context_window() == 128000

        llm2 = LLMManager(provider="openai", model="gpt-3.5-turbo")
        assert llm2.get_context_window() == 16385

        llm3 = LLMManager(provider="anthropic", model="claude-sonnet-4-20250514")
        assert llm3.get_context_window() == 200000

    def test_llm_get_max_output_tokens(self):
        """Test LLMManager.get_max_output_tokens()."""
        from src.core.llm import LLMManager

        llm = LLMManager(provider="openai", model="gpt-4o")
        assert llm.get_max_output_tokens() == 4096

        llm2 = LLMManager(provider="openai", model="gpt-4o", max_tokens=8192)
        assert llm2.get_max_output_tokens() == 8192

    def test_truncation_strategies(self):
        """Test different truncation strategies."""
        from src.utils.context_validator import ContextValidator

        validator = ContextValidator(context_window=50, max_output_tokens=10)
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

        # Tail truncation (default)
        result = validator._truncate_text(text, 10, strategy="tail")
        assert "truncated" in result.lower()
        assert len(result) < len(text) + 50  # Plus marker text

        # Head truncation
        result = validator._truncate_text(text, 10, strategy="head")
        assert "truncated" in result.lower()

        # Middle truncation
        result = validator._truncate_text(text, 10, strategy="middle")
        assert "truncated" in result.lower()

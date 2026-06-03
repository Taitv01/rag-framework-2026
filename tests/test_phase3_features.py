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

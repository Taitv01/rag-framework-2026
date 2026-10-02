"""
Tests for Upgraded RAG Features: AgenticRAG, GraphRAG, and AdaptiveRAG
========================================================================
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document

from src.rag import AgenticRAG, GraphRAG, AdaptiveRAG
from src.rag.graph_rag import Entity, Relationship


def test_agentic_rag_check_hallucination():
    """Verify AgenticRAG hallucination checking functionality."""
    rag = AgenticRAG.__new__(AgenticRAG)
    rag.llm = Mock()
    rag.llm.generate.return_value = "Grounded: yes - Trả lời hoàn toàn dựa vào ngữ cảnh."
    
    result = rag.check_hallucination("Thạch Sanh bắn đại bàng.", "Thạch Sanh đã bắn rơi con đại bàng.")
    assert result["is_grounded"] is True
    assert result["hallucination_score"] == 0.0


def test_agentic_rag_check_hallucination_parses_structured_score():
    """Structured grades preserve the model's bounded hallucination score."""
    rag = AgenticRAG.__new__(AgenticRAG)
    rag.llm = Mock()
    rag.llm.generate.return_value = (
        '{"grounded": false, "hallucination_score": 0.75, '
        '"reasoning": "Chi tiết này không có trong ngữ cảnh."}'
    )

    result = rag.check_hallucination("Thạch Sanh bắn đại bàng.", "Lý Thông bắn đại bàng.")

    assert result["is_grounded"] is False
    assert result["hallucination_score"] == 0.75


def test_agentic_rag_check_hallucination_fails_closed():
    """A failed or ambiguous grade must never be reported as grounded."""
    rag = AgenticRAG.__new__(AgenticRAG)
    rag.llm = Mock()
    rag.llm.generate.return_value = "Tôi không thể đánh giá. Có thể đúng."

    result = rag.check_hallucination("Ngữ cảnh", "Câu trả lời")

    assert result["is_grounded"] is False
    assert result["hallucination_score"] == 1.0


def test_graph_rag_extract_subgraph_context():
    """Verify GraphRAG sub-graph extraction functionality."""
    rag = GraphRAG.__new__(GraphRAG)
    rag.knowledge_graph = Mock()
    rag.knowledge_graph.entities = {"Thạch Sanh": Entity("Thạch Sanh", "NhanVat", "Dũng sĩ")}
    
    rel = Relationship("Thạch Sanh", "Công Chúa", "GIAI_CUU", "Giải cứu công chúa")
    rag.knowledge_graph.relationships = [rel]
    rag.llm = Mock()
    rag.llm.generate.return_value = "Thạch Sanh, Công Chúa"
    rag.knowledge_graph.get_entity.return_value = Entity("Thạch Sanh", "NhanVat", "Dũng sĩ")
    rag.knowledge_graph.get_neighbors.return_value = {}
    rag.knowledge_graph.get_relationships.return_value = [rel]

    res = rag.extract_subgraph_context("Thạch Sanh giải cứu ai?")
    assert "matched_entities" in res
    assert "subgraph_triples" in res
    assert len(res["subgraph_triples"]) > 0


def test_graph_rag_extract_subgraph_context_respects_max_hops():
    """Subgraph traversal includes a chain only up to the requested depth."""
    rag = GraphRAG.__new__(GraphRAG)
    rag.knowledge_graph = MagicMock()
    rag.knowledge_graph.entities = {
        "A": Entity("A", "NhanVat", "Điểm bắt đầu"),
        "B": Entity("B", "NhanVat", "Một hop"),
        "C": Entity("C", "DiaDiem", "Hai hop"),
        "D": Entity("D", "VatThe", "Ba hop"),
    }
    rag.knowledge_graph.relationships = [
        Relationship("A", "B", "GAP_GO", ""),
        Relationship("B", "C", "DEN", ""),
        Relationship("C", "D", "SO_HUU", ""),
    ]

    one_hop = rag.extract_subgraph_context("A là ai?", max_hops=1)
    two_hops = rag.extract_subgraph_context("A là ai?", max_hops=2)

    assert one_hop["subgraph_triples"] == ["A -[GAP_GO]-> B"]
    assert two_hops["subgraph_triples"] == [
        "A -[GAP_GO]-> B",
        "B -[DEN]-> C",
    ]
    assert "Entity: C" not in one_hop["formatted_context"]
    assert "Entity: C" in two_hops["formatted_context"]


def test_graph_rag_extract_subgraph_context_validates_max_hops():
    rag = GraphRAG.__new__(GraphRAG)
    rag.knowledge_graph = MagicMock()
    rag.knowledge_graph.entities = {}
    rag.knowledge_graph.relationships = []

    with pytest.raises(ValueError, match="max_hops"):
        rag.extract_subgraph_context("question", max_hops=-1)


def test_adaptive_rag_route_stats():
    """Verify AdaptiveRAG routing analytics."""
    rag = AdaptiveRAG.__new__(AdaptiveRAG)
    rag._route_stats = {"simple": 5, "medium": 3, "complex": 2}
    stats = rag.route_stats
    assert stats["simple"] == 5
    assert stats["medium"] == 3
    assert stats["complex"] == 2

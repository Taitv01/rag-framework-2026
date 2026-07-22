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


def test_adaptive_rag_route_stats():
    """Verify AdaptiveRAG routing analytics."""
    rag = AdaptiveRAG.__new__(AdaptiveRAG)
    rag._route_stats = {"simple": 5, "medium": 3, "complex": 2}
    stats = rag.route_stats
    assert stats["simple"] == 5
    assert stats["medium"] == 3
    assert stats["complex"] == 2

"""
RAG Pipelines
=============

Multiple RAG implementation patterns for different use cases.

Supported patterns:
- NaiveRAG: Basic vector search + LLM generation
- AdvancedRAG: Hybrid search with re-ranking
- AgenticRAG: Agent-based retrieval with LangGraph
- GraphRAG: Knowledge graph integration
- AdaptiveRAG: Intelligent routing based on query complexity
- MultimodalRAG: Multi-format document support
"""

from src.rag.naive_rag import NaiveRAG
from src.rag.advanced_rag import AdvancedRAG
from src.rag.agentic_rag import AgenticRAG
from src.rag.graph_rag import GraphRAG
from src.rag.adaptive_rag import AdaptiveRAG

__all__ = [
    "NaiveRAG",
    "AdvancedRAG",
    "AgenticRAG",
    "GraphRAG",
    "AdaptiveRAG",
]

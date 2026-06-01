"""
Ultimate RAG Framework
======================

A comprehensive Retrieval-Augmented Generation framework supporting multiple
RAG paradigms for AI models (2026).

Supported RAG Patterns:
- Naive RAG: Basic vector search + LLM generation
- Advanced RAG: Hybrid search with re-ranking
- Agentic RAG: Agent-based retrieval with LangGraph
- Graph RAG: Knowledge graph integration
- Multimodal RAG: Multi-format document support

Usage:
    from src.rag import NaiveRAG, AdvancedRAG, AgenticRAG
    from src.core import DocumentLoader, TextSplitter, Embeddings
"""

__version__ = "1.0.0"
__author__ = "RAG Framework Contributors"

from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingsManager
from src.core.vector_store import VectorStoreManager
from src.core.retriever import RetrieverManager
from src.core.llm import LLMManager

__all__ = [
    "DocumentLoader",
    "TextSplitter",
    "EmbeddingsManager",
    "VectorStoreManager",
    "RetrieverManager",
    "LLMManager",
]

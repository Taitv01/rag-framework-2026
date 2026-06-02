"""
Core Components
===============

Fundamental building blocks for the RAG framework.
"""

from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingsManager
from src.core.vector_store import VectorStoreManager
from src.core.retriever import RetrieverManager
from src.core.llm import LLMManager
from src.core.vietnamese_processor import VietnameseProcessor, get_vietnamese_processor

__all__ = [
    "DocumentLoader",
    "TextSplitter",
    "EmbeddingsManager",
    "VectorStoreManager",
    "RetrieverManager",
    "LLMManager",
    "VietnameseProcessor",
    "get_vietnamese_processor",
]

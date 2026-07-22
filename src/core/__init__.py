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
from src.core.metadata_enhancer import MetadataEnhancer
from src.core.graph_store import Neo4jBackend
from src.core.web_search import SafeWebSearcher, DuckDuckGoSearchProvider, TavilySearchProvider, create_web_searcher
from src.core.markdown_index import MarkdownFolderIndexer, MarkdownRefreshResult
from src.core.ocr_engine import OCREngine
from src.core.library_manager import LibraryManager, DocumentClassifier

__all__ = [
    "DocumentLoader",
    "TextSplitter",
    "EmbeddingsManager",
    "VectorStoreManager",
    "RetrieverManager",
    "LLMManager",
    "VietnameseProcessor",
    "get_vietnamese_processor",
    "MetadataEnhancer",
    "Neo4jBackend",
    "SafeWebSearcher",
    "DuckDuckGoSearchProvider",
    "TavilySearchProvider",
    "create_web_searcher",
    "MarkdownFolderIndexer",
    "MarkdownRefreshResult",
    "OCREngine",
    "LibraryManager",
    "DocumentClassifier",
]

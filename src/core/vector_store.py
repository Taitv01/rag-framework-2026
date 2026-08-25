"""
Vector Store Manager
===================

Abstraction layer for vector databases supporting multiple backends.

Supported backends:
- FAISS (in-memory, prototyping)
- ChromaDB (persistent, local)
- Qdrant (production, scalable)

Usage:
    # FAISS (in-memory)
    store = VectorStoreManager(provider="faiss", embeddings=embeddings)

    # ChromaDB (persistent)
    store = VectorStoreManager(provider="chroma", persist_directory="./chroma_db")

    # Add documents
    store.add_documents(documents)

    # Search
    results = store.similarity_search("query", k=5)
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    provider: str = "faiss"
    collection_name: str = "default"
    persist_directory: Optional[str] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class VectorStoreManager:
    """
    Vector store manager with multi-backend support.

    Provides a unified interface for different vector databases.

    Example:
        from src.core import EmbeddingsManager, VectorStoreManager

        # Create embeddings
        embeddings = EmbeddingsManager(provider="huggingface")

        # Create vector store
        store = VectorStoreManager(
            provider="chroma",
            embeddings=embeddings,
            persist_directory="./chroma_db"
        )

        # Add documents
        store.add_documents(documents)

        # Search
        results = store.similarity_search("What is Python?", k=5)
    """

    def __init__(
        self,
        provider: str = "faiss",
        embeddings=None,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize vector store manager.

        Args:
            provider: Vector store backend ('faiss', 'chroma', 'qdrant')
            embeddings: Embeddings instance
            collection_name: Name of the collection/index
            persist_directory: Directory for persistent storage
            url: URL for remote vector stores
            api_key: API key for remote vector stores
        """
        self.config = VectorStoreConfig(
            provider=provider,
            collection_name=collection_name,
            persist_directory=persist_directory,
            url=url,
            api_key=api_key,
        )

        self.embeddings = embeddings
        self._store = None

    @property
    def store(self):
        """Get or create vector store instance."""
        if self._store is None:
            self._store = self._create_store()
        return self._store

    def _create_store(self):
        """Create vector store instance based on provider."""
        if self.config.provider == "faiss":
            return self._create_faiss_store()
        elif self.config.provider == "chroma":
            return self._create_chroma_store()
        elif self.config.provider == "qdrant":
            return self._create_qdrant_store()
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _create_faiss_store(self):
        """Create FAISS vector store."""
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISS. "
                "Install it with: pip install faiss-cpu"
            )

        # Return empty store - will be populated with add_documents
        return None

    def _create_chroma_store(self):
        """Create ChromaDB vector store."""
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            raise ImportError(
                "chromadb is required for Chroma. "
                "Install it with: pip install chromadb"
            )

        return Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings.embeddings,
            persist_directory=self.config.persist_directory,
        )

    def _create_qdrant_store(self):
        """Create Qdrant vector store."""
        try:
            from langchain_community.vectorstores import Qdrant
        except ImportError:
            raise ImportError(
                "qdrant-client is required for Qdrant. "
                "Install it with: pip install qdrant-client"
            )

        import os

        url = self.config.url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = self.config.api_key or os.getenv("QDRANT_API_KEY")

        return Qdrant(
            collection_name=self.config.collection_name,
            embeddings=self.embeddings.embeddings,
            url=url,
            api_key=api_key,
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to vector store.

        Args:
            documents: List of Document objects
            ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        if self.config.provider == "faiss":
            return self._add_to_faiss(documents, ids)
        else:
            return self.store.add_documents(documents, ids=ids)

    def _add_to_faiss(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to FAISS store."""
        from langchain_community.vectorstores import FAISS

        if self._store is None:
            self._store = FAISS.from_documents(
                documents,
                self.embeddings.embeddings
            )
        else:
            self._store.add_documents(documents)

        # Generate unique IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter
            **kwargs: Additional search parameters

        Returns:
            List of similar Document objects
        """
        return self.store.similarity_search(
            query,
            k=k,
            filter=filter,
            **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter

        Returns:
            List of (Document, score) tuples
        """
        return self.store.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
            **kwargs
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete documents from vector store.

        Args:
            ids: Document IDs to delete
            filter: Metadata filter for deletion
        """
        if self.config.provider == "faiss":
            raise NotImplementedError(
                "FAISS does not support deletion. "
                "Workaround: rebuild the index without the documents you want to remove. "
                "Or use ChromaDB/Qdrant for full CRUD support."
            )

        if ids:
            self.store.delete(ids)
        elif filter:
            self.store.delete(filter=filter)

    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = 4,
        **kwargs
    ):
        """
        Get a retriever interface.

        Args:
            search_type: Type of search ('similarity', 'mmr', 'similarity_score_threshold')
            k: Number of results
            **kwargs: Additional retriever parameters

        Returns:
            Retriever instance
        """
        return self.store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, **kwargs}
        )

    def persist(self) -> None:
        """Persist vector store to disk (if supported)."""
        if self.config.provider == "chroma":
            # ChromaDB auto-persists
            pass
        elif self.config.provider == "faiss":
            if self._store and self.config.persist_directory:
                self._store.save_local(self.config.persist_directory)

    @classmethod
    def from_existing(
        cls,
        provider: str,
        embeddings,
        persist_directory: str,
        collection_name: str = "default",
        **kwargs
    ) -> "VectorStoreManager":
        """
        Load existing vector store.

        Args:
            provider: Vector store provider
            embeddings: Embeddings instance
            persist_directory: Directory with existing data
            collection_name: Collection name
            **kwargs: Additional arguments

        Returns:
            VectorStoreManager instance
        """
        manager = cls(
            provider=provider,
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )

        # Force creation of store
        _ = manager.store

        return manager


# Convenience functions
def create_faiss_store(embeddings, documents: Optional[List[Document]] = None):
    """
    Create FAISS vector store.

    Args:
        embeddings: Embeddings instance
        documents: Optional documents to add

    Returns:
        VectorStoreManager instance
    """
    store = VectorStoreManager(provider="faiss", embeddings=embeddings)
    if documents:
        store.add_documents(documents)
    return store


def create_chroma_store(
    embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "default"
):
    """
    Create ChromaDB vector store.

    Args:
        embeddings: Embeddings instance
        persist_directory: Directory for persistence
        collection_name: Collection name

    Returns:
        VectorStoreManager instance
    """
    return VectorStoreManager(
        provider="chroma",
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

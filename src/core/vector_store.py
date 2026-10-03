"""
Vector Store Manager
===================

Abstraction layer for vector databases supporting multiple backends.

Supported backends:
- FAISS (local; saved to ``persist_directory`` when one is given)
- ChromaDB (persistent, local)
- Qdrant (production): a server (``url``), an embedded on-disk store
  (``persist_directory``) or an in-memory store (``url=":memory:"``)

Usage:
    # FAISS (in-memory)
    store = VectorStoreManager(provider="faiss", embeddings=embeddings)

    # Qdrant without a server, kept on disk
    store = VectorStoreManager(provider="qdrant", embeddings=embeddings,
                               persist_directory="./qdrant_data")

    # Add documents
    store.add_documents(documents, ids=chunk_ids)

    # Search
    results = store.similarity_search("query", k=5, filter={"source_root": "..."})
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from langchain_core.documents import Document

# Namespace for mapping arbitrary document ids to the UUIDs Qdrant requires.
QDRANT_ID_NAMESPACE = uuid.UUID("6f1d3c1e-7a51-4c55-9b0a-2f4d1b9e8c11")


def qdrant_point_id(doc_id: Any):
    """Qdrant accepts only UUIDs and unsigned ints; map other ids to a stable UUID."""
    if isinstance(doc_id, int) and doc_id >= 0:
        return doc_id
    text = str(doc_id)
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return str(uuid.uuid5(QDRANT_ID_NAMESPACE, text))


def metadata_matches(metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
    """Equality filter on metadata; a list value matches any of its items."""
    metadata = metadata or {}
    for key, expected in filter.items():
        value = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True


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

    Provides a unified interface for different vector databases. Metadata
    filters are plain dicts (``{"source_root": "/docs"}``) for every backend.

    Example:
        from src.core import EmbeddingsManager, VectorStoreManager

        # Create embeddings
        embeddings = EmbeddingsManager(provider="huggingface")

        # Create vector store
        store = VectorStoreManager(
            provider="qdrant",
            embeddings=embeddings,
            persist_directory="./qdrant_data"
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
            persist_directory: Directory for persistent storage (Qdrant: embedded
                on-disk mode when no url is given)
            url: URL for remote vector stores (Qdrant: ":memory:" for in-memory)
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

    def _faiss_index_file(self) -> Optional[Path]:
        if not self.config.persist_directory:
            return None
        return Path(self.config.persist_directory) / f"{self.config.collection_name}.faiss"

    def _create_faiss_store(self):
        """Load a saved FAISS index, or return None until documents are added."""
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISS. "
                "Install it with: pip install faiss-cpu"
            )

        index_file = self._faiss_index_file()
        if index_file is None or not index_file.is_file():
            return None

        # The pickle beside the index is the docstore this class saved; only
        # configured persist directories are loaded.
        return FAISS.load_local(
            self.config.persist_directory,
            self.embeddings.embeddings,
            index_name=self.config.collection_name,
            allow_dangerous_deserialization=True,
        )

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

    def _qdrant_client(self):
        """Client for the configured Qdrant mode: memory, server or embedded on disk."""
        from qdrant_client import QdrantClient

        api_key = self.config.api_key or os.getenv("QDRANT_API_KEY") or None
        if self.config.url == ":memory:":
            return QdrantClient(location=":memory:")
        if self.config.url:
            return QdrantClient(url=self.config.url, api_key=api_key)
        if self.config.persist_directory:
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=self.config.persist_directory)
        return QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key,
        )

    def _create_qdrant_store(self):
        """Create a Qdrant vector store, creating the collection on first use."""
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import models
        except ImportError:
            raise ImportError(
                "qdrant-client and langchain-qdrant are required for Qdrant. "
                "Install them with: pip install -e \".[qdrant]\""
            )

        client = self._qdrant_client()
        embeddings = self.embeddings.embeddings
        name = self.config.collection_name
        if not client.collection_exists(name):
            size = len(embeddings.embed_query("dimension probe"))
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
            )

        return QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)

    def _native_filter(self, filter: Optional[Dict[str, Any]]):
        """Translate a metadata dict into the backend's filter type."""
        if not filter or self.config.provider != "qdrant" or not isinstance(filter, dict):
            return filter

        from qdrant_client import models

        conditions = []
        for key, value in filter.items():
            match = (
                models.MatchAny(any=list(value))
                if isinstance(value, (list, tuple, set))
                else models.MatchValue(value=value)
            )
            conditions.append(models.FieldCondition(key=f"metadata.{key}", match=match))
        return models.Filter(must=conditions)

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add (or replace) documents in the vector store.

        Args:
            documents: List of Document objects
            ids: Optional list of document IDs; an existing ID is overwritten

        Returns:
            List of document IDs
        """
        if not documents:
            return []
        if self.config.provider == "faiss":
            return self._add_to_faiss(documents, ids)
        if self.config.provider == "qdrant":
            point_ids = [qdrant_point_id(i) for i in ids] if ids else None
            self.store.add_documents(documents, ids=point_ids)
            return ids or point_ids
        return self.store.add_documents(documents, ids=ids)

    def _add_to_faiss(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to FAISS, overwriting documents that reuse an ID."""
        from langchain_community.vectorstores import FAISS

        ids = list(ids) if ids else [str(uuid.uuid4()) for _ in documents]
        if self.store is None:
            self._store = FAISS.from_documents(documents, self.embeddings.embeddings, ids=ids)
        else:
            existing = set(self._store.index_to_docstore_id.values())
            replaced = [doc_id for doc_id in ids if doc_id in existing]
            if replaced:
                self._store.delete(replaced)
            self._store.add_documents(documents, ids=ids)
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
            filter=self._native_filter(filter),
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
        Search for similar documents with scores.

        FAISS and Chroma return distances (lower is better), Qdrant returns
        cosine similarity (higher is better).

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
            filter=self._native_filter(filter),
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
            ids: Document IDs to delete (unknown IDs are ignored)
            filter: Metadata filter for deletion
        """
        if self.config.provider == "faiss":
            store = self.store
            if store is None:
                return
            existing = set(store.index_to_docstore_id.values())
            if filter:
                ids = [
                    doc_id for doc_id in existing
                    if metadata_matches(store.docstore.search(doc_id).metadata, filter)
                ]
            ids = [doc_id for doc_id in ids or [] if doc_id in existing]
            if ids:
                store.delete(ids)
            return

        if self.config.provider == "qdrant":
            from qdrant_client import models

            if ids:
                selector = models.PointIdsList(points=[qdrant_point_id(i) for i in ids])
            elif filter:
                selector = models.FilterSelector(filter=self._native_filter(filter))
            else:
                return
            self.store.client.delete(
                collection_name=self.config.collection_name,
                points_selector=selector,
            )
            return

        if ids:
            self.store.delete(ids)
        elif filter:
            self.store.delete(filter=filter)

    def get_all_documents(self) -> List[Document]:
        """Every stored document with its metadata, e.g. to rebuild BM25 after a restart."""
        provider = self.config.provider
        if provider == "faiss":
            store = self.store
            if store is None:
                return []
            return [
                store.docstore.search(doc_id)
                for doc_id in store.index_to_docstore_id.values()
            ]

        if provider == "qdrant":
            store = self.store
            documents, offset = [], None
            while True:
                points, offset = store.client.scroll(
                    collection_name=self.config.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    documents.append(Document(
                        page_content=payload.get(store.content_payload_key, ""),
                        metadata=payload.get(store.metadata_payload_key) or {},
                    ))
                if offset is None:
                    return documents

        data = self.store.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(data["documents"], data["metadatas"])
        ]

    def count(self) -> int:
        """Number of stored documents."""
        if self.config.provider == "faiss":
            return self.store.index.ntotal if self.store is not None else 0
        if self.config.provider == "qdrant":
            return self.store.client.count(
                collection_name=self.config.collection_name, exact=True
            ).count
        return len(self.get_all_documents())

    def health(self) -> Dict[str, Any]:
        """Whether the backend answers; raises when it does not."""
        info: Dict[str, Any] = {"provider": self.config.provider}
        if self.config.provider == "qdrant":
            info["collection"] = self.config.collection_name
            info["points"] = self.count()
        return info

    def close(self) -> None:
        """Release the backend (an embedded Qdrant store locks its directory)."""
        client = getattr(self._store, "client", None)
        if client is not None and hasattr(client, "close"):
            client.close()
        self._store = None

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
        """Save the store to disk where the backend does not do it itself (FAISS)."""
        if self.config.provider == "faiss" and self._store and self.config.persist_directory:
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
            self._store.save_local(
                self.config.persist_directory,
                index_name=self.config.collection_name,
            )

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

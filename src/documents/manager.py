"""
Document Manager
================

Document management system with CRUD operations.

Features:
- Add, update, delete documents
- Metadata management
- Document versioning
- Bulk operations
- Search by metadata

Usage:
    from src.documents import DocumentManager

    manager = DocumentManager()
    doc_id = manager.add("document.pdf", metadata={"category": "legal"})
    manager.update(doc_id, "new_document.pdf")
    manager.delete(doc_id)
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

from langchain_core.documents import Document


@dataclass
class DocumentRecord:
    """Record for a managed document."""
    id: str
    source: str
    filename: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int = 1
    versions: List[Dict[str, Any]] = field(default_factory=list)
    chunk_count: int = 0
    status: str = "active"


class DocumentManager:
    """
    Document management system.

    Provides CRUD operations for documents with versioning and metadata.

    Example:
        manager = DocumentManager()

        # Add documents
        doc_id = manager.add("document.pdf", metadata={"category": "legal"})

        # List documents
        docs = manager.list(category="legal")

        # Update document
        manager.update(doc_id, "updated_document.pdf")

        # Delete document
        manager.delete(doc_id)
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize document manager.

        Args:
            storage_path: Path for storing document records
        """
        self.storage_path = storage_path
        self.documents: Dict[str, DocumentRecord] = {}

        # Load existing records if storage path exists
        if storage_path and Path(storage_path).exists():
            self._load_records()

    def add(
        self,
        source: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add document to manager.

        Args:
            source: Path to document
            metadata: Document metadata
            tags: Document tags

        Returns:
            Document ID
        """
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        doc_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        metadata = metadata or {}
        metadata["tags"] = tags or []

        record = DocumentRecord(
            id=doc_id,
            source=str(source),
            filename=source.name,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )

        self.documents[doc_id] = record

        # Save records
        if self.storage_path:
            self._save_records()

        return doc_id

    def update(
        self,
        doc_id: str,
        source: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update document.

        Args:
            doc_id: Document ID
            source: New document path
            metadata: New metadata (optional)
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        record = self.documents[doc_id]
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        # Save version history
        record.versions.append({
            "version": record.version,
            "source": record.source,
            "metadata": record.metadata.copy(),
            "updated_at": record.updated_at.isoformat(),
        })

        # Update record
        record.source = str(source)
        record.filename = source.name
        record.updated_at = datetime.now()
        record.version += 1

        if metadata:
            record.metadata.update(metadata)

        # Save records
        if self.storage_path:
            self._save_records()

    def delete(self, doc_id: str, permanent: bool = False) -> None:
        """
        Delete document.

        Args:
            doc_id: Document ID
            permanent: If True, permanently delete; otherwise mark as deleted
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        if permanent:
            del self.documents[doc_id]
        else:
            self.documents[doc_id].status = "deleted"

        # Save records
        if self.storage_path:
            self._save_records()

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        """
        Get document record.

        Args:
            doc_id: Document ID

        Returns:
            Document record or None
        """
        return self.documents.get(doc_id)

    def list(
        self,
        status: str = "active",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[DocumentRecord]:
        """
        List documents.

        Args:
            status: Filter by status
            category: Filter by category
            tags: Filter by tags

        Returns:
            List of document records
        """
        results = []

        for record in self.documents.values():
            # Filter by status
            if record.status != status:
                continue

            # Filter by category
            if category and record.metadata.get("category") != category:
                continue

            # Filter by tags
            if tags:
                doc_tags = record.metadata.get("tags", [])
                if not any(tag in doc_tags for tag in tags):
                    continue

            results.append(record)

        return results

    def search(
        self,
        query: str,
        fields: Optional[List[str]] = None
    ) -> List[DocumentRecord]:
        """
        Search documents by metadata.

        Args:
            query: Search query
            fields: Fields to search in

        Returns:
            List of matching documents
        """
        fields = fields or ["filename", "category", "tags"]
        results = []
        query_lower = query.lower()

        for record in self.documents.values():
            if record.status != "active":
                continue

            for field in fields:
                value = record.metadata.get(field, "")
                if isinstance(value, list):
                    value = " ".join(str(v) for v in value)
                else:
                    value = str(value)

                if query_lower in value.lower():
                    results.append(record)
                    break

        return results

    def add_tags(self, doc_id: str, tags: List[str]) -> None:
        """
        Add tags to document.

        Args:
            doc_id: Document ID
            tags: Tags to add
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        record = self.documents[doc_id]
        existing_tags = record.metadata.get("tags", [])
        record.metadata["tags"] = list(set(existing_tags + tags))

        if self.storage_path:
            self._save_records()

    def remove_tags(self, doc_id: str, tags: List[str]) -> None:
        """
        Remove tags from document.

        Args:
            doc_id: Document ID
            tags: Tags to remove
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        record = self.documents[doc_id]
        existing_tags = record.metadata.get("tags", [])
        record.metadata["tags"] = [t for t in existing_tags if t not in tags]

        if self.storage_path:
            self._save_records()

    def get_versions(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for document.

        Args:
            doc_id: Document ID

        Returns:
            List of version records
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        record = self.documents[doc_id]
        versions = record.versions.copy()

        # Add current version
        versions.append({
            "version": record.version,
            "source": record.source,
            "metadata": record.metadata.copy(),
            "updated_at": record.updated_at.isoformat(),
        })

        return versions

    def bulk_add(
        self,
        sources: List[Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Add multiple documents.

        Args:
            sources: List of document paths
            metadata: Common metadata for all documents

        Returns:
            List of document IDs
        """
        doc_ids = []

        for source in sources:
            try:
                doc_id = self.add(source, metadata)
                doc_ids.append(doc_id)
            except Exception as e:
                print(f"Error adding {source}: {e}")

        return doc_ids

    def bulk_delete(self, doc_ids: List[str]) -> None:
        """
        Delete multiple documents.

        Args:
            doc_ids: List of document IDs
        """
        for doc_id in doc_ids:
            try:
                self.delete(doc_id)
            except Exception as e:
                print(f"Error deleting {doc_id}: {e}")

    def export_records(self, path: str) -> None:
        """
        Export document records to JSON.

        Args:
            path: Export file path
        """
        records = {}
        for doc_id, record in self.documents.items():
            records[doc_id] = {
                "id": record.id,
                "source": record.source,
                "filename": record.filename,
                "metadata": record.metadata,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
                "version": record.version,
                "chunk_count": record.chunk_count,
                "status": record.status,
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def import_records(self, path: str) -> None:
        """
        Import document records from JSON.

        Args:
            path: Import file path
        """
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for doc_id, data in records.items():
            self.documents[doc_id] = DocumentRecord(
                id=data["id"],
                source=data["source"],
                filename=data["filename"],
                metadata=data["metadata"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                version=data.get("version", 1),
                chunk_count=data.get("chunk_count", 0),
                status=data.get("status", "active"),
            )

    def _save_records(self) -> None:
        """Save records to storage."""
        if not self.storage_path:
            return

        self.export_records(self.storage_path)

    def _load_records(self) -> None:
        """Load records from storage."""
        if not self.storage_path:
            return

        self.import_records(self.storage_path)

    @property
    def count(self) -> int:
        """Number of active documents."""
        return len([r for r in self.documents.values() if r.status == "active"])

    def __len__(self) -> int:
        return self.count

"""
Document Management
===================

Document management system with CRUD operations.

Features:
- Add, update, delete documents
- Metadata management
- Document versioning
- Bulk operations

Usage:
    from src.documents import DocumentManager

    manager = DocumentManager()
    doc_id = manager.add("document.pdf")
    manager.update(doc_id, "new_document.pdf")
    manager.delete(doc_id)
"""

from src.documents.manager import DocumentManager

__all__ = ["DocumentManager"]

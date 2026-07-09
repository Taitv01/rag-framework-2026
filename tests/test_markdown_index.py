"""Tests for Markdown folder indexing."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.core.document_loader import DocumentLoader
from src.core.markdown_index import MarkdownFolderIndexer
from src.core.text_splitter import TextSplitter
from src.rag.naive_rag import NaiveRAG


def test_markdown_manifest_detects_added_updated_removed(tmp_path):
    """Manifest comparison tracks Markdown file lifecycle by content hash."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    source = docs_dir / "metrics.md"
    source.write_text("# Metrics\n\nRevenue: 100", encoding="utf-8")

    indexer = MarkdownFolderIndexer()
    result, current = indexer.compare(docs_dir, manifest_path)
    assert result.added == ["metrics.md"]
    assert result.changed is True

    indexer.save_manifest(manifest_path, current)
    result, _ = indexer.compare(docs_dir, manifest_path)
    assert result.changed is False
    assert result.unchanged == ["metrics.md"]

    source.write_text("# Metrics\n\nRevenue: 200", encoding="utf-8")
    result, current = indexer.compare(docs_dir, manifest_path)
    assert result.updated == ["metrics.md"]

    indexer.save_manifest(manifest_path, current)
    source.unlink()
    result, _ = indexer.compare(docs_dir, manifest_path)
    assert result.removed == ["metrics.md"]


def test_naive_rag_refresh_markdown_directory_replaces_existing_chunks(tmp_path):
    """Refreshing a Markdown folder replaces old chunks instead of appending."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source = docs_dir / "metrics.md"
    source.write_text("# Metrics\n\nRevenue: 100", encoding="utf-8")

    rag = NaiveRAG.__new__(NaiveRAG)
    rag.document_loader = DocumentLoader()
    rag.text_splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
    rag.embeddings = Mock()
    rag.vector_store = Mock()
    rag.vector_store.config = SimpleNamespace(
        provider="faiss",
        collection_name="default",
        persist_directory=None,
        url=None,
        api_key=None,
    )
    rag._documents = []
    rag._chunks = []

    rebuilt_store = Mock()
    rebuilt_store.config = rag.vector_store.config

    with patch("src.rag.naive_rag.VectorStoreManager", return_value=rebuilt_store):
        first = rag.refresh_markdown_directory(docs_dir)
        assert first["added"] == ["metrics.md"]
        assert rag.num_documents == 1
        assert "Revenue: 100" in rag._documents[0].page_content

        source.write_text("# Metrics\n\nRevenue: 200", encoding="utf-8")
        second = rag.refresh_markdown_directory(docs_dir)
        assert second["updated"] == ["metrics.md"]
        assert rag.num_documents == 1
        assert "Revenue: 200" in rag._documents[0].page_content
        assert all("chunk_id" in chunk.metadata for chunk in rag._chunks)

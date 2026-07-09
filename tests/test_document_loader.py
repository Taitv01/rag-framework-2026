"""
Tests for Document Loader
=========================
"""

import pytest
import tempfile
from pathlib import Path

from src.core.document_loader import DocumentLoader


class TestDocumentLoader:
    """Test DocumentLoader class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.loader = DocumentLoader()

    def test_load_text_file(self):
        """Test loading a text file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write("This is a test document.")
            temp_path = f.name

        try:
            docs = self.loader.load(temp_path)
            assert len(docs) == 1
            assert docs[0].page_content == "This is a test document."
            assert "source" in docs[0].metadata
        finally:
            Path(temp_path).unlink()

    def test_load_markdown_file(self):
        """Test loading a markdown file."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.md',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write("# Test Heading\n\nThis is markdown content.")
            temp_path = f.name

        try:
            docs = self.loader.load(temp_path)
            assert len(docs) == 1
            assert "Test Heading" in docs[0].page_content
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.loader.load("nonexistent_file.txt")

    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                self.loader.load(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_with_metadata(self):
        """Test loading with custom metadata."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write("Test content.")
            temp_path = f.name

        try:
            custom_metadata = {"category": "test", "author": "pytest"}
            docs = self.loader.load(temp_path, metadata=custom_metadata)

            assert docs[0].metadata["category"] == "test"
            assert docs[0].metadata["author"] == "pytest"
        finally:
            Path(temp_path).unlink()

    def test_load_directory(self):
        """Test loading a directory of files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            for i in range(3):
                file_path = Path(temp_dir) / f"test_{i}.txt"
                file_path.write_text(f"Content of file {i}")

            # Load directory
            docs = self.loader.load_directory(temp_dir)

            assert len(docs) == 3

    def test_load_csv_file(self):
        """Test loading a CSV file."""
        import csv

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
            newline=''
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["name", "age", "city"])
            writer.writerow(["Alice", "30", "New York"])
            writer.writerow(["Bob", "25", "San Francisco"])
            temp_path = f.name

        try:
            docs = self.loader.load(temp_path)
            assert len(docs) == 2
            assert "Alice" in docs[0].page_content
        finally:
            Path(temp_path).unlink()

    def test_load_json_file(self):
        """Test loading a JSON file."""
        import json

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            json.dump({"key": "value", "number": 42}, f)
            temp_path = f.name

        try:
            docs = self.loader.load(temp_path)
            assert len(docs) == 1
            assert "key" in docs[0].page_content
        finally:
            Path(temp_path).unlink()

    def test_load_markdown_directory_only_markdown(self):
        """Test loading only Markdown files from a directory tree."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.txt").write_text("ignore me", encoding="utf-8")
            (root / "a.md").write_text("# A\n\nRevenue: 100", encoding="utf-8")
            subdir = root / "sub"
            subdir.mkdir()
            (subdir / "c.markdown").write_text("# C\n\nCost: 50", encoding="utf-8")

            docs = self.loader.load_markdown_directory(root)

            assert [doc.metadata["relative_source"] for doc in docs] == [
                "a.md",
                "sub/c.markdown",
            ]
            assert all(doc.metadata["file_type"] == "markdown" for doc in docs)
            assert all("source_root" in doc.metadata for doc in docs)
            assert all("source_sha256" in doc.metadata for doc in docs)

    def test_markdown_source_hash_changes_after_update(self):
        """Test source hash metadata changes when Markdown content changes."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("# Metrics\n\nRevenue: 100")
            temp_path = f.name

        try:
            first_hash = self.loader.load(temp_path)[0].metadata["source_sha256"]
            Path(temp_path).write_text("# Metrics\n\nRevenue: 200", encoding="utf-8")
            second_hash = self.loader.load(temp_path)[0].metadata["source_sha256"]

            assert first_hash != second_hash
        finally:
            Path(temp_path).unlink()

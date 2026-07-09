"""
Markdown Folder Indexing
========================

Utilities for tracking Markdown knowledge folders by content hash.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document


MARKDOWN_EXTENSIONS = {".md", ".markdown"}


@dataclass
class MarkdownRefreshResult:
    """Result of comparing a Markdown folder with its previous manifest."""

    directory: str
    manifest_path: str
    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    documents_loaded: int = 0
    chunks_indexed: int = 0
    rebuilt: bool = False

    @property
    def changed(self) -> bool:
        """Whether the folder content changed since the previous manifest."""
        return bool(self.added or self.updated or self.removed)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable result."""
        return {
            "directory": self.directory,
            "manifest_path": self.manifest_path,
            "added": self.added,
            "updated": self.updated,
            "removed": self.removed,
            "unchanged": self.unchanged,
            "changed": self.changed,
            "documents_loaded": self.documents_loaded,
            "chunks_indexed": self.chunks_indexed,
            "rebuilt": self.rebuilt,
        }


class MarkdownFolderIndexer:
    """
    Track Markdown folder state with a manifest and stable chunk IDs.

    The manifest lets a RAG pipeline detect changed source files before
    rebuilding an index, which avoids mixing old numeric facts with updated
    Markdown content.
    """

    def __init__(
        self,
        manifest_name: str = ".rag_markdown_manifest.json",
        extensions: Optional[Iterable[str]] = None,
    ):
        self.manifest_name = manifest_name
        self.extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (extensions or MARKDOWN_EXTENSIONS)
        }

    def manifest_path(
        self,
        directory: Path,
        manifest_path: Optional[Path] = None,
    ) -> Path:
        """Resolve the manifest path for a Markdown folder."""
        if manifest_path:
            return Path(manifest_path).resolve()
        return (Path(directory).resolve() / self.manifest_name).resolve()

    def scan(self, directory: Path) -> Dict[str, Dict[str, Any]]:
        """Scan Markdown files and return relative-path keyed fingerprints."""
        directory = Path(directory).resolve()
        fingerprints: Dict[str, Dict[str, Any]] = {}
        excluded_dirs = {".git", ".venv", "__pycache__", ".pytest_cache"}

        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.extensions:
                continue
            if any(part in excluded_dirs for part in file_path.parts):
                continue

            rel_path = file_path.relative_to(directory).as_posix()
            stat = file_path.stat()
            fingerprints[rel_path] = {
                "sha256": self._sha256_file(file_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }

        return fingerprints

    def compare(
        self,
        directory: Path,
        manifest_path: Optional[Path] = None,
    ) -> tuple[MarkdownRefreshResult, Dict[str, Dict[str, Any]]]:
        """Compare the current folder scan with the saved manifest."""
        directory = Path(directory).resolve()
        resolved_manifest = self.manifest_path(directory, manifest_path)
        previous = self.load_manifest(resolved_manifest)
        current = self.scan(directory)

        previous_files = previous.get("files", {})
        current_names = set(current)
        previous_names = set(previous_files)

        added = sorted(current_names - previous_names)
        removed = sorted(previous_names - current_names)
        updated = sorted(
            name
            for name in current_names & previous_names
            if current[name].get("sha256") != previous_files[name].get("sha256")
        )
        unchanged = sorted((current_names & previous_names) - set(updated))

        result = MarkdownRefreshResult(
            directory=str(directory),
            manifest_path=str(resolved_manifest),
            added=added,
            updated=updated,
            removed=removed,
            unchanged=unchanged,
        )
        return result, current

    def load_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        """Load a manifest, returning an empty state when absent."""
        if not manifest_path.exists():
            return {"version": 1, "files": {}}

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {"version": 1, "files": {}}
        data.setdefault("version", 1)
        data.setdefault("files", {})
        return data

    def save_manifest(
        self,
        manifest_path: Path,
        files: Dict[str, Dict[str, Any]],
    ) -> None:
        """Persist a manifest for the current Markdown folder state."""
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "files": files,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")

    def assign_stable_chunk_ids(self, chunks: List[Document]) -> List[str]:
        """Attach deterministic document_id and chunk_id metadata to chunks."""
        per_source_index: Dict[str, int] = {}
        chunk_ids = []

        for chunk in chunks:
            metadata = chunk.metadata or {}
            source_key = (
                metadata.get("relative_source")
                or metadata.get("source")
                or metadata.get("file_name")
                or "unknown"
            )
            source_key = str(source_key)
            chunk_index = per_source_index.get(source_key, 0)
            per_source_index[source_key] = chunk_index + 1

            source_hash = str(metadata.get("source_sha256") or "")
            document_id = self._hash_text(f"{source_key}:{source_hash}")
            content_hash = self._hash_text(chunk.page_content)
            start_index = metadata.get("start_index", "")
            chunk_id = self._hash_text(
                f"{source_key}:{source_hash}:{start_index}:{chunk_index}:{content_hash}"
            )

            metadata["document_id"] = document_id
            metadata["chunk_id"] = chunk_id
            metadata["chunk_index"] = chunk_index
            metadata["chunk_sha256"] = content_hash
            chunk.metadata = metadata
            chunk_ids.append(chunk_id)

        return chunk_ids

    def _sha256_file(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _hash_text(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

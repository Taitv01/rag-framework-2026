"""Vietnamese fairy tale dataset builder."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from langchain_core.documents import Document

from src.rag.cross_story_rag import CrossStoryRAG, stable_story_id

StoryInput = Union[Mapping[str, Any], Tuple[str, str]]


@dataclass
class FairyTaleRecord:
    """Normalized fairy tale dataset record."""

    story_id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    characters: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    motifs: List[str] = field(default_factory=list)
    morals: List[str] = field(default_factory=list)
    language: str = "unknown"
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "text": self.text,
            "metadata": dict(self.metadata),
            "characters": list(self.characters),
            "locations": list(self.locations),
            "motifs": list(self.motifs),
            "morals": list(self.morals),
            "language": self.language,
            "word_count": self.word_count,
        }

    def to_document(self) -> Document:
        metadata = dict(self.metadata)
        metadata.update({
            "story_id": self.story_id,
            "title": self.title,
            "characters": list(self.characters),
            "locations": list(self.locations),
            "motifs": list(self.motifs),
            "morals": list(self.morals),
            "language": self.language,
            "word_count": self.word_count,
        })
        return Document(page_content=self.text, metadata=metadata)


class FairyTaleDatasetBuilder:
    """Build labeled Vietnamese fairy tale datasets from text or directories."""

    def __init__(self, vietnamese_processor: Optional[Any] = None):
        self._processor = vietnamese_processor

    def build_record(
        self,
        title: str,
        text: str,
        metadata: Optional[Mapping[str, Any]] = None,
        story_id: Optional[str] = None,
    ) -> FairyTaleRecord:
        metadata_dict = dict(metadata or {})
        normalized_text = self.normalize_text(text)
        story_id = story_id or metadata_dict.get("story_id") or stable_story_id(title, normalized_text)
        analyzer = CrossStoryRAG(vietnamese_processor=self._processor)
        story = analyzer.add_story(title=title, text=normalized_text, metadata=metadata_dict, story_id=story_id)
        return FairyTaleRecord(
            story_id=story.story_id,
            title=story.title,
            text=story.text,
            metadata=metadata_dict,
            characters=story.characters,
            locations=story.locations,
            motifs=story.motifs,
            morals=story.morals,
            language=self._detect_language(story.text),
            word_count=len(re.findall(r"\S+", story.text)),
        )

    def build_from_texts(self, stories: Iterable[StoryInput]) -> List[FairyTaleRecord]:
        records = []
        for item in stories:
            if isinstance(item, Mapping):
                records.append(self.build_record(
                    title=str(item.get("title", "")),
                    text=str(item.get("text", "")),
                    metadata=item.get("metadata") or {},
                    story_id=item.get("story_id"),
                ))
            else:
                title, text = item
                records.append(self.build_record(title=str(title), text=str(text)))
        return records

    def build_from_directory(
        self,
        directory: Union[str, Path],
        patterns: Sequence[str] = ("*.txt", "*.md"),
        encoding: str = "utf-8",
    ) -> List[FairyTaleRecord]:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(directory.rglob(pattern))
        records = []
        for path in sorted(set(paths)):
            text = path.read_text(encoding=encoding, errors="replace")
            title = path.stem.replace("_", " ").replace("-", " ").strip() or path.name
            records.append(self.build_record(title=title, text=text, metadata={"source": str(path), "filename": path.name}))
        return records

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def to_documents(self, records: Iterable[FairyTaleRecord]) -> List[Document]:
        return [record.to_document() for record in records]

    def export_jsonl(self, records: Iterable[FairyTaleRecord], path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return path

    def export_json(self, records: Iterable[FairyTaleRecord], path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [record.to_dict() for record in records]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _detect_language(self, text: str) -> str:
        if self._processor is not None:
            try:
                return self._processor.detect_language(text)
            except Exception:
                pass
        lowered = text.lower()
        vi_diacritics = "\u00e0\u00e1\u1ea3\u00e3\u1ea1\u0103\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7\u00e2\u1ea5\u1ea7\u1ea9\u1eab\u1ead\u0111\u00e8\u00e9\u1ebb\u1ebd\u1eb9\u00ea\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7\u00f2\u00f3\u1ecf\u00f5\u1ecd\u00f4\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9\u01a1\u1edb\u1edd\u1edf\u1ee1\u1ee3\u00f9\u00fa\u1ee7\u0169\u1ee5\u01b0\u1ee9\u1eeb\u1eed\u1eef\u1ef1"
        if any(ch in lowered for ch in vi_diacritics):
            return "vi"
        if any(term in lowered for term in ("thach", "sanh", "tam", "cam", "vua", "cong chua")):
            return "vi"
        return "unknown"

"""
Chapter Manager
===============

Chapter management system for story writing.

Usage:
    from src.story import ChapterManager, Chapter

    manager = ChapterManager()
    manager.add_chapter(Chapter(number=1, title="Khởi đầu", content="..."))
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import json


@dataclass
class Chapter:
    """Chapter in the story."""
    number: int
    title: str
    content: str = ""
    summary: str = ""
    characters_present: List[str] = field(default_factory=list)
    location: Optional[str] = None
    time_period: Optional[str] = None
    word_count: int = 0
    status: str = "draft"  # "draft", "revised", "final"
    notes: str = ""


class ChapterManager:
    """
    Chapter management system.

    Example:
        manager = ChapterManager()
        manager.add_chapter(Chapter(number=1, title="Khởi đầu"))
        chapter = manager.get_chapter(1)
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.chapters: Dict[int, Chapter] = {}

        if storage_path:
            self._load_data()

    def add_chapter(self, chapter: Chapter) -> None:
        """Add chapter."""
        chapter.word_count = len(chapter.content.split())
        self.chapters[chapter.number] = chapter
        self._save_data()

    def get_chapter(self, number: int) -> Optional[Chapter]:
        """Get chapter by number."""
        return self.chapters.get(number)

    def update_chapter(self, number: int, **kwargs) -> None:
        """Update chapter."""
        chapter = self.chapters.get(number)
        if not chapter:
            raise ValueError(f"Chapter not found: {number}")

        for key, value in kwargs.items():
            if hasattr(chapter, key):
                setattr(chapter, key, value)

        chapter.word_count = len(chapter.content.split())
        self._save_data()

    def get_chapters_summary(self) -> str:
        """Get summary of all chapters."""
        parts = []
        for number in sorted(self.chapters.keys()):
            chapter = self.chapters[number]
            parts.append(f"Chương {number}: {chapter.title} ({chapter.word_count} từ)")
        return "\n".join(parts)

    def get_total_word_count(self) -> int:
        """Get total word count."""
        return sum(ch.word_count for ch in self.chapters.values())

    def _save_data(self) -> None:
        if not self.storage_path:
            return

        data = {
            str(num): {
                "number": ch.number,
                "title": ch.title,
                "content": ch.content,
                "summary": ch.summary,
                "characters_present": ch.characters_present,
                "location": ch.location,
                "word_count": ch.word_count,
                "status": ch.status,
            }
            for num, ch in self.chapters.items()
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_data(self) -> None:
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for num, ch_data in data.items():
                self.chapters[int(num)] = Chapter(
                    number=ch_data["number"],
                    title=ch_data["title"],
                    content=ch_data.get("content", ""),
                    summary=ch_data.get("summary", ""),
                    characters_present=ch_data.get("characters_present", []),
                    location=ch_data.get("location"),
                    word_count=ch_data.get("word_count", 0),
                    status=ch_data.get("status", "draft"),
                )
        except FileNotFoundError:
            pass

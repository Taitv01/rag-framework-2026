"""
Timeline Manager
================

Timeline management for story writing.

Usage:
    from src.story import TimelineManager, TimelineEvent

    manager = TimelineManager()
    manager.add_event(TimelineEvent(
        chapter=1,
        time="Sáng sớm",
        event="A thức dậy",
        location="Nhà A"
    ))
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import json


@dataclass
class TimelineEvent:
    """Event in the story timeline."""
    chapter: int
    time: str  # "Sáng sớm", "Buổi trưa", etc.
    event: str
    location: Optional[str] = None
    characters_involved: List[str] = field(default_factory=list)
    duration: Optional[str] = None
    importance: str = "medium"


class TimelineManager:
    """
    Timeline management system.

    Example:
        manager = TimelineManager()
        manager.add_event(TimelineEvent(
            chapter=1,
            time="Sáng sớm",
            event="A thức dậy",
            location="Nhà A"
        ))
        timeline = manager.get_timeline()
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.events: List[TimelineEvent] = []

        if storage_path:
            self._load_data()

    def add_event(self, event: TimelineEvent) -> None:
        """Add event to timeline."""
        self.events.append(event)
        self._save_data()

    def get_chapter_events(self, chapter: int) -> List[TimelineEvent]:
        """Get events for chapter."""
        return [e for e in self.events if e.chapter == chapter]

    def get_timeline(self) -> str:
        """Get formatted timeline."""
        parts = ["=== TIMELINE ===\n"]

        chapters = sorted(set(e.chapter for e in self.events))
        for chapter in chapters:
            events = self.get_chapter_events(chapter)
            parts.append(f"Chương {chapter}:")
            for event in events:
                parts.append(f"  {event.time}: {event.event}")
                if event.location:
                    parts.append(f"    Địa điểm: {event.location}")
                if event.characters_involved:
                    parts.append(f"    Nhân vật: {', '.join(event.characters_involved)}")
            parts.append("")

        return "\n".join(parts)

    def _save_data(self) -> None:
        if not self.storage_path:
            return

        data = [
            {
                "chapter": e.chapter,
                "time": e.time,
                "event": e.event,
                "location": e.location,
                "characters_involved": e.characters_involved,
            }
            for e in self.events
        ]

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_data(self) -> None:
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for e in data:
                self.events.append(TimelineEvent(
                    chapter=e["chapter"],
                    time=e["time"],
                    event=e["event"],
                    location=e.get("location"),
                    characters_involved=e.get("characters_involved", []),
                ))
        except FileNotFoundError:
            pass

"""
Story Writing System
====================

Comprehensive story writing system with RAG support.

Features:
- Character management
- Plot management
- World building
- Consistency checking
- Style analysis
- Chapter management
- Timeline management
- Writing assistant

Usage:
    from src.story import CharacterManager, PlotManager, WorldBuilder
    from src.story import ConsistencyChecker, WritingAssistant
"""

from src.story.character_manager import CharacterManager, Character
from src.story.plot_manager import PlotManager, PlotPoint, PlotArc
from src.story.world_builder import WorldBuilder, Location, Lore
from src.story.consistency_checker import ConsistencyChecker
from src.story.chapter_manager import ChapterManager, Chapter
from src.story.timeline_manager import TimelineManager, TimelineEvent
from src.story.writing_assistant import WritingAssistant

__all__ = [
    "CharacterManager",
    "Character",
    "PlotManager",
    "PlotPoint",
    "PlotArc",
    "WorldBuilder",
    "Location",
    "Lore",
    "ConsistencyChecker",
    "ChapterManager",
    "Chapter",
    "TimelineManager",
    "TimelineEvent",
    "WritingAssistant",
]

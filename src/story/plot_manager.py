"""
Plot Manager
============

Comprehensive plot management for story writing.

Features:
- Plot points tracking
- Plot arcs management
- Foreshadowing tracking
- Subplot management
- Plot consistency checking

Usage:
    from src.story import PlotManager, PlotPoint, PlotArc

    manager = PlotManager()

    # Add plot point
    point = PlotPoint(
        chapter=1,
        event="A tìm thấy bức thư",
        importance="high"
    )
    manager.add_plot_point(point)

    # Track foreshadowing
    manager.add_foreshadowing(
        chapter=1,
        hint="Bức thư có mùi lạ",
        resolution_chapter=10,
        resolution="Bức thư bị ngấm thuốc độc"
    )
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class PlotPoint:
    """Single plot point."""
    chapter: int
    event: str
    importance: str = "medium"  # "low", "medium", "high", "critical"
    characters_involved: List[str] = field(default_factory=list)
    location: Optional[str] = None
    consequences: List[str] = field(default_factory=list)
    description: str = ""
    is_resolved: bool = False


@dataclass
class PlotArc:
    """Plot arc spanning multiple chapters."""
    name: str
    description: str
    start_chapter: int
    end_chapter: Optional[int] = None
    status: str = "active"  # "active", "resolved", "abandoned"
    plot_points: List[int] = field(default_factory=list)  # Chapter numbers
    characters_involved: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)


@dataclass
class Foreshadowing:
    """Foreshadowing element."""
    id: str
    chapter_planted: int
    hint: str
    resolution_chapter: Optional[int] = None
    resolution: Optional[str] = None
    is_resolved: bool = False
    importance: str = "medium"


@dataclass
class Subplot:
    """Subplot within the main story."""
    name: str
    description: str
    main_characters: List[str] = field(default_factory=list)
    start_chapter: int = 1
    end_chapter: Optional[int] = None
    status: str = "active"
    relationship_to_main: str = ""


class PlotManager:
    """
    Plot management system for story writing.

    Manages plot points, arcs, foreshadowing, and subplots.

    Example:
        manager = PlotManager()

        # Add plot points
        manager.add_plot_point(PlotPoint(
            chapter=1,
            event="A tìm thấy bức thư",
            importance="high"
        ))

        # Create plot arc
        manager.create_plot_arc(PlotArc(
            name="Bí mật gia đình",
            description="A khám phá bí mật về gia đình",
            start_chapter=1,
            characters_involved=["A", "B"]
        ))

        # Track foreshadowing
        manager.add_foreshadowing(
            chapter=1,
            hint="Bức thư có mùi lạ",
            resolution_chapter=10
        )

        # Get plot context
        context = manager.get_plot_context()
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize plot manager.

        Args:
            storage_path: Path for storing plot data
        """
        self.storage_path = storage_path
        self.plot_points: List[PlotPoint] = []
        self.plot_arcs: Dict[str, PlotArc] = {}
        self.foreshadowing: Dict[str, Foreshadowing] = {}
        self.subplots: Dict[str, Subplot] = {}

        # Load existing data
        if storage_path:
            self._load_data()

    def add_plot_point(self, point: PlotPoint) -> None:
        """
        Add plot point.

        Args:
            point: Plot point to add
        """
        self.plot_points.append(point)
        self._save_data()

    def create_plot_arc(self, arc: PlotArc) -> None:
        """
        Create plot arc.

        Args:
            arc: Plot arc to create
        """
        self.plot_arcs[arc.name] = arc
        self._save_data()

    def add_foreshadowing(
        self,
        chapter: int,
        hint: str,
        resolution_chapter: Optional[int] = None,
        importance: str = "medium"
    ) -> str:
        """
        Add foreshadowing element.

        Args:
            chapter: Chapter where hint is planted
            hint: The hint or clue
            resolution_chapter: Chapter where it's resolved
            importance: Importance level

        Returns:
            Foreshadowing ID
        """
        foreshadow_id = f"fs_{len(self.foreshadowing) + 1}"

        self.foreshadowing[foreshadow_id] = Foreshadowing(
            id=foreshadow_id,
            chapter_planted=chapter,
            hint=hint,
            resolution_chapter=resolution_chapter,
            importance=importance
        )

        self._save_data()
        return foreshadow_id

    def resolve_foreshadowing(
        self,
        foreshadow_id: str,
        resolution: str,
        resolution_chapter: int
    ) -> None:
        """
        Resolve foreshadowing element.

        Args:
            foreshadow_id: Foreshadowing ID
            resolution: Resolution description
            resolution_chapter: Chapter where resolved
        """
        if foreshadow_id not in self.foreshadowing:
            raise ValueError(f"Foreshadowing not found: {foreshadow_id}")

        fs = self.foreshadowing[foreshadow_id]
        fs.resolution = resolution
        fs.resolution_chapter = resolution_chapter
        fs.is_resolved = True

        self._save_data()

    def create_subplot(self, subplot: Subplot) -> None:
        """
        Create subplot.

        Args:
            subplot: Subplot to create
        """
        self.subplots[subplot.name] = subplot
        self._save_data()

    def get_chapter_plot_points(self, chapter: int) -> List[PlotPoint]:
        """
        Get plot points for chapter.

        Args:
            chapter: Chapter number

        Returns:
            List of plot points
        """
        return [p for p in self.plot_points if p.chapter == chapter]

    def get_unresolved_foreshadowing(self) -> List[Foreshadowing]:
        """
        Get all unresolved foreshadowing.

        Returns:
            List of unresolved foreshadowing
        """
        return [fs for fs in self.foreshadowing.values() if not fs.is_resolved]

    def get_active_arcs(self) -> List[PlotArc]:
        """
        Get all active plot arcs.

        Returns:
            List of active plot arcs
        """
        return [arc for arc in self.plot_arcs.values() if arc.status == "active"]

    def get_plot_context(self) -> str:
        """
        Get full plot context for RAG.

        Returns:
            Formatted plot context
        """
        parts = []

        # Plot points by chapter
        chapters = sorted(set(p.chapter for p in self.plot_points))
        for chapter in chapters:
            points = self.get_chapter_plot_points(chapter)
            parts.append(f"Chương {chapter}:")
            for p in points:
                parts.append(f"  - {p.event} ({p.importance})")

        # Active arcs
        active_arcs = self.get_active_arcs()
        if active_arcs:
            parts.append("\nPlot Arcs đang diễn ra:")
            for arc in active_arcs:
                parts.append(f"  - {arc.name}: {arc.description}")

        # Unresolved foreshadowing
        unresolved = self.get_unresolved_foreshadowing()
        if unresolved:
            parts.append("\nForeshadowing chưa giải quyết:")
            for fs in unresolved:
                parts.append(f"  - Chương {fs.chapter_planted}: {fs.hint}")

        # Active subplots
        active_subplots = [s for s in self.subplots.values() if s.status == "active"]
        if active_subplots:
            parts.append("\nSubplot đang diễn ra:")
            for subplot in active_subplots:
                parts.append(f"  - {subplot.name}: {subplot.description}")

        return "\n".join(parts)

    def get_plot_summary(self) -> str:
        """
        Get plot summary.

        Returns:
            Plot summary text
        """
        parts = [
            f"Tổng số plot points: {len(self.plot_points)}",
            f"Tổng số plot arcs: {len(self.plot_arcs)}",
            f"Foreshadowing chưa giải quyết: {len(self.get_unresolved_foreshadowing())}",
            f"Subplot đang diễn ra: {len([s for s in self.subplots.values() if s.status == 'active'])}",
        ]
        return "\n".join(parts)

    def _save_data(self) -> None:
        """Save plot data to storage."""
        if not self.storage_path:
            return

        data = {
            "plot_points": [
                {
                    "chapter": p.chapter,
                    "event": p.event,
                    "importance": p.importance,
                    "characters_involved": p.characters_involved,
                    "location": p.location,
                    "description": p.description,
                    "is_resolved": p.is_resolved,
                }
                for p in self.plot_points
            ],
            "plot_arcs": {
                name: {
                    "name": arc.name,
                    "description": arc.description,
                    "start_chapter": arc.start_chapter,
                    "end_chapter": arc.end_chapter,
                    "status": arc.status,
                    "characters_involved": arc.characters_involved,
                    "themes": arc.themes,
                }
                for name, arc in self.plot_arcs.items()
            },
            "foreshadowing": {
                fid: {
                    "id": fs.id,
                    "chapter_planted": fs.chapter_planted,
                    "hint": fs.hint,
                    "resolution_chapter": fs.resolution_chapter,
                    "resolution": fs.resolution,
                    "is_resolved": fs.is_resolved,
                    "importance": fs.importance,
                }
                for fid, fs in self.foreshadowing.items()
            },
            "subplots": {
                name: {
                    "name": subplot.name,
                    "description": subplot.description,
                    "main_characters": subplot.main_characters,
                    "start_chapter": subplot.start_chapter,
                    "end_chapter": subplot.end_chapter,
                    "status": subplot.status,
                    "relationship_to_main": subplot.relationship_to_main,
                }
                for name, subplot in self.subplots.items()
            },
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_data(self) -> None:
        """Load plot data from storage."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load plot points
            for p in data.get("plot_points", []):
                self.plot_points.append(PlotPoint(
                    chapter=p["chapter"],
                    event=p["event"],
                    importance=p.get("importance", "medium"),
                    characters_involved=p.get("characters_involved", []),
                    location=p.get("location"),
                    description=p.get("description", ""),
                    is_resolved=p.get("is_resolved", False),
                ))

            # Load plot arcs
            for name, arc_data in data.get("plot_arcs", {}).items():
                self.plot_arcs[name] = PlotArc(
                    name=arc_data["name"],
                    description=arc_data["description"],
                    start_chapter=arc_data["start_chapter"],
                    end_chapter=arc_data.get("end_chapter"),
                    status=arc_data.get("status", "active"),
                    characters_involved=arc_data.get("characters_involved", []),
                    themes=arc_data.get("themes", []),
                )

            # Load foreshadowing
            for fid, fs_data in data.get("foreshadowing", {}).items():
                self.foreshadowing[fid] = Foreshadowing(
                    id=fs_data["id"],
                    chapter_planted=fs_data["chapter_planted"],
                    hint=fs_data["hint"],
                    resolution_chapter=fs_data.get("resolution_chapter"),
                    resolution=fs_data.get("resolution"),
                    is_resolved=fs_data.get("is_resolved", False),
                    importance=fs_data.get("importance", "medium"),
                )

            # Load subplots
            for name, subplot_data in data.get("subplots", {}).items():
                self.subplots[name] = Subplot(
                    name=subplot_data["name"],
                    description=subplot_data["description"],
                    main_characters=subplot_data.get("main_characters", []),
                    start_chapter=subplot_data.get("start_chapter", 1),
                    end_chapter=subplot_data.get("end_chapter"),
                    status=subplot_data.get("status", "active"),
                    relationship_to_main=subplot_data.get("relationship_to_main", ""),
                )

        except FileNotFoundError:
            pass

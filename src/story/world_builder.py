"""
World Builder
=============

World building system for story writing.

Features:
- Location management
- Lore and history tracking
- World rules management
- Cultural elements
- Geography tracking

Usage:
    from src.story import WorldBuilder, Location, Lore

    builder = WorldBuilder()

    # Add location
    location = Location(
        name="Đà Lạt",
        description="Thành phố mù sương",
        climate="Mát mẻ quanh năm"
    )
    builder.add_location(location)

    # Add lore
    lore = Lore(
        name="Bí mật ngọn đồi",
        content="Người ta kể rằng...",
        chapter_revealed=5
    )
    builder.add_lore(lore)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import json


@dataclass
class Location:
    """Location in the story world."""
    name: str
    description: str
    climate: str = ""
    geography: str = ""
    population: Optional[str] = None
    culture: str = ""
    landmarks: List[str] = field(default_factory=list)
    connected_locations: List[str] = field(default_factory=list)
    significance: str = ""  # Why this location matters
    first_appearance: Optional[int] = None  # Chapter number
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Lore:
    """Lore or history element."""
    name: str
    content: str
    category: str = "general"  # "history", "legend", "rule", "belief"
    chapter_revealed: Optional[int] = None
    is_secret: bool = False
    related_locations: List[str] = field(default_factory=list)
    related_characters: List[str] = field(default_factory=list)


@dataclass
class WorldRule:
    """Rule of the world."""
    name: str
    description: str
    category: str = "physics"  # "physics", "magic", "social", "political"
    exceptions: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)


@dataclass
class CulturalElement:
    """Cultural element of the world."""
    name: str
    description: str
    category: str = "custom"  # "custom", "language", "religion", "art"
    locations: List[str] = field(default_factory=list)
    significance: str = ""


class WorldBuilder:
    """
    World building system for story writing.

    Manages locations, lore, rules, and cultural elements.

    Example:
        builder = WorldBuilder()

        # Add locations
        builder.add_location(Location(
            name="Hà Nội",
            description="Thủ đô ngàn năm văn hiến",
            climate="Nhiệt đới gió mùa"
        ))

        # Add lore
        builder.add_lore(Lore(
            name="Hồ Gươm",
            content="Truyền thuyết về thanh gươm thần...",
            category="legend"
        ))

        # Get world context
        context = builder.get_world_context()
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize world builder.

        Args:
            storage_path: Path for storing world data
        """
        self.storage_path = storage_path
        self.locations: Dict[str, Location] = {}
        self.lore: Dict[str, Lore] = {}
        self.rules: Dict[str, WorldRule] = {}
        self.cultural_elements: Dict[str, CulturalElement] = {}

        # Load existing data
        if storage_path:
            self._load_data()

    def add_location(self, location: Location) -> None:
        """
        Add location to world.

        Args:
            location: Location to add
        """
        self.locations[location.name] = location
        self._save_data()

    def add_lore(self, lore: Lore) -> None:
        """
        Add lore to world.

        Args:
            lore: Lore to add
        """
        self.lore[lore.name] = lore
        self._save_data()

    def add_rule(self, rule: WorldRule) -> None:
        """
        Add world rule.

        Args:
            rule: Rule to add
        """
        self.rules[rule.name] = rule
        self._save_data()

    def add_cultural_element(self, element: CulturalElement) -> None:
        """
        Add cultural element.

        Args:
            element: Cultural element to add
        """
        self.cultural_elements[element.name] = element
        self._save_data()

    def get_location_context(self, name: str) -> str:
        """
        Get location context for RAG.

        Args:
            name: Location name

        Returns:
            Formatted location context
        """
        location = self.locations.get(name)
        if not location:
            return f"Không tìm thấy địa điểm: {name}"

        parts = [
            f"Địa điểm: {location.name}",
            f"Mô tả: {location.description}",
            f"Khí hậu: {location.climate}" if location.climate else None,
            f"Địa lý: {location.geography}" if location.geography else None,
            f"Dân số: {location.population}" if location.population else None,
            f"Văn hóa: {location.culture}" if location.culture else None,
            f"Địa danh nổi tiếng: {', '.join(location.landmarks)}" if location.landmarks else None,
            f"Kết nối với: {', '.join(location.connected_locations)}" if location.connected_locations else None,
            f"Ý nghĩa: {location.significance}" if location.significance else None,
        ]

        return "\n".join([p for p in parts if p])

    def get_world_context(self) -> str:
        """
        Get full world context for RAG.

        Returns:
            Formatted world context
        """
        parts = []

        # Locations
        if self.locations:
            parts.append("=== ĐỊA ĐIỂM ===")
            for location in self.locations.values():
                parts.append(self.get_location_context(location.name))
                parts.append("")

        # Lore
        if self.lore:
            parts.append("=== LỊCH SỬ & TRUYỀN THUYẾT ===")
            for lore in self.lore.values():
                if not lore.is_secret:
                    parts.append(f"{lore.name}: {lore.content}")
                    parts.append("")

        # Rules
        if self.rules:
            parts.append("=== QUY TẮC THẾ GIỚI ===")
            for rule in self.rules.values():
                parts.append(f"{rule.name}: {rule.description}")
                parts.append("")

        # Cultural elements
        if self.cultural_elements:
            parts.append("=== VĂN HÓA ===")
            for element in self.cultural_elements.values():
                parts.append(f"{element.name}: {element.description}")
                parts.append("")

        return "\n".join(parts)

    def get_secret_lore(self) -> List[Lore]:
        """
        Get all secret lore.

        Returns:
            List of secret lore
        """
        return [lore for lore in self.lore.values() if lore.is_secret]

    def get_locations_by_chapter(self, chapter: int) -> List[Location]:
        """
        Get locations first appearing in chapter.

        Args:
            chapter: Chapter number

        Returns:
            List of locations
        """
        return [
            loc for loc in self.locations.values()
            if loc.first_appearance == chapter
        ]

    def _save_data(self) -> None:
        """Save world data to storage."""
        if not self.storage_path:
            return

        data = {
            "locations": {
                name: {
                    "name": loc.name,
                    "description": loc.description,
                    "climate": loc.climate,
                    "geography": loc.geography,
                    "population": loc.population,
                    "culture": loc.culture,
                    "landmarks": loc.landmarks,
                    "connected_locations": loc.connected_locations,
                    "significance": loc.significance,
                    "first_appearance": loc.first_appearance,
                }
                for name, loc in self.locations.items()
            },
            "lore": {
                name: {
                    "name": lore.name,
                    "content": lore.content,
                    "category": lore.category,
                    "chapter_revealed": lore.chapter_revealed,
                    "is_secret": lore.is_secret,
                    "related_locations": lore.related_locations,
                    "related_characters": lore.related_characters,
                }
                for name, lore in self.lore.items()
            },
            "rules": {
                name: {
                    "name": rule.name,
                    "description": rule.description,
                    "category": rule.category,
                    "exceptions": rule.exceptions,
                    "consequences": rule.consequences,
                }
                for name, rule in self.rules.items()
            },
            "cultural_elements": {
                name: {
                    "name": elem.name,
                    "description": elem.description,
                    "category": elem.category,
                }
                for name, elem in self.cultural_elements.items()
            },
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_data(self) -> None:
        """Load world data from storage."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load locations
            for name, loc_data in data.get("locations", {}).items():
                self.locations[name] = Location(
                    name=loc_data["name"],
                    description=loc_data["description"],
                    climate=loc_data.get("climate", ""),
                    geography=loc_data.get("geography", ""),
                    population=loc_data.get("population"),
                    culture=loc_data.get("culture", ""),
                    landmarks=loc_data.get("landmarks", []),
                    connected_locations=loc_data.get("connected_locations", []),
                    significance=loc_data.get("significance", ""),
                    first_appearance=loc_data.get("first_appearance"),
                )

            # Load lore
            for name, lore_data in data.get("lore", {}).items():
                self.lore[name] = Lore(
                    name=lore_data["name"],
                    content=lore_data["content"],
                    category=lore_data.get("category", "general"),
                    chapter_revealed=lore_data.get("chapter_revealed"),
                    is_secret=lore_data.get("is_secret", False),
                    related_locations=lore_data.get("related_locations", []),
                    related_characters=lore_data.get("related_characters", []),
                )

            # Load rules
            for name, rule_data in data.get("rules", {}).items():
                self.rules[name] = WorldRule(
                    name=rule_data["name"],
                    description=rule_data["description"],
                    category=rule_data.get("category", "physics"),
                    exceptions=rule_data.get("exceptions", []),
                    consequences=rule_data.get("consequences", []),
                )

            # Load cultural elements
            for name, elem_data in data.get("cultural_elements", {}).items():
                self.cultural_elements[name] = CulturalElement(
                    name=elem_data["name"],
                    description=elem_data["description"],
                    category=elem_data.get("category", "custom"),
                )

        except FileNotFoundError:
            pass

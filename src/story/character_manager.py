"""
Character Manager
=================

Comprehensive character management for story writing.

Features:
- Character profiles with detailed attributes
- Relationship tracking
- Character development arcs
- Character voice consistency
- Character appearance tracking

Usage:
    from src.story import CharacterManager, Character

    manager = CharacterManager()

    # Add character
    character = Character(
        name="Nguyễn Văn A",
        age=25,
        personality="Thông minh, quyết đoán",
        backstory="Mồ côi từ nhỏ..."
    )
    manager.add_character(character)

    # Track development
    manager.add_development("A", chapter=5, event="Phát hiện bí mật")
"""

from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class CharacterTrait:
    """Single character trait."""
    name: str
    description: str
    intensity: float = 0.5  # 0.0 to 1.0
    is_positive: bool = True


@dataclass
class CharacterRelationship:
    """Relationship between characters."""
    target_character: str
    relationship_type: str  # "friend", "enemy", "lover", "family", "mentor", etc.
    description: str
    intensity: float = 0.5  # 0.0 to 1.0
    is_positive: bool = True
    history: List[str] = field(default_factory=list)


@dataclass
class CharacterDevelopment:
    """Character development event."""
    chapter: int
    event: str
    impact: str  # "positive", "negative", "neutral"
    traits_affected: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Character:
    """Complete character profile."""
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    personality: str = ""
    backstory: str = ""
    appearance: str = ""
    motivations: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    traits: List[CharacterTrait] = field(default_factory=list)
    relationships: List[CharacterRelationship] = field(default_factory=list)
    development: List[CharacterDevelopment] = field(default_factory=list)
    voice_patterns: List[str] = field(default_factory=list)  # Speech patterns
    catchphrases: List[str] = field(default_factory=list)
    current_status: str = "alive"
    current_location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "personality": self.personality,
            "backstory": self.backstory,
            "appearance": self.appearance,
            "motivations": self.motivations,
            "fears": self.fears,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "traits": [{"name": t.name, "description": t.description, "intensity": t.intensity} for t in self.traits],
            "relationships": [{"target": r.target_character, "type": r.relationship_type, "description": r.description} for r in self.relationships],
            "voice_patterns": self.voice_patterns,
            "catchphrases": self.catchphrases,
            "current_status": self.current_status,
            "current_location": self.current_location,
        }

    def get_profile_text(self) -> str:
        """Get character profile as text for RAG."""
        parts = [
            f"Tên: {self.name}",
            f"Tuổi: {self.age}" if self.age else None,
            f"Giới tính: {self.gender}" if self.gender else None,
            f"Tính cách: {self.personality}",
            f"Ngoại hình: {self.appearance}" if self.appearance else None,
            f"Quá khứ: {self.backstory}" if self.backstory else None,
            f"Động lực: {', '.join(self.motivations)}" if self.motivations else None,
            f"Sợ hãi: {', '.join(self.fears)}" if self.fears else None,
            f"Điểm mạnh: {', '.join(self.strengths)}" if self.strengths else None,
            f"Điểm yếu: {', '.join(self.weaknesses)}" if self.weaknesses else None,
            f"Trạng thái: {self.current_status}",
            f"Vị trí: {self.current_location}" if self.current_location else None,
        ]
        return "\n".join([p for p in parts if p])


class CharacterManager:
    """
    Character management system for story writing.

    Manages character profiles, relationships, and development.

    Example:
        manager = CharacterManager()

        # Add character
        character = Character(
            name="A",
            age=25,
            personality="Thông minh, quyết đoán"
        )
        manager.add_character(character)

        # Add relationship
        manager.add_relationship("A", "B", "friend", "Bạn thân từ nhỏ")

        # Track development
        manager.add_development("A", chapter=5, event="Phát hiện bí mật")

        # Get character context for RAG
        context = manager.get_character_context("A")
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize character manager.

        Args:
            storage_path: Path for storing character data
        """
        self.storage_path = storage_path
        self.characters: Dict[str, Character] = {}

        # Load existing data
        if storage_path:
            self._load_data()

    def add_character(self, character: Character) -> None:
        """
        Add character to manager.

        Args:
            character: Character to add
        """
        self.characters[character.name] = character
        self._save_data()

    def get_character(self, name: str) -> Optional[Character]:
        """
        Get character by name.

        Args:
            name: Character name

        Returns:
            Character or None
        """
        return self.characters.get(name)

    def update_character(self, name: str, **kwargs) -> None:
        """
        Update character attributes.

        Args:
            name: Character name
            **kwargs: Attributes to update
        """
        character = self.characters.get(name)
        if not character:
            raise ValueError(f"Character not found: {name}")

        for key, value in kwargs.items():
            if hasattr(character, key):
                setattr(character, key, value)

        character.updated_at = datetime.now()
        self._save_data()

    def delete_character(self, name: str) -> None:
        """
        Delete character.

        Args:
            name: Character name
        """
        if name in self.characters:
            del self.characters[name]
            self._save_data()

    def add_relationship(
        self,
        source: str,
        target: str,
        relationship_type: str,
        description: str,
        is_positive: bool = True
    ) -> None:
        """
        Add relationship between characters.

        Args:
            source: Source character name
            target: Target character name
            relationship_type: Type of relationship
            description: Description of relationship
            is_positive: Whether relationship is positive
        """
        source_char = self.characters.get(source)
        target_char = self.characters.get(target)

        if not source_char or not target_char:
            raise ValueError(f"Character not found: {source} or {target}")

        # Add to source
        source_char.relationships.append(CharacterRelationship(
            target_character=target,
            relationship_type=relationship_type,
            description=description,
            is_positive=is_positive
        ))

        # Add reverse to target
        reverse_type = self._get_reverse_relationship(relationship_type)
        target_char.relationships.append(CharacterRelationship(
            target_character=source,
            relationship_type=reverse_type,
            description=description,
            is_positive=is_positive
        ))

        self._save_data()

    def add_development(
        self,
        character_name: str,
        chapter: int,
        event: str,
        impact: str = "neutral",
        traits_affected: Optional[List[str]] = None,
        description: str = ""
    ) -> None:
        """
        Track character development.

        Args:
            character_name: Character name
            chapter: Chapter number
            event: Development event
            impact: Impact type ("positive", "negative", "neutral")
            traits_affected: Traits affected by this development
            description: Detailed description
        """
        character = self.characters.get(character_name)
        if not character:
            raise ValueError(f"Character not found: {character_name}")

        character.development.append(CharacterDevelopment(
            chapter=chapter,
            event=event,
            impact=impact,
            traits_affected=traits_affected or [],
            description=description
        ))

        character.updated_at = datetime.now()
        self._save_data()

    def add_trait(
        self,
        character_name: str,
        trait_name: str,
        description: str,
        intensity: float = 0.5,
        is_positive: bool = True
    ) -> None:
        """
        Add trait to character.

        Args:
            character_name: Character name
            trait_name: Trait name
            description: Trait description
            intensity: Trait intensity (0.0 to 1.0)
            is_positive: Whether trait is positive
        """
        character = self.characters.get(character_name)
        if not character:
            raise ValueError(f"Character not found: {character_name}")

        character.traits.append(CharacterTrait(
            name=trait_name,
            description=description,
            intensity=intensity,
            is_positive=is_positive
        ))

        self._save_data()

    def get_character_context(self, name: str) -> str:
        """
        Get character context for RAG.

        Args:
            name: Character name

        Returns:
            Formatted character context
        """
        character = self.characters.get(name)
        if not character:
            return f"Không tìm thấy nhân vật: {name}"

        return character.get_profile_text()

    def get_all_characters_context(self) -> str:
        """
        Get all characters context for RAG.

        Returns:
            Formatted context for all characters
        """
        parts = []
        for char in self.characters.values():
            parts.append(char.get_profile_text())
        return "\n\n---\n\n".join(parts)

    def get_relationships_context(self, name: str) -> str:
        """
        Get relationships context for character.

        Args:
            name: Character name

        Returns:
            Formatted relationships context
        """
        character = self.characters.get(name)
        if not character:
            return f"Không tìm thấy nhân vật: {name}"

        if not character.relationships:
            return f"{name} không có mối quan hệ nào được ghi nhận."

        parts = [f"Mối quan hệ của {name}:"]
        for rel in character.relationships:
            parts.append(f"- {rel.target_character}: {rel.relationship_type} - {rel.description}")

        return "\n".join(parts)

    def get_development_arc(self, name: str) -> str:
        """
        Get character development arc.

        Args:
            name: Character name

        Returns:
            Formatted development arc
        """
        character = self.characters.get(name)
        if not character:
            return f"Không tìm thấy nhân vật: {name}"

        if not character.development:
            return f"{name} chưa có sự phát triển nào được ghi nhận."

        parts = [f"Hành trình phát triển của {name}:"]
        for dev in character.development:
            parts.append(f"Chương {dev.chapter}: {dev.event} ({dev.impact})")
            if dev.description:
                parts.append(f"  → {dev.description}")

        return "\n".join(parts)

    def find_characters_by_location(self, location: str) -> List[Character]:
        """
        Find characters at location.

        Args:
            location: Location name

        Returns:
            List of characters at location
        """
        return [
            char for char in self.characters.values()
            if char.current_location == location
        ]

    def find_related_characters(self, name: str) -> List[str]:
        """
        Find all characters related to given character.

        Args:
            name: Character name

        Returns:
            List of related character names
        """
        character = self.characters.get(name)
        if not character:
            return []

        return [rel.target_character for rel in character.relationships]

    def _get_reverse_relationship(self, rel_type: str) -> str:
        """Get reverse relationship type."""
        reverse_map = {
            "friend": "friend",
            "enemy": "enemy",
            "lover": "lover",
            "parent": "child",
            "child": "parent",
            "mentor": "student",
            "student": "mentor",
            "boss": "employee",
            "employee": "boss",
        }
        return reverse_map.get(rel_type, "related")

    def _save_data(self) -> None:
        """Save character data to storage."""
        if not self.storage_path:
            return

        data = {}
        for name, char in self.characters.items():
            data[name] = char.to_dict()

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_data(self) -> None:
        """Load character data from storage."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for name, char_data in data.items():
                character = Character(
                    name=char_data["name"],
                    age=char_data.get("age"),
                    gender=char_data.get("gender"),
                    personality=char_data.get("personality", ""),
                    backstory=char_data.get("backstory", ""),
                    appearance=char_data.get("appearance", ""),
                    motivations=char_data.get("motivations", []),
                    fears=char_data.get("fears", []),
                    strengths=char_data.get("strengths", []),
                    weaknesses=char_data.get("weaknesses", []),
                    voice_patterns=char_data.get("voice_patterns", []),
                    catchphrases=char_data.get("catchphrases", []),
                    current_status=char_data.get("current_status", "alive"),
                    current_location=char_data.get("current_location"),
                )

                # Load traits
                for t in char_data.get("traits", []):
                    character.traits.append(CharacterTrait(
                        name=t["name"],
                        description=t["description"],
                        intensity=t.get("intensity", 0.5),
                    ))

                # Load relationships
                for r in char_data.get("relationships", []):
                    character.relationships.append(CharacterRelationship(
                        target_character=r["target"],
                        relationship_type=r["type"],
                        description=r["description"],
                    ))

                self.characters[name] = character

        except FileNotFoundError:
            pass

    @property
    def character_count(self) -> int:
        """Number of characters."""
        return len(self.characters)

    def list_characters(self) -> List[str]:
        """List all character names."""
        return list(self.characters.keys())

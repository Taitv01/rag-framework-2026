"""
Example 08: Story Writing System
================================

Complete story writing system with RAG support.

This example demonstrates:
1. Character management
2. Plot management
3. World building
4. Consistency checking
5. Writing assistance

Requirements:
    pip install -r requirements.txt

Usage:
    python examples/08_story_writing.py
"""

import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run story writing example."""

    print("=" * 60)
    print("Story Writing System")
    print("=" * 60)

    from src.story import (
        CharacterManager, Character,
        PlotManager, PlotPoint, PlotArc, Foreshadowing,
        WorldBuilder, Location, Lore,
        ChapterManager, Chapter,
        TimelineManager, TimelineEvent,
        WritingAssistant
    )

    # =========================================================================
    # 1. Setup Character Manager
    # =========================================================================
    print("\n1. Setting up Characters...")
    print("-" * 40)

    char_manager = CharacterManager()

    # Add main character
    character_a = Character(
        name="Nguyễn Văn A",
        age=25,
        gender="Nam",
        personality="Thông minh, quyết đoán, đôi khi nóng nảy",
        backstory="Mồ côi từ nhỏ, lớn lên ở trại trẻ mồ côi",
        appearance="Cao 1m75, tóc đen, mắt nâu, nụ cười ấm áp",
        motivations=["Tìm lại gia đình", "Chứng minh bản thân"],
        fears=["Bị bỏ rơi", "Không được yêu thương"],
        strengths=["Trí nhớ tốt", "Can đảm", "Trung thành"],
        weaknesses=["Nóng nảy", "Hay nghi ngờ"],
        voice_patterns=["Thường nói ngắn gọn", "Hay dùng câu hỏi"],
    )
    char_manager.add_character(character_a)

    # Add second character
    character_b = Character(
        name="Trần Thị B",
        age=23,
        gender="Nữ",
        personality="Hiền lành, thông minh, đôi khi nhút nhát",
        backstory="Con gái duy nhất của gia đình giàu có",
        appearance="Dáng người nhỏ nhắn, tóc dài, mắt to",
        motivations=["Tự lập", "Giúp đỡ người khác"],
        fears=["Mất người thân", "Cô đơn"],
        strengths=["Kiên nhẫn", "Tốt bụng", "Am hiểu y học"],
        weaknesses=["Nhút nhát", "Hay lo lắng"],
    )
    char_manager.add_character(character_b)

    # Add relationship
    char_manager.add_relationship(
        "Nguyễn Văn A", "Trần Thị B",
        "friend", "Bạn thân từ nhỏ, cùng lớn lên ở trại mồ côi"
    )

    print(f"   Added {char_manager.character_count} characters")
    print(f"   Characters: {', '.join(char_manager.list_characters())}")

    # =========================================================================
    # 2. Setup Plot Manager
    # =========================================================================
    print("\n2. Setting up Plot...")
    print("-" * 40)

    plot_manager = PlotManager()

    # Add plot points
    plot_manager.add_plot_point(PlotPoint(
        chapter=1,
        event="A tìm thấy bức thư cũ trong ngăn kéo",
        importance="high",
        characters_involved=["Nguyễn Văn A"],
        location="Nhà A"
    ))

    plot_manager.add_plot_point(PlotPoint(
        chapter=1,
        event="Bức thư tiết lộ A là con nuôi",
        importance="critical",
        characters_involved=["Nguyễn Văn A"],
    ))

    # Create plot arc
    plot_manager.create_plot_arc(PlotArc(
        name="Hành trình tìm lại gia đình",
        description="A khám phá bí mật về gia đình thực sự",
        start_chapter=1,
        characters_involved=["Nguyễn Văn A", "Trần Thị B"],
        themes=["Gia đình", "Bản sắc", "Tình bạn"]
    ))

    # Add foreshadowing
    plot_manager.add_foreshadowing(
        chapter=1,
        hint="Bức thư có mùi lạ, giấy vàng ố",
        resolution_chapter=10,
        importance="high"
    )

    print(f"   Added {len(plot_manager.plot_points)} plot points")
    print(f"   Added {len(plot_manager.plot_arcs)} plot arcs")
    print(f"   Added {len(plot_manager.foreshadowing)} foreshadowing elements")

    # =========================================================================
    # 3. Setup World Builder
    # =========================================================================
    print("\n3. Setting up World...")
    print("-" * 40)

    world_builder = WorldBuilder()

    # Add locations
    world_builder.add_location(Location(
        name="Hà Nội",
        description="Thủ đô ngàn năm văn hiến, nơi A lớn lên",
        climate="Nhiệt đới gió mùa, nóng ẩm mùa hè, lạnh mùa đông",
        landmarks=["Hồ Gươm", "Phố Cổ", "Văn Miếu"],
        first_appearance=1
    ))

    world_builder.add_location(Location(
        name="Đà Lạt",
        description="Thành phố mù sương, nơi A tìm manh mối",
        climate="Mát mẻ quanh năm, nhiều sương mù",
        landmarks=["Hồ Xuân Hương", "Đồi Cù", "Thung lũng Tình Yêu"],
        first_appearance=5
    ))

    # Add lore
    world_builder.add_lore(Lore(
        name="Bí mật ngọn đồi",
        content="Người ta kể rằng trên ngọn đồi có một ngôi nhà cổ...",
        category="legend",
        related_locations=["Đà Lạt"]
    ))

    print(f"   Added {len(world_builder.locations)} locations")
    print(f"   Added {len(world_builder.lore)} lore elements")

    # =========================================================================
    # 4. Setup Timeline
    # =========================================================================
    print("\n4. Setting up Timeline...")
    print("-" * 40)

    timeline = TimelineManager()

    timeline.add_event(TimelineEvent(
        chapter=1,
        time="Buổi sáng",
        event="A thức dậy, tìm thấy bức thư",
        location="Nhà A",
        characters_involved=["Nguyễn Văn A"]
    ))

    timeline.add_event(TimelineEvent(
        chapter=1,
        time="Buổi chiều",
        event="A đến trại trẻ mồ côi tìm thông tin",
        location="Trại trẻ mồ côi",
        characters_involved=["Nguyễn Văn A"]
    ))

    print(f"   Added timeline events")

    # =========================================================================
    # 5. Get Context for RAG
    # =========================================================================
    print("\n5. Getting Context for RAG...")
    print("-" * 40)

    # Character context
    char_context = char_manager.get_character_context("Nguyễn Văn A")
    print("\n   Character Context:")
    print(f"   {char_context[:200]}...")

    # Plot context
    plot_context = plot_manager.get_plot_context()
    print("\n   Plot Context:")
    print(f"   {plot_context[:200]}...")

    # World context
    world_context = world_builder.get_world_context()
    print("\n   World Context:")
    print(f"   {world_context[:200]}...")

    # Timeline
    timeline_text = timeline.get_timeline()
    print("\n   Timeline:")
    print(f"   {timeline_text[:200]}...")

    # =========================================================================
    # 6. Consistency Report
    # =========================================================================
    print("\n6. Consistency Report...")
    print("-" * 40)

    # Check unresolved foreshadowing
    unresolved = plot_manager.get_unresolved_foreshadowing()
    print(f"   Unresolved foreshadowing: {len(unresolved)}")

    # Check active arcs
    active_arcs = plot_manager.get_active_arcs()
    print(f"   Active plot arcs: {len(active_arcs)}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Characters: {char_manager.character_count}")
    print(f"Plot Points: {len(plot_manager.plot_points)}")
    print(f"Plot Arcs: {len(plot_manager.plot_arcs)}")
    print(f"Locations: {len(world_builder.locations)}")
    print(f"Lore Elements: {len(world_builder.lore)}")
    print("=" * 60)

    print("\nSystem ready for story writing!")
    print("Use the managers to maintain consistency across chapters.")


if __name__ == "__main__":
    main()

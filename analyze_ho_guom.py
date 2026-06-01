"""
Phân Tích và Chỉnh Sửa Truyện Hồ Gươm
========================================

Sử dụng RAG Framework để:
1. Phân tích tính nhất quán
2. Kiểm tra lặp từ
3. Gợi ý cải thiện
4. Tạo bản chỉnh sửa
"""

import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.story import (
    CharacterManager, Character,
    PlotManager, PlotPoint, PlotArc,
    WorldBuilder, Location, Lore,
    TimelineManager, TimelineEvent,
    ChapterManager, Chapter
)


def analyze_story():
    """Phân tích toàn bộ truyện Hồ Gươm."""

    print("=" * 70)
    print("PHÂN TÍCH TRUYỆN HỒ GƯƠM - SỬ DỤNG RAG FRAMEWORK")
    print("=" * 70)

    # =========================================================================
    # 1. Setup Character Manager
    # =========================================================================
    print("\n1. THIẾT LẬP NHÂN VẬT")
    print("-" * 50)

    char_manager = CharacterManager()

    # Lê Lợi
    char_manager.add_character(Character(
        name="Lê Lợi",
        age=None,
        gender="Nam",
        personality="Cương trực, nghĩa hiệp, quyết đoán, kiên trì, thương dân",
        backstory="Hào trưởng đất Lam Sơn, Thanh Hóa, nung nấu chí lớn đánh đuổi quân Minh",
        appearance="Dáng người oai phong, mắt sáng quắc, mặc áo bào nâu sẫm của hào trưởng",
        motivations=["Giải phóng đất nước", "Cứu dân khỏi ách nô lệ"],
        fears=["Thất bại", "Dân chúng chịu khổ"],
        strengths=["Lãnh đạo", "Kiên trì", "Được lòng dân"],
        weaknesses=["Đôi khi nóng vội"],
        voice_patterns=["Nói ngắn gọn, mạnh mẽ", "Hay dùng lời thề"],
    ))

    # Lê Thận
    char_manager.add_character(Character(
        name="Lê Thận",
        age=None,
        gender="Nam",
        personality="Trung thành, dũng cảm, yêu nước, chất phác",
        backstory="Ngư dân nghèo bên dòng sông Lương, chứng kiến cảnh dân bị áp bức",
        appearance="Dáng người rắn rỏi, mặc áo nâu ngắn, quần xắn, đội nón lá",
        motivations=["Theo Lê Lợi đánh giặc", "Cứu nước"],
        fears=["Mất đồng đội"],
        strengths=["Trung thành", "Dũng cảm", "Am hiểu sông nước"],
        weaknesses=["Ít học"],
    ))

    # Nguyễn Trãi
    char_manager.add_character(Character(
        name="Nguyễn Trãi",
        age=None,
        gender="Nam",
        personality="Thông minh, mưu lược, văn võ song toàn",
        backstory="Học trò lỗi lạc, giúp Lê Lợi bày mưu tính kế",
        appearance="Mặc áo dài trắng của học trò, khăn bịt đầu, vẻ mặt điềm đạm",
        motivations=["Giúp Lê Lợi đánh giặc", "Bình thiên hạ"],
        strengths=["Mưu lược", "Văn chương", "Chiến thuật"],
    ))

    # Kim Quy
    char_manager.add_character(Character(
        name="Kim Quy",
        age=None,
        gender=None,
        personality="Uy nghiêm, cổ xưa, công bằng",
        backstory="Sứ giả của Long Vương, đến nhận lại kiếm thần",
        appearance="Rùa vàng khổng lồ, mai sáng như vàng ròng, mắt sáng như sao",
        motivations=["Nhận lại kiếm thần", "Duy trì công bằng"],
    ))

    # Add relationships
    char_manager.add_relationship("Lê Lợi", "Lê Thận", "mentor", "Chủ tướng và tướng tiên phong")
    char_manager.add_relationship("Lê Lợi", "Nguyễn Trãi", "mentor", "Vua và quân sư")
    char_manager.add_relationship("Lê Lợi", "Kim Quy", "related", "Gặp nhau khi trả kiếm")

    print(f"   Đã thiết lập {char_manager.character_count} nhân vật")

    # =========================================================================
    # 2. Setup Plot Manager
    # =========================================================================
    print("\n2. THIẾT LẬP CỐT TRUYỆN")
    print("-" * 50)

    plot_manager = PlotManager()

    # Plot points chính
    plot_points = [
        (1, "Việt Nam dưới ách Minh", "high"),
        (2, "Giới thiệu Lê Lợi và Lê Thận", "high"),
        (3, "Lê Thận vớt được lưỡi kiếm thần", "critical"),
        (4, "Ráp kiếm Thuận Thiên hoàn chỉnh", "critical"),
        (5, "Khởi nghĩa Lam Sơn", "high"),
        (6, "Những ngày gian khổ", "medium"),
        (7, "Tưởng nhớ anh hùng", "medium"),
        (8, "Nguyễn Trãi xuất hiện", "high"),
        (9, "Sức mạnh kiếm thần", "medium"),
        (10, "Mười năm kháng chiến", "high"),
        (11, "Trận đại chiến cuối cùng", "critical"),
        (12, "Ngày toàn thắng", "critical"),
        (13, "Lê Thái Tổ đăng quang", "high"),
        (14, "Kinh đô Thăng Long", "medium"),
        (15, "Rùa Vàng xuất hiện", "critical"),
        (16, "Trả kiếm cho Long Vương", "critical"),
        (17, "Kiếm chìm dưới hồ", "high"),
        (18, "Hồ Hoàn Kiếm ngày nay", "medium"),
        (19, "Bóng Rùa Vàng", "medium"),
        (20, "Bài học muôn đời", "high"),
    ]

    for chapter, event, importance in plot_points:
        plot_manager.add_plot_point(PlotPoint(
            chapter=chapter,
            event=event,
            importance=importance,
        ))

    # Plot arcs
    plot_manager.create_plot_arc(PlotArc(
        name="Cuộc kháng chiến chống Minh",
        description="Từ khởi nghĩa đến toàn thắng",
        start_chapter=1,
        end_chapter=12,
        themes=["Yêu nước", "Kiên trì", "Chính nghĩa"],
    ))

    plot_manager.create_plot_arc(PlotArc(
        name="Kiếm thần Thuận Thiên",
        description="Từ tìm kiếm đến trả lại",
        start_chapter=3,
        end_chapter=17,
        themes=["Thần kỳ", "Sứ mệnh", "Khiêm nhường"],
    ))

    # Foreshadowing
    plot_manager.add_foreshadowing(1, "Ngọn lửa yêu nước âm ỉ cháy", 5)
    plot_manager.add_foreshadowing(3, "Kiếm phát sáng khi Lê Lợi đến", 4)
    plot_manager.add_foreshadowing(15, "Kiếm đã hoàn thành sứ mệnh", 16)

    print(f"   Đã thiết lập {len(plot_manager.plot_points)} plot points")
    print(f"   Đã thiết lập {len(plot_manager.plot_arcs)} plot arcs")
    print(f"   Đã thiết lập {len(plot_manager.foreshadowing)} foreshadowing")

    # =========================================================================
    # 3. Setup World Builder
    # =========================================================================
    print("\n3. THIẾT LẬP THẾ GIỚI")
    print("-" * 50)

    world_builder = WorldBuilder()

    world_builder.add_location(Location(
        name="Lam Sơn, Thanh Hóa",
        description="Quê hương của Lê Lợi, nơi khởi nghĩa",
        first_appearance=2,
    ))

    world_builder.add_location(Location(
        name="Sông Lương",
        description="Dòng sông nơi Lê Thận vớt được kiếm thần",
        first_appearance=3,
    ))

    world_builder.add_location(Location(
        name="Hồ Hoàn Kiếm",
        description="Hồ trả kiếm, biểu tượng Hà Nội",
        first_appearance=15,
    ))

    world_builder.add_location(Location(
        name="Thăng Long",
        description="Kinh đô mới, Hà Nội ngày nay",
        first_appearance=14,
    ))

    world_builder.add_lore(Lore(
        name="Kiếm thần Thuận Thiên",
        content="Thanh kiếm thần do Long Vương cho mượn để cứu nước",
        category="legend",
    ))

    world_builder.add_lore(Lore(
        name="Rùa Vàng",
        content="Sứ giả của Long Vương, đến nhận lại kiếm thần",
        category="legend",
    ))

    print(f"   Đã thiết lập {len(world_builder.locations)} địa điểm")
    print(f"   Đã thiết lập {len(world_builder.lore)} truyền thuyết")

    # =========================================================================
    # 4. Phân Tích Tính Nhất Quán
    # =========================================================================
    print("\n4. PHÂN TÍCH TÍNH NHẤT QUÁN")
    print("-" * 50)

    # Đọc kịch bản
    story_path = r"C:\Users\Dungt\OneDrive\TÀI\Máy tính\VEO 3\Ho guomg-20260529T040738Z-3-001\KICH_BAN_20_CANH.txt"

    with open(story_path, "r", encoding="utf-8") as f:
        story_content = f.read()

    # Tách các cảnh
    scenes = []
    current_scene = ""
    current_scene_num = 0

    for line in story_content.split("\n"):
        if line.startswith("--- CẢNH"):
            if current_scene:
                scenes.append((current_scene_num, current_scene.strip()))
            current_scene_num += 1
            current_scene = ""
        else:
            current_scene += line + "\n"

    if current_scene:
        scenes.append((current_scene_num, current_scene.strip()))

    print(f"   Tổng số cảnh: {len(scenes)}")

    # =========================================================================
    # 5. Phát Hiện Vấn Đề
    # =========================================================================
    print("\n5. PHÁT HIỆN VẤN ĐỀ")
    print("-" * 50)

    issues = []

    # Kiểm tra lặp từ
    print("\n   a) Kiểm tra lặp từ:")
    repetitive_words = [
        ("vang dội", "Lặp quá nhiều"),
        ("sáng rực", "Nên thay bằng từ đồng nghĩa"),
        ("quân Minh", "Có thể dùng 'quân thù', 'giặc'"),
        ("nước mắt", "Lặp nhiều ở cuối"),
        ("không bao giờ", "Lặp nhiều lần"),
    ]

    for word, note in repetitive_words:
        count = story_content.lower().count(word.lower())
        if count > 3:
            issues.append(f"      - '{word}': xuất hiện {count} lần - {note}")
            print(f"      ⚠️ '{word}': xuất hiện {count} lần - {note}")

    # Kiểm tra lặp ý
    print("\n   b) Kiểm tra lặp ý:")
    duplicate_ideas = [
        ("Lê Lợi đứng trên thuyền rất lâu", "Lặp 2 lần ở cảnh 17"),
        ("Rùa Vàng từ từ lặn xuống", "Lặp 2 lần ở cảnh 16"),
        ("nhân dân từ khắp nơi kéo về", "Lặp nhiều lần"),
        ("tiếng hô vang dội núi sông", "Lặp nhiều lần"),
    ]

    for idea, note in duplicate_ideas:
        if story_content.count(idea) > 1:
            issues.append(f"      - '{idea}': {note}")
            print(f"      ⚠️ '{idea}': {note}")

    # Kiểm tra câu quá dài
    print("\n   c) Kiểm tra câu quá dài:")
    for i, (scene_num, scene_text) in enumerate(scenes):
        sentences = scene_text.split(".")
        for sentence in sentences:
            if len(sentence) > 200:
                issues.append(f"      - Cảnh {scene_num}: Câu quá dài ({len(sentence)} ký tự)")
                print(f"      ⚠️ Cảnh {scene_num}: Câu quá dài ({len(sentence)} ký tự)")
                break

    # =========================================================================
    # 6. Gợi Ý Cải Thiện
    # =========================================================================
    print("\n6. GỢI Ý CẢI THIỆN")
    print("-" * 50)

    suggestions = [
        ("Giảm lặp từ", "Thay 'vang dội' bằng: vang vọng, ngân vang, réo rắt"),
        ("Giảm lặp ý", "Gộp các câu có cùng ý nghĩa"),
        ("Câu ngắn hơn", "Chia câu dài thành 2-3 câu ngắn"),
        ("Thêm cảm xúc", "Mô tả nội tâm nhân vật sâu hơn"),
        ("Nhịp điệu", "Đan xen câu ngắn và câu dài"),
        ("Hội thoại", "Thêm lời thoại tự nhiên hơn"),
        ("Mô tả", "Thêm chi tiết giác quan (âm thanh, mùi, vị)"),
    ]

    for title, detail in suggestions:
        print(f"   💡 {title}: {detail}")

    # =========================================================================
    # 7. Tóm Tắt
    # =========================================================================
    print("\n" + "=" * 70)
    print("TÓM TẮT PHÂN TÍCH")
    print("=" * 70)
    print(f"   Tổng số cảnh: {len(scenes)}")
    print(f"   Nhân vật: {char_manager.character_count}")
    print(f"   Plot points: {len(plot_manager.plot_points)}")
    print(f"   Vấn đề phát hiện: {len(issues)}")
    print(f"   Gợi ý cải thiện: {len(suggestions)}")
    print("=" * 70)

    return issues, suggestions


if __name__ == "__main__":
    analyze_story()

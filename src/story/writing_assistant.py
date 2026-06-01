"""
Writing Assistant
=================

AI-powered writing assistant for story writing.

Features:
- Content generation
- Dialogue writing
- Description generation
- Style consistency
- Pacing suggestions

Usage:
    from src.story import WritingAssistant

    assistant = WritingAssistant(llm, character_manager, plot_manager)
    content = assistant.write_chapter(chapter_number=5)
"""

from typing import List, Optional, Dict, Any


class WritingAssistant:
    """
    AI-powered writing assistant.

    Example:
        assistant = WritingAssistant(llm, character_manager, plot_manager)

        # Write chapter
        content = assistant.write_chapter(
            chapter_number=5,
            main_characters=["A", "B"],
            location="Đà Lạt",
            plot_points=["A tìm thấy manh mối"]
        )

        # Generate dialogue
        dialogue = assistant.generate_dialogue(
            characters=["A", "B"],
            topic="Bí mật"
        )
    """

    def __init__(
        self,
        llm,
        character_manager=None,
        plot_manager=None,
        world_builder=None,
        style_guide: Optional[str] = None
    ):
        """
        Initialize writing assistant.

        Args:
            llm: LLM instance
            character_manager: CharacterManager instance
            plot_manager: PlotManager instance
            world_builder: WorldBuilder instance
            style_guide: Writing style guide
        """
        self.llm = llm
        self.character_manager = character_manager
        self.plot_manager = plot_manager
        self.world_builder = world_builder
        self.style_guide = style_guide or self._default_style_guide()

    def _default_style_guide(self) -> str:
        """Default style guide."""
        return """
Phong cách viết:
- Ngôn ngữ: Tiếng Việt
- Giọng điệu: Tự nhiên, gần gũi
- Mô tả: Chi tiết nhưng không dài dòng
- Hội thoại: Tự nhiên, phù hợp tính cách nhân vật
- Nhịp điệu: Đan xen nhanh và chậm
"""

    def write_chapter(
        self,
        chapter_number: int,
        main_characters: List[str],
        location: str,
        plot_points: List[str],
        tone: str = "neutral",
        word_count: int = 2000
    ) -> str:
        """
        Write a chapter.

        Args:
            chapter_number: Chapter number
            main_characters: Main characters in chapter
            location: Chapter location
            plot_points: Plot points to cover
            tone: Chapter tone
            word_count: Target word count

        Returns:
            Chapter content
        """
        # Get character context
        char_context = ""
        if self.character_manager:
            char_parts = []
            for name in main_characters:
                char_parts.append(self.character_manager.get_character_context(name))
            char_context = "\n\n".join(char_parts)

        # Get plot context
        plot_context = ""
        if self.plot_manager:
            plot_context = self.plot_manager.get_plot_context()

        # Get world context
        world_context = ""
        if self.world_builder:
            world_context = self.world_builder.get_location_context(location)

        prompt = f"""Viết chương {chapter_number} với thông tin sau:

Nhân vật chính:
{char_context}

Địa điểm: {location}
{world_context}

Plot points cần đề cập:
{chr(10).join([f"- {p}" for p in plot_points])}

Tông giọng: {tone}
Số từ mục tiêu: {word_count}

{self.style_guide}

Yêu cầu:
1. Nhất quán với tính cách nhân vật
2. Mô tả địa điểm sinh động
3. Phát triển plot points
4. Giữ nhịp điệu hợp lý
5. Hội thoại tự nhiên

Viết chương hoàn chỉnh:"""

        return self.llm.generate(prompt)

    def generate_dialogue(
        self,
        characters: List[str],
        topic: str,
        context: str = "",
        num_exchanges: int = 5
    ) -> str:
        """
        Generate dialogue between characters.

        Args:
            characters: Characters in dialogue
            topic: Dialogue topic
            context: Additional context
            num_exchanges: Number of exchanges

        Returns:
            Dialogue text
        """
        # Get character voices
        char_voices = []
        if self.character_manager:
            for name in characters:
                char = self.character_manager.get_character(name)
                if char:
                    char_voices.append(f"{name}: {char.personality}. Nói: {', '.join(char.voice_patterns) if char.voice_patterns else 'Bình thường'}")

        prompt = f"""Tạo hội thoại giữa các nhân vật:

Nhân vật:
{chr(10).join(char_voices)}

Chủ đề: {topic}
Ngữ cảnh: {context}
Số lần trao đổi: {num_exchanges}

Yêu cầu:
1. Mỗi nhân vật nói đúng tính cách
2. Hội thoại tự nhiên
3. Đẩy mạnh plot
4. Có cảm xúc

Hội thoại:"""

        return self.llm.generate(prompt)

    def generate_description(
        self,
        subject: str,
        details: str = "",
        mood: str = "neutral",
        length: str = "medium"
    ) -> str:
        """
        Generate description.

        Args:
            subject: Subject to describe
            details: Additional details
            mood: Description mood
            length: Description length ("short", "medium", "long")

        Returns:
            Description text
        """
        prompt = f"""Viết mô tả cho:

Đối tượng: {subject}
Chi tiết: {details}
Tâm trạng: {mood}
Độ dài: {length}

{self.style_guide}

Mô tả:"""

        return self.llm.generate(prompt)

    def suggest_pacing(self, chapter_content: str) -> str:
        """
        Suggest pacing improvements.

        Args:
            chapter_content: Chapter content

        Returns:
            Pacing suggestions
        """
        prompt = f"""Phân tích nhịp điệu của chương sau và gợi ý cải thiện:

{chapter_content[:3000]}

Đánh giá:
1. Nhịp nhanh/chậm có cân bằng không?
2. Có đoạn nào quá dài không?
3. Có đoạn nào quá ngắn không?
4. Hội thoại và mô tả có cân bằng không?

Gợi ý cải thiện:"""

        return self.llm.generate(prompt)

    def check_style_consistency(self, chapter_content: str) -> str:
        """
        Check style consistency.

        Args:
            chapter_content: Chapter content

        Returns:
            Style consistency report
        """
        prompt = f"""Kiểm tra tính nhất quán phong cách của chương sau:

{chapter_content[:3000]}

{self.style_guide}

Đánh giá:
1. Ngôn ngữ có nhất quán không?
2. Giọng điệu có đồng đều không?
3. Có đoạn nào lệch phong cách không?
4. Có lỗi chính tả hoặc ngữ pháp không?

Báo cáo:"""

        return self.llm.generate(prompt)

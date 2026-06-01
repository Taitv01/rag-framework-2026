"""
Consistency Checker
===================

Consistency checking system for story writing.

Features:
- Character consistency checking
- Plot consistency checking
- Timeline consistency checking
- Fact checking
- Contradiction detection

Usage:
    from src.story import ConsistencyChecker

    checker = ConsistencyChecker(llm, character_manager, plot_manager)

    # Check chapter consistency
    issues = checker.check_chapter(chapter_text, chapter_number)

    # Check character consistency
    issues = checker.check_character_consistency("A")

    # Check plot consistency
    issues = checker.check_plot_consistency()
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ConsistencyIssue:
    """Consistency issue found."""
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "character", "plot", "timeline", "fact"
    description: str
    location: Optional[str] = None  # Where the issue was found
    suggestion: Optional[str] = None  # How to fix it


class ConsistencyChecker:
    """
    Consistency checking system for story writing.

    Checks for:
    - Character trait consistency
    - Plot contradiction detection
    - Timeline accuracy
    - Fact consistency
    - Foreshadowing resolution

    Example:
        checker = ConsistencyChecker(llm, character_manager, plot_manager)

        # Check chapter
        issues = checker.check_chapter(chapter_text, chapter_number)

        # Check character
        issues = checker.check_character_consistency("A")

        # Get report
        report = checker.get_consistency_report()
    """

    def __init__(
        self,
        llm,
        character_manager=None,
        plot_manager=None,
        world_builder=None
    ):
        """
        Initialize consistency checker.

        Args:
            llm: LLM instance
            character_manager: CharacterManager instance
            plot_manager: PlotManager instance
            world_builder: WorldBuilder instance
        """
        self.llm = llm
        self.character_manager = character_manager
        self.plot_manager = plot_manager
        self.world_builder = world_builder

    def check_chapter(
        self,
        chapter_text: str,
        chapter_number: int
    ) -> List[ConsistencyIssue]:
        """
        Check chapter for consistency issues.

        Args:
            chapter_text: Chapter text
            chapter_number: Chapter number

        Returns:
            List of consistency issues
        """
        issues = []

        # Check character consistency
        character_issues = self._check_characters_in_text(chapter_text, chapter_number)
        issues.extend(character_issues)

        # Check plot consistency
        plot_issues = self._check_plot_in_text(chapter_text, chapter_number)
        issues.extend(plot_issues)

        # Check timeline consistency
        timeline_issues = self._check_timeline_in_text(chapter_text, chapter_number)
        issues.extend(timeline_issues)

        # Check world consistency
        if self.world_builder:
            world_issues = self._check_world_in_text(chapter_text, chapter_number)
            issues.extend(world_issues)

        return issues

    def check_character_consistency(self, character_name: str) -> List[ConsistencyIssue]:
        """
        Check character consistency across story.

        Args:
            character_name: Character name

        Returns:
            List of consistency issues
        """
        if not self.character_manager:
            return []

        character = self.character_manager.get_character(character_name)
        if not character:
            return [ConsistencyIssue(
                severity="high",
                category="character",
                description=f"Không tìm thấy nhân vật: {character_name}"
            )]

        issues = []

        # Check trait consistency
        prompt = f"""Kiểm tra tính nhất quán của nhân vật sau:

{character.get_profile_text()}

Phát triển nhân vật:
{chr(10).join([f"- Chương {d.chapter}: {d.event}" for d in character.development])}

Kiểm tra:
1. Tính cách có thay đổi đột ngột không?
2. Hành động có phù hợp với tính cách không?
3. Mối quan hệ có nhất quán không?
4. Ngoại hình có thay đổi bất thường không?

Trả về danh sách các vấn đề, mỗi dòng một vấn đề. Nếu không có vấn đề, trả về "Không có vấn đề"."""

        response = self.llm.generate(prompt)

        if "không có vấn đề" not in response.lower():
            for line in response.strip().split("\n"):
                if line.strip():
                    issues.append(ConsistencyIssue(
                        severity="medium",
                        category="character",
                        description=line.strip(),
                        location=f"Nhân vật: {character_name}"
                    ))

        return issues

    def check_plot_consistency(self) -> List[ConsistencyIssue]:
        """
        Check plot consistency.

        Returns:
            List of consistency issues
        """
        if not self.plot_manager:
            return []

        issues = []

        # Check unresolved foreshadowing
        unresolved = self.plot_manager.get_unresolved_foreshadowing()
        for fs in unresolved:
            issues.append(ConsistencyIssue(
                severity="medium",
                category="plot",
                description=f"Foreshadowing chưa giải quyết: {fs.hint}",
                location=f"Chương {fs.chapter_planted}",
                suggestion="Cần giải quyết hoặc bỏ foreshadowing này"
            ))

        # Check active arcs
        active_arcs = self.plot_manager.get_active_arcs()
        for arc in active_arcs:
            if arc.end_chapter and arc.end_chapter < 100:  # Assuming story isn't 100+ chapters
                issues.append(ConsistencyIssue(
                    severity="low",
                    category="plot",
                    description=f"Plot arc '{arc.name}' chưa kết thúc",
                    suggestion="Xác nhận arc này vẫn đang diễn ra"
                ))

        return issues

    def _check_characters_in_text(
        self,
        text: str,
        chapter: int
    ) -> List[ConsistencyIssue]:
        """Check character consistency in text."""
        if not self.character_manager:
            return []

        issues = []

        # Get character context
        char_context = self.character_manager.get_all_characters_context()

        prompt = f"""Kiểm tra văn bản sau có nhất quán với thông tin nhân vật không:

Thông tin nhân vật:
{char_context}

Văn bản (Chương {chapter}):
{text[:2000]}

Kiểm tra:
1. Nhân vật có hành động trái tính cách không?
2. Mô tả ngoại hình có khớp không?
3. Mối quan hệ có đúng không?
4. Có chi tiết mâu thuẫn với thông tin đã có không?

Trả về danh sách vấn đề, mỗi dòng một vấn đề. Nếu không có, trả về "OK"."""

        response = self.llm.generate(prompt)

        if "ok" not in response.lower():
            for line in response.strip().split("\n"):
                if line.strip():
                    issues.append(ConsistencyIssue(
                        severity="high",
                        category="character",
                        description=line.strip(),
                        location=f"Chương {chapter}"
                    ))

        return issues

    def _check_plot_in_text(
        self,
        text: str,
        chapter: int
    ) -> List[ConsistencyIssue]:
        """Check plot consistency in text."""
        if not self.plot_manager:
            return []

        issues = []

        # Get plot context
        plot_context = self.plot_manager.get_plot_context()

        prompt = f"""Kiểm tra văn bản sau có nhất quán với cốt truyện không:

Cốt truyện hiện tại:
{plot_context}

Văn bản (Chương {chapter}):
{text[:2000]}

Kiểm tra:
1. Sự kiện có mâu thuẫn với plot points đã có không?
2. Foreshadowing có được giải quyết đúng không?
3. Plot arc có phát triển hợp lý không?
4. Có sự kiện nào bị trùng lặp không?

Trả về danh sách vấn đề, mỗi dòng một vấn đề. Nếu không có, trả về "OK"."""

        response = self.llm.generate(prompt)

        if "ok" not in response.lower():
            for line in response.strip().split("\n"):
                if line.strip():
                    issues.append(ConsistencyIssue(
                        severity="high",
                        category="plot",
                        description=line.strip(),
                        location=f"Chương {chapter}"
                    ))

        return issues

    def _check_timeline_in_text(
        self,
        text: str,
        chapter: int
    ) -> List[ConsistencyIssue]:
        """Check timeline consistency in text."""
        issues = []

        prompt = f"""Kiểm tra tính nhất quán thời gian trong văn bản sau:

Văn bản (Chương {chapter}):
{text[:2000]}

Kiểm tra:
1. Thời gian có bị nhảy cóc không?
2. Có mâu thuẫn về thứ tự sự kiện không?
3. Thời gian giữa các cảnh có hợp lý không?
4. Có đề cập đến thời gian mâu thuẫn với chương trước không?

Trả về danh sách vấn đề, mỗi dòng một vấn đề. Nếu không có, trả về "OK"."""

        response = self.llm.generate(prompt)

        if "ok" not in response.lower():
            for line in response.strip().split("\n"):
                if line.strip():
                    issues.append(ConsistencyIssue(
                        severity="medium",
                        category="timeline",
                        description=line.strip(),
                        location=f"Chương {chapter}"
                    ))

        return issues

    def _check_world_in_text(
        self,
        text: str,
        chapter: int
    ) -> List[ConsistencyIssue]:
        """Check world consistency in text."""
        issues = []

        world_context = self.world_builder.get_world_context()

        prompt = f"""Kiểm tra văn bản sau có nhất quán với thế giới truyện không:

Thế giới truyện:
{world_context}

Văn bản (Chương {chapter}):
{text[:2000]}

Kiểm tra:
1. Địa điểm có mô tả đúng không?
2. Quy tắc thế giới có bị vi phạm không?
3. Chi tiết văn hóa có nhất quán không?
4. Có mâu thuẫn với lore đã có không?

Trả về danh sách vấn đề, mỗi dòng một vấn đề. Nếu không có, trả về "OK"."""

        response = self.llm.generate(prompt)

        if "ok" not in response.lower():
            for line in response.strip().split("\n"):
                if line.strip():
                    issues.append(ConsistencyIssue(
                        severity="medium",
                        category="fact",
                        description=line.strip(),
                        location=f"Chương {chapter}"
                    ))

        return issues

    def get_consistency_report(self) -> str:
        """
        Get consistency report.

        Returns:
            Formatted consistency report
        """
        parts = ["=== BÁO CÁO TÍNH NHẤT QUÁN ===\n"]

        # Character issues
        if self.character_manager:
            char_issues = []
            for char_name in self.character_manager.list_characters():
                char_issues.extend(self.check_character_consistency(char_name))

            if char_issues:
                parts.append("NHÂN VẬT:")
                for issue in char_issues:
                    parts.append(f"  [{issue.severity.upper()}] {issue.description}")
                parts.append("")

        # Plot issues
        if self.plot_manager:
            plot_issues = self.check_plot_consistency()
            if plot_issues:
                parts.append("CỐT TRUYỆN:")
                for issue in plot_issues:
                    parts.append(f"  [{issue.severity.upper()}] {issue.description}")
                parts.append("")

        if len(parts) == 1:
            parts.append("Không phát hiện vấn đề nào!")

        return "\n".join(parts)

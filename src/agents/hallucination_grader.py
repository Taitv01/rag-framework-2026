"""
Hallucination Grader
====================

Verifies whether an answer is grounded in retrieved documents.

Features:
- Claim-level granularity (identifies specific unsupported claims)
- Structured output with Pydantic
- Configurable strictness threshold
- Safe regeneration with stricter prompting
- Bilingual support (Vietnamese/English)

Usage:
    from src.agents.hallucination_grader import HallucinationGrader

    grader = HallucinationGrader(llm)

    # Grade an answer
    grade = grader.grade(answer="Thạch Sanh giết đại bàng bằng cung tên.",
                         context="Thạch Sanh dùng cây đàn đánh đại bàng...")
    print(grade.is_grounded)        # False
    print(grade.unsupported_claims) # ["giết đại bàng bằng cung tên"]

    # Safe generation with verification
    answer, grade = grader.safe_generate(question, context)
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HallucinationGrade(BaseModel):
    """Structured output for hallucination grading."""

    is_grounded: bool = Field(
        description="True if ALL claims in the answer are supported by the context. "
                    "True nếu TẤT CẢ các thông tin trong câu trả lời được hỗ trợ bởi ngữ cảnh."
    )
    grounded_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Proportion of claims that are supported (0.0 to 1.0). "
                    "Tỷ lệ thông tin được hỗ trợ."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of specific claims in the answer NOT found in the context. "
                    "Danh sách thông tin KHÔNG tìm thấy trong ngữ cảnh."
    )
    supported_claims: List[str] = Field(
        default_factory=list,
        description="List of claims that ARE supported by the context. "
                    "Danh sách thông tin ĐƯỢC hỗ trợ bởi ngữ cảnh."
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of the grading decision. "
                    "Giải thích ngắn gọn về quyết định đánh giá."
    )


class HallucinationGrader:
    """
    Grades whether an answer is grounded in retrieved documents.

    Uses an LLM to verify each claim in the answer against the
    provided context/documents. Returns detailed grading with
    claim-level attribution.

    Example:
        grader = HallucinationGrader(llm)

        # Simple grading
        grade = grader.grade(answer, context)
        if not grade.is_grounded:
            print(f"Unsupported: {grade.unsupported_claims}")

        # Safe generation with auto-verification
        answer, grade = grader.safe_generate(question, context, max_retries=2)
    """

    def __init__(
        self,
        llm,
        grounded_threshold: float = 0.8,
        max_retries: int = 2,
    ):
        """
        Initialize hallucination grader.

        Args:
            llm: LLMManager instance
            grounded_threshold: Minimum grounded_score to accept an answer (0.0-1.0)
            max_retries: Max regeneration attempts if hallucination detected
        """
        self.llm = llm
        self.grounded_threshold = grounded_threshold
        self.max_retries = max_retries

    def grade(self, answer: str, context: str) -> HallucinationGrade:
        """
        Grade whether an answer is grounded in the provided context.

        Args:
            answer: The generated answer to verify
            context: The source context (retrieved documents)

        Returns:
            HallucinationGrade with detailed grounding assessment
        """
        prompt = f"""You are a fact-checker / Bạn là người kiểm tra sự kiện.
Verify whether the answer is fully supported by the provided context.
Kiểm tra xem câu trả lời có được hỗ trợ đầy đủ bởi ngữ cảnh không.

Context / Ngữ cảnh:
{context[:3000]}

Answer / Câu trả lời:
{answer[:2000]}

Analyze each claim in the answer:
Phân tích từng thông tin trong câu trả lời:
1. Is every claim supported by the context? / Mỗi thông tin có được hỗ trợ không?
2. Which specific claims are NOT in the context? / Thông tin nào KHÔNG có trong ngữ cảnh?
3. Which claims ARE in the context? / Thông tin nào CÓ trong ngữ cảnh?

Be strict: if a claim adds details not explicitly stated in the context, mark it as unsupported.
Hãy nghiêm ngặt: nếu thông tin thêm chi tiết không được nêu rõ trong ngữ cảnh, đánh dấu là không được hỗ trợ."""

        try:
            grade = self.llm.with_structured_output(HallucinationGrade).invoke(
                [{"role": "user", "content": prompt}]
            )
            return grade
        except Exception as e:
            logger.warning(f"Structured hallucination grading failed: {e}")
            # Fallback: try free-form parsing
            return self._grade_freeform(answer, context)

    def _grade_freeform(self, answer: str, context: str) -> HallucinationGrade:
        """Fallback grading using free-form LLM output."""
        try:
            prompt = f"""Verify if this answer is supported by the context.
Reply in this exact format:
GROUNDED: yes/no
SCORE: 0.0-1.0
UNSUPPORTED: claim1; claim2; ...
SUPPORTED: claim1; claim2; ...

Context: {context[:2000]}
Answer: {answer[:1000]}"""

            response = self.llm.generate(prompt).strip()

            # Parse
            is_grounded = "yes" in response.lower().split("\n")[0] if response else False
            score = 0.5
            unsupported = []
            supported = []

            for line in response.split("\n"):
                line = line.strip()
                if line.upper().startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.upper().startswith("UNSUPPORTED:"):
                    claims = line.split(":", 1)[1].strip()
                    unsupported = [c.strip() for c in claims.split(";") if c.strip()]
                elif line.upper().startswith("SUPPORTED:"):
                    claims = line.split(":", 1)[1].strip()
                    supported = [c.strip() for c in claims.split(";") if c.strip()]

            return HallucinationGrade(
                is_grounded=is_grounded,
                grounded_score=score,
                unsupported_claims=unsupported,
                supported_claims=supported,
                explanation="Graded via free-form parsing (fallback).",
            )
        except Exception as e:
            logger.warning(f"Free-form hallucination grading also failed: {e}")
            # Last resort: assume grounded (fail-open)
            return HallucinationGrade(
                is_grounded=True,
                grounded_score=0.5,
                explanation=f"Grading failed: {e}. Assuming grounded (fail-open).",
            )

    def grade_with_sources(
        self, answer: str, docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Grade answer with per-source attribution.

        Args:
            answer: The generated answer
            docs: List of source documents

        Returns:
            Dict with grade, per-source attribution, and overall assessment
        """
        # Build context with source labels
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", f"Document {i}")
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content[:500]}")

        context = "\n\n".join(context_parts)

        # Grade
        grade = self.grade(answer, context)

        # Build attribution
        attribution = {
            "grade": grade,
            "is_grounded": grade.is_grounded,
            "grounded_score": grade.grounded_score,
            "unsupported_claims": grade.unsupported_claims,
            "supported_claims": grade.supported_claims,
            "num_sources": len(docs),
            "sources_used": [
                {
                    "source": doc.metadata.get("source", f"Document {i}"),
                    "content_preview": doc.page_content[:200],
                }
                for i, doc in enumerate(docs, 1)
            ],
        }

        return attribution

    def safe_generate(
        self,
        question: str,
        context: str,
        max_retries: Optional[int] = None,
    ) -> Tuple[str, HallucinationGrade]:
        """
        Generate an answer and verify it's grounded.

        If hallucination is detected, retry with a stricter prompt.
        Returns the best answer found within the retry limit.

        Args:
            question: The user's question
            context: The source context (retrieved documents)
            max_retries: Max retry attempts (overrides default)

        Returns:
            Tuple of (answer, grade)
        """
        retries = max_retries if max_retries is not None else self.max_retries

        # Initial generation
        answer = self._generate_answer(question, context, strictness="normal")
        grade = self.grade(answer, context)

        # Retry loop
        attempt = 0
        while not grade.is_grounded and grade.grounded_score < self.grounded_threshold and attempt < retries:
            attempt += 1
            logger.info(
                f"Hallucination detected (score={grade.grounded_score:.2f}), "
                f"retry {attempt}/{retries}"
            )

            # Generate with stricter prompt
            stricter_context = context
            if grade.unsupported_claims:
                # Add explicit "do NOT include these claims" instruction
                avoid_claims = "\n".join(f"- {c}" for c in grade.unsupported_claims)
                stricter_context = f"{context}\n\nIMPORTANT / QUAN TRỌNG: Do NOT include these unverified claims / KHONG dua vao cac thong tin chua xac minh:\n{avoid_claims}"

            answer = self._generate_answer(question, stricter_context, strictness="strict")
            grade = self.grade(answer, stricter_context)

        if not grade.is_grounded:
            logger.warning(
                f"Answer still not fully grounded after {retries} retries "
                f"(score={grade.grounded_score:.2f})"
            )

        return answer, grade

    def _generate_answer(
        self, question: str, context: str, strictness: str = "normal"
    ) -> str:
        """Generate an answer with varying strictness levels."""
        if strictness == "strict":
            prompt = f"""You are a careful, accurate assistant / Bạn là trợ lý cẩn thận, chính xác.
Answer the question using ONLY the provided context. Do NOT add any information not in the context.
Trả lời câu hỏi CHỈ bằng ngữ cảnh được cung cấp. KHÔNG thêm thông tin không có trong ngữ cảnh.

If the context doesn't fully answer the question, say what you CAN confirm and note what is uncertain.
Nếu ngữ cảnh không trả lời đầy đủ, hãy nói những gì bạn CÓ THỂ xác nhận và ghi chú điều chưa chắc chắn.

Rules / Quy tắc:
1. NEVER invent or assume facts / KHÔNG bịa đặt hoặc suy đoán
2. Cite specific parts of the context / Trích dẫn cụ thể từ ngữ cảnh
3. Use hedging language for uncertain info / Ngôn ngữ thận trọng cho thông tin không chắc chắn
4. Answer in the same language as the question / Trả lời bằng ngôn ngữ của câu hỏi

Context / Ngữ cảnh:
{context[:3000]}

Question / Câu hỏi: {question}

Answer / Câu trả lời:"""
        else:
            prompt = f"""You are a helpful assistant / Bạn là trợ lý hữu ích.
Answer the question based on the provided context.
Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.

If the context doesn't contain the answer, say "I don't have enough information."
Nếu ngữ cảnh không chứa câu trả lời, hãy nói "Tôi không có đủ thông tin."

Context / Ngữ cảnh:
{context[:3000]}

Question / Câu hỏi: {question}

Answer / Câu trả lời:"""

        try:
            return self.llm.generate(prompt).strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {e}"

    def verify_answer(
        self,
        answer: str,
        docs: List[Document],
        regenerate_if_ungrounded: bool = False,
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Convenience method: verify an existing answer against documents.

        Args:
            answer: The answer to verify
            docs: Source documents
            regenerate_if_ungrounded: If True, regenerate answer when hallucination detected
            question: Original question (required if regenerate_if_ungrounded=True)

        Returns:
            Dict with answer, grade, and metadata
        """
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        grade = self.grade(answer, context)

        result = {
            "answer": answer,
            "is_grounded": grade.is_grounded,
            "grounded_score": grade.grounded_score,
            "unsupported_claims": grade.unsupported_claims,
            "supported_claims": grade.supported_claims,
            "explanation": grade.explanation,
        }

        if not grade.is_grounded and regenerate_if_ungrounded and question:
            new_answer, new_grade = self.safe_generate(question, context)
            result["regenerated_answer"] = new_answer
            result["regenerated_grade"] = new_grade
            result["regenerated"] = True
        else:
            result["regenerated"] = False

        return result

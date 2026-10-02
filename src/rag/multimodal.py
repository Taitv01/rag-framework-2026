"""Shared helpers for image/video-aware RAG queries."""


def build_multimodal_prompt(question: str, context: str = "") -> str:
    """Build a bilingual prompt that combines visual evidence with RAG context."""
    retrieved_context = context.strip() or "(Không có tài liệu truy xuất / No retrieved context)"
    return f"""Bạn là trợ lý RAG đa phương thức / You are a multimodal RAG assistant.

Phân tích ảnh hoặc video đính kèm và trả lời câu hỏi bằng Tiếng Việt, trừ khi người dùng yêu cầu ngôn ngữ khác.
Analyze the attached images or videos and answer in Vietnamese unless another language is requested.

Quy tắc / Rules:
1. Dùng bằng chứng trực quan từ media và ngữ cảnh truy xuất bên dưới.
2. Phân biệt rõ điều quan sát trực tiếp với suy luận; nói rõ khi không chắc chắn.
3. Khi dùng thông tin từ tài liệu, trích dẫn Source ID như [S1], [S2].
4. Không bịa chi tiết không xuất hiện trong media hoặc ngữ cảnh.

Ngữ cảnh truy xuất / Retrieved context:
{retrieved_context}

Câu hỏi / Question:
{question}"""

"""
Document cards
==============

One short LLM-written card per source document: what it is about, its main
characters or entities, key events, motifs and themes, setting.

Passages tell events ("người anh ... rơi xuống biển"); questions across a
collection ask about motifs ("những truyện nào có kẻ tham lam bị trừng
phạt?"). A passage search misses that gap, a card bridges it: cards are
searched to find which documents a question is about, then the best passage
of each of those documents is retrieved.

Cards cost one LLM call per document when it is indexed (not per chunk), and
are rebuilt only when the document's content changes.
"""

import hashlib
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

# Long documents are cut: the card is a summary, not a copy.
DEFAULT_CARD_MAX_CHARS = 20000

# Page- and chunk-level metadata a card (one per document) must not inherit.
_NOT_DOCUMENT_LEVEL = {"page", "start_index", "parent_id", "parent_text", "chunk_id", "relevance_score"}

CARD_PROMPT = """You write a document card for a search index / Bạn viết thẻ tóm tắt tài liệu cho chỉ mục tìm kiếm.
The card decides which documents a question is about, so it must name concretely what the document contains.
Thẻ dùng để xác định câu hỏi liên quan tới tài liệu nào, nên phải nêu cụ thể những gì tài liệu chứa.

Rules / Quy tắc:
1. Use ONLY the document below / Chỉ dùng nội dung tài liệu dưới đây
2. Write in the document's language / Viết bằng ngôn ngữ của tài liệu
3. At most 200 words, one line per field, no other text / Tối đa 200 từ, mỗi mục một dòng, không thêm gì khác:
Title / Tiêu đề:
Summary / Tóm tắt:
Characters and entities / Nhân vật, thực thể:
Key events / Sự kiện chính:
Motifs and themes / Mô-típ, chủ đề:
Setting / Bối cảnh:

Document / Tài liệu: {name}
{text}"""


def document_key(metadata: Dict[str, Any]) -> str:
    """Which source document a loaded page or chunk belongs to."""
    return str(
        metadata.get("document_id")
        or metadata.get("source")
        or metadata.get("file_name")
        or ""
    )


def group_by_document(docs: List[Document]) -> Dict[str, List[Document]]:
    """Loaded pages grouped by source document, in load order."""
    groups: Dict[str, List[Document]] = {}
    for doc in docs:
        key = document_key(doc.metadata or {})
        if key:
            groups.setdefault(key, []).append(doc)
    return groups


def document_text(pages: List[Document]) -> str:
    return "\n\n".join(page.page_content.strip() for page in pages if page.page_content.strip())


def content_hash(pages: List[Document]) -> str:
    """Fingerprint of a document's text: its card is rebuilt only when this changes."""
    return hashlib.sha256(document_text(pages).encode("utf-8")).hexdigest()


def card_id(key: str) -> str:
    """Stable store id of a document's card, so a new card replaces the old one."""
    return hashlib.sha1(f"document-card|{key}".encode("utf-8")).hexdigest()[:20]


def card_prompt(pages: List[Document], max_chars: int = DEFAULT_CARD_MAX_CHARS) -> str:
    """
    Prompt for one document's card.

    The document is named by its file name, never its path, so the same file
    gives the same prompt on every machine (and replays with AgentLLM).
    """
    metadata = pages[0].metadata or {}
    name = metadata.get("relative_source") or metadata.get("file_name") or "document"
    text = document_text(pages)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[...]"
    return CARD_PROMPT.format(name=name, text=text)


def card_document(pages: List[Document], card_text: str, model: Optional[str] = None) -> Document:
    """
    The card as a Document with its source's document-level metadata.

    Metadata filters (source, file name, tenant...) therefore apply to cards
    as they do to passages.
    """
    metadata = pages[0].metadata or {}
    kept = {name: value for name, value in metadata.items() if name not in _NOT_DOCUMENT_LEVEL}
    key = document_key(metadata)
    return Document(
        page_content=card_text.strip(),
        metadata={
            **kept,
            "document_card": True,
            "document_key": key,
            "chunk_id": card_id(key),
            "content_sha256": content_hash(pages),
            "card_model": model,
        },
    )

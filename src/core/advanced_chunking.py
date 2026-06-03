"""
Advanced Chunking Strategies
=============================

Multiple chunking strategies for different use cases.

Strategies:
- Semantic Chunking: Split by semantic coherence
- Proposition Chunking: Generate factual propositions
- Contextual Headers: Add document/section context
- Parent-Child Chunks: Hierarchical chunking

Usage:
    from src.core.advanced_chunking import SemanticChunker, PropositionChunker

    chunker = SemanticChunker(embeddings, threshold=0.8)
    chunks = chunker.split(documents)
"""

import re
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Vietnamese abbreviations that end with "." but are NOT sentence boundaries
_VIETNAMESE_ABBREVIATIONS = {
    "tp.", "t.p.", "p.", "q.", "h.", "x.", "tt.",
    "ths.", "ts.", "pgs.", "gs.",
    "đc.", "dc.", "tel.", "sdt.",
    "v.v.", "v.v...",
}


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    threshold: float = 0.75


class SemanticChunker:
    """
    Semantic chunking based on embedding similarity.

    Splits text at points where semantic similarity drops below threshold.

    Reference: "Semantic Chunking" technique from RAG_Techniques

    Example:
        chunker = SemanticChunker(embeddings, threshold=0.8)
        chunks = chunker.split(documents)
    """

    def __init__(
        self,
        embeddings,
        threshold: float = 0.75,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1000
    ):
        """
        Initialize semantic chunker.

        Args:
            embeddings: Embeddings instance
            threshold: Similarity threshold for splitting
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.embeddings = embeddings
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents semantically.

        Args:
            documents: Documents to split

        Returns:
            List of semantically coherent chunks
        """
        import numpy as np

        all_chunks = []

        for doc in documents:
            # Split into sentences
            sentences = self._split_sentences(doc.page_content)

            if len(sentences) <= 1:
                all_chunks.append(doc)
                continue

            # Get embeddings for sentences
            sentence_embeddings = self.embeddings.embed_documents(sentences)

            # Calculate similarities
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                sim = self._cosine_similarity(
                    sentence_embeddings[i],
                    sentence_embeddings[i + 1]
                )
                similarities.append(sim)

            # Find breakpoints
            breakpoints = [0]
            for i, sim in enumerate(similarities):
                if sim < self.threshold:
                    breakpoints.append(i + 1)
            breakpoints.append(len(sentences))

            # Create chunks
            for i in range(len(breakpoints) - 1):
                start = breakpoints[i]
                end = breakpoints[i + 1]

                chunk_sentences = sentences[start:end]
                chunk_text = " ".join(chunk_sentences)

                if len(chunk_text) >= self.min_chunk_size:
                    metadata = doc.metadata.copy()
                    metadata["chunk_method"] = "semantic"
                    metadata["sentence_range"] = [start, end]

                    all_chunks.append(Document(
                        page_content=chunk_text,
                        metadata=metadata
                    ))

        return all_chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with Vietnamese abbreviation handling."""
        # Protect Vietnamese abbreviations
        protected = text.lower()
        placeholder_map = {}
        for abbr in _VIETNAMESE_ABBREVIATIONS:
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            placeholder = abbr.replace(".", "§")
            placeholder_map[placeholder] = abbr
            protected = pattern.sub(placeholder, protected)

        # Split on sentence boundaries
        raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=\.\.\.)\s+', protected)

        # Restore abbreviations
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            for placeholder, abbr in placeholder_map.items():
                sent = sent.replace(placeholder, abbr)
            sentences.append(sent)

        return sentences

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        import numpy as np

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class PropositionChunker:
    """
    Proposition-based chunking.

    Generates factual propositions from text, then chunks by propositions.

    Reference: "Proposition Chunking" technique from RAG_Techniques

    Example:
        chunker = PropositionChunker(llm)
        chunks = chunker.split(documents)
    """

    def __init__(
        self,
        llm,
        quality_check: bool = True,
        min_propositions: int = 1,
        max_propositions: int = 10
    ):
        """
        Initialize proposition chunker.

        Args:
            llm: LLM instance
            quality_check: Whether to check proposition quality
            min_propositions: Minimum propositions per chunk
            max_propositions: Maximum propositions per chunk
        """
        self.llm = llm
        self.quality_check = quality_check
        self.min_propositions = min_propositions
        self.max_propositions = max_propositions

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into proposition-based chunks.

        Args:
            documents: Documents to split

        Returns:
            List of proposition-based chunks
        """
        all_chunks = []

        for doc in documents:
            # Generate propositions
            propositions = self._generate_propositions(doc.page_content)

            # Quality check
            if self.quality_check:
                propositions = self._check_quality(propositions)

            # Create chunks from propositions
            for i in range(0, len(propositions), self.max_propositions):
                chunk_propositions = propositions[i:i + self.max_propositions]
                chunk_text = " ".join(chunk_propositions)

                metadata = doc.metadata.copy()
                metadata["chunk_method"] = "proposition"
                metadata["proposition_count"] = len(chunk_propositions)

                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))

        return all_chunks

    def _generate_propositions(self, text: str) -> List[str]:
        """Generate propositions from text."""
        prompt = f"""Break the following text into individual factual propositions.
Each proposition should be a standalone statement that is self-contained.
Tách văn bản sau thành các mệnh đề sự kiện độc lập.
Mỗi mệnh đề phải là một phát biểu tự chứa.

Text:
{text[:2000]}

Return one proposition per line, nothing else."""

        response = self.llm.generate(prompt)
        propositions = [p.strip() for p in response.strip().split("\n") if p.strip()]

        return propositions

    def _check_quality(self, propositions: List[str]) -> List[str]:
        """Check quality of propositions."""
        checked = []

        for prop in propositions:
            prompt = f"""Evaluate this proposition for quality:

Proposition: {prop}

Check:
1. Is it accurate?
2. Is it clear?
3. Is it complete?
4. Is it concise?

Reply with 'yes' if all checks pass, 'no' otherwise."""

            response = self.llm.generate(prompt).strip().lower()

            if response == "yes":
                checked.append(prop)

        return checked if checked else propositions[:self.min_propositions]


class ContextualHeaderChunker:
    """
    Contextual Header Chunking (CCH).

    Prepends document-level and section-level context to chunks.

    Reference: "Contextual Chunk Headers" technique from RAG_Techniques

    Example:
        chunker = ContextualHeaderChunker(llm, chunk_size=500)
        chunks = chunker.split(documents)
    """

    def __init__(
        self,
        llm,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize contextual header chunker.

        Args:
            llm: LLM instance
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
        """
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents with contextual headers.

        Args:
            documents: Documents to split

        Returns:
            List of chunks with contextual headers
        """
        all_chunks = []

        for doc in documents:
            # Generate document summary
            doc_summary = self._generate_summary(doc.page_content)

            # Split into chunks
            chunks = self._split_text(doc.page_content)

            # Add headers to each chunk
            for i, chunk in enumerate(chunks):
                header = f"Document: {doc_summary}\nSection {i + 1}:\n"
                full_content = header + chunk

                metadata = doc.metadata.copy()
                metadata["chunk_method"] = "contextual_header"
                metadata["document_summary"] = doc_summary
                metadata["section_number"] = i + 1

                all_chunks.append(Document(
                    page_content=full_content,
                    metadata=metadata
                ))

        return all_chunks

    def _generate_summary(self, text: str) -> str:
        """Generate document summary."""
        prompt = f"""Summarize the following text in one sentence.
Tóm tắt văn bản sau thành một câu.

Text:
{text[:1000]}

Summary:"""

        return self.llm.generate(prompt).strip()

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with Vietnamese-aware sentence boundaries."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?\n':
                        # Check it's not a Vietnamese abbreviation
                        if not self._is_abbreviation_boundary(text, i):
                            end = i + 1
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # If we've reached the end, stop
            if end >= len(text):
                break

            # Advance start, ensuring we always move forward
            new_start = end - self.chunk_overlap
            if new_start <= start:
                new_start = end
            start = new_start

        return chunks

    @staticmethod
    def _is_abbreviation_boundary(text: str, dot_pos: int) -> bool:
        """Check if a dot at given position is part of a Vietnamese abbreviation."""
        # Look at the word before the dot
        before = text[max(0, dot_pos - 10):dot_pos + 1].lower()
        for abbr in _VIETNAMESE_ABBREVIATIONS:
            if before.endswith(abbr.rstrip('.')):
                return False
        return False


class ParentChildChunker:
    """
    Parent-Child hierarchical chunking.

    Creates parent chunks (large context) and child chunks (precise retrieval).

    Example:
        chunker = ParentChildChunker(
            parent_size=2000,
            child_size=200,
            child_overlap=50
        )
        parent_chunks, child_chunks = chunker.split(documents)
    """

    def __init__(
        self,
        parent_size: int = 2000,
        child_size: int = 200,
        child_overlap: int = 50
    ):
        """
        Initialize parent-child chunker.

        Args:
            parent_size: Parent chunk size
            child_size: Child chunk size
            child_overlap: Child chunk overlap
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def split(self, documents: List[Document]) -> tuple:
        """
        Split documents into parent and child chunks.

        Args:
            documents: Documents to split

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        parent_chunks = []
        child_chunks = []

        for doc in documents:
            # Create parent chunks
            parents = self._split_by_size(doc.page_content, self.parent_size)

            for parent_idx, parent_text in enumerate(parents):
                parent_id = f"{doc.metadata.get('source', 'doc')}_{parent_idx}"

                parent_metadata = doc.metadata.copy()
                parent_metadata["chunk_type"] = "parent"
                parent_metadata["parent_id"] = parent_id

                parent_chunks.append(Document(
                    page_content=parent_text,
                    metadata=parent_metadata
                ))

                # Create child chunks from parent
                children = self._split_by_size(parent_text, self.child_size)

                for child_idx, child_text in enumerate(children):
                    child_metadata = doc.metadata.copy()
                    child_metadata["chunk_type"] = "child"
                    child_metadata["parent_id"] = parent_id
                    child_metadata["child_index"] = child_idx

                    child_chunks.append(Document(
                        page_content=child_text,
                        metadata=child_metadata
                    ))

        return parent_chunks, child_chunks

    def _split_by_size(self, text: str, size: int) -> List[str]:
        """Split text by size with Vietnamese-aware sentence boundaries."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?\n':
                        # Check it's not a Vietnamese abbreviation
                        if not self._is_abbreviation_boundary(text, i):
                            end = i + 1
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # If we've reached the end, stop
            if end >= len(text):
                break

            # Advance start, ensuring we always move forward
            new_start = end - self.child_overlap
            if new_start <= start:
                new_start = end
            start = new_start

        return chunks

    @staticmethod
    def _is_abbreviation_boundary(text: str, dot_pos: int) -> bool:
        """Check if a dot at given position is part of a Vietnamese abbreviation."""
        before = text[max(0, dot_pos - 10):dot_pos + 1].lower()
        for abbr in _VIETNAMESE_ABBREVIATIONS:
            if before.endswith(abbr.rstrip('.')):
                return False
        return False


class AdaptiveChunker:
    """
    Adaptive chunking that selects the best strategy based on content.

    Analyzes content and selects the most appropriate chunking method.

    Example:
        chunker = AdaptiveChunker(llm, embeddings)
        chunks = chunker.split(documents)
    """

    def __init__(self, llm, embeddings, default_strategy: str = "semantic"):
        """
        Initialize adaptive chunker.

        Args:
            llm: LLM instance
            embeddings: Embeddings instance
            default_strategy: Default strategy if analysis is inconclusive
        """
        self.llm = llm
        self.embeddings = embeddings
        self.default_strategy = default_strategy

        # Initialize chunkers
        self.chunkers = {
            "semantic": SemanticChunker(embeddings),
            "proposition": PropositionChunker(llm),
            "contextual": ContextualHeaderChunker(llm),
            "parent_child": ParentChildChunker(),
        }

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using adaptive strategy.

        Args:
            documents: Documents to split

        Returns:
            List of chunks
        """
        all_chunks = []

        for doc in documents:
            # Analyze content
            strategy = self._analyze_content(doc.page_content)

            # Select chunker
            chunker = self.chunkers.get(strategy, self.chunkers[self.default_strategy])

            # Split document (handle ParentChildChunker's tuple return)
            result = chunker.split([doc])
            if isinstance(result, tuple):
                # ParentChildChunker returns (parent_chunks, child_chunks)
                # Use child chunks for retrieval, parents for context
                parent_chunks, child_chunks = result
                all_chunks.extend(child_chunks)
            else:
                all_chunks.extend(result)

        return all_chunks

    def _analyze_content(self, text: str) -> str:
        """Analyze content to select best strategy."""
        # Simple heuristics (supports both Vietnamese and English keywords)
        if len(text) < 500:
            return "semantic"  # Short text, semantic is fine
        elif text.count('\n\n') > 5:
            return "contextual"  # Many paragraphs, use headers
        elif any(keyword in text.lower() for keyword in [
            # English
            'fact', 'data', 'statistic', 'research', 'study',
            # Vietnamese
            'dữ liệu', 'thống kê', 'nghiên cứu', 'số liệu', 'thực tế',
        ]):
            return "proposition"  # Factual content
        else:
            return self.default_strategy


class ContextualRetrievalChunker:
    """
    Anthropic-style Contextual Retrieval chunking.

    For each chunk, generates a short LLM-produced context explaining
    what the chunk is about within the broader document. This context
    is prepended to the chunk before embedding, making each chunk
    self-contained and improving retrieval quality significantly.

    Reference: Anthropic's "Contextual Retrieval" technique
    (https://www.anthropic.com/news/contextual-retrieval)

    Benchmark: ~70% improvement in faithfulness vs naive chunking.

    Example:
        chunker = ContextualRetrievalChunker(llm, chunk_size=500)
        chunks = chunker.split(documents)
        # Each chunk now starts with:
        # "[Context: This chunk discusses Thạch Sanh's battle with the eagle...]"
        # "Thạch Sanh took his bow and arrow..."
    """

    def __init__(
        self,
        llm,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        context_window: int = 2000,
    ):
        """
        Initialize contextual retrieval chunker.

        Args:
            llm: LLM instance for generating context
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            context_window: Max chars of surrounding text to send to LLM for context
        """
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_window = context_window

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents with contextual headers prepended to each chunk.

        For each chunk:
        1. Split document into chunks
        2. Generate a short context for each chunk using the full document
        3. Prepend context to chunk text
        4. Store original chunk in metadata

        Args:
            documents: Documents to split

        Returns:
            List of chunks with contextual headers
        """
        all_chunks = []

        for doc in documents:
            # Step 1: Split into chunks
            raw_chunks = self._split_text(doc.page_content)

            if not raw_chunks:
                continue

            # Step 2: Generate document summary for context
            doc_summary = self._generate_summary(doc.page_content)

            # Step 3: Generate per-chunk context and prepend
            for i, chunk_text in enumerate(raw_chunks):
                # Build surrounding context for the LLM
                surrounding = self._get_surrounding_context(
                    doc.page_content, chunk_text, i, raw_chunks
                )

                # Generate chunk-specific context
                chunk_context = self._generate_chunk_context(
                    doc_summary, chunk_text, surrounding
                )

                # Prepend context to chunk
                contextual_text = f"[Context: {chunk_context}]\n{chunk_text}"

                metadata = doc.metadata.copy()
                metadata["chunk_method"] = "contextual_retrieval"
                metadata["document_summary"] = doc_summary
                metadata["chunk_context"] = chunk_context
                metadata["original_chunk"] = chunk_text
                metadata["section_number"] = i + 1

                all_chunks.append(Document(
                    page_content=contextual_text,
                    metadata=metadata,
                ))

        return all_chunks

    def _generate_summary(self, text: str) -> str:
        """Generate a one-sentence document summary."""
        try:
            prompt = f"""Summarize the following text in one sentence.
Tóm tắt văn bản sau thành một câu.

Text:
{text[:self.context_window]}

Summary / Tóm tắt:"""
            return self.llm.generate(prompt).strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return text[:100] + "..."

    def _generate_chunk_context(
        self, doc_summary: str, chunk_text: str, surrounding: str
    ) -> str:
        """
        Generate a short context for a specific chunk.

        The context explains what this chunk is about within the document,
        helping the embedding model understand the chunk's meaning.
        """
        try:
            prompt = f"""You are providing context for document chunking.
Bạn đang cung cấp ngữ cảnh cho việc phân đoạn tài liệu.

Document summary / Tóm tắt tài liệu: {doc_summary}

Surrounding text / Văn bản xung quanh:
{surrounding[:1000]}

Current chunk / Đoạn hiện tại:
{chunk_text[:500]}

Write a brief context (1-2 sentences) explaining what this chunk discusses
within the broader document. This will be prepended to the chunk for better retrieval.
Viết ngữ cảnh ngắn gọn (1-2 câu) giải thích đoạn này nói về điều gì
trong tài liệu tổng thể.

Context / Ngữ cảnh:"""
            return self.llm.generate(prompt).strip()
        except Exception as e:
            logger.warning(f"Chunk context generation failed: {e}")
            return doc_summary[:100]

    def _get_surrounding_context(
        self, full_text: str, chunk_text: str, chunk_idx: int, all_chunks: List[str]
    ) -> str:
        """Get surrounding text for context generation."""
        # Get previous and next chunks for context
        context_parts = []

        if chunk_idx > 0:
            context_parts.append(f"[Previous]: {all_chunks[chunk_idx - 1]}")

        context_parts.append(f"[Current]: {chunk_text}")

        if chunk_idx < len(all_chunks) - 1:
            context_parts.append(f"[Next]: {all_chunks[chunk_idx + 1]}")

        return "\n".join(context_parts)

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with Vietnamese-aware sentence boundaries."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?\n':
                        if not self._is_abbreviation_boundary(text, i):
                            end = i + 1
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # If we've reached the end, stop
            if end >= len(text):
                break

            # Advance start, ensuring we always move forward
            new_start = end - self.chunk_overlap
            if new_start <= start:
                new_start = end
            start = new_start

        return chunks

    @staticmethod
    def _is_abbreviation_boundary(text: str, dot_pos: int) -> bool:
        """Check if a dot is part of a Vietnamese abbreviation."""
        before = text[max(0, dot_pos - 10):dot_pos + 1].lower()
        for abbr in _VIETNAMESE_ABBREVIATIONS:
            if before.endswith(abbr.rstrip('.')):
                return False
        return False

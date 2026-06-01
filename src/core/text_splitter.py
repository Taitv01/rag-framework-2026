"""
Text Splitter
=============

Advanced text chunking strategies for RAG pipelines.

Supported strategies:
- Recursive character splitting (default)
- Sentence-aware splitting
- Semantic splitting (using embeddings)
- Custom delimiter splitting

Usage:
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


@dataclass
class SplitterConfig:
    """Configuration for text splitting."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    length_function: Callable[[str], int] = len
    separators: Optional[List[str]] = None
    keep_separator: bool = True
    add_start_index: bool = True


class TextSplitter:
    """
    Advanced text splitter with multiple chunking strategies.

    Supports:
    - Recursive character splitting (best for general text)
    - Token-based splitting (for precise token control)
    - Sentence-aware splitting (preserves sentence boundaries)

    Example:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)

        # Split documents
        chunks = splitter.split_documents(documents)

        # Use specific strategy
        chunks = splitter.split_by_tokens(documents, chunk_size=1000)

        # Split text directly
        texts = splitter.split_text("Your long text here...")
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        """
        Initialize text splitter.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            separators: Custom separators for splitting
            keep_separator: Whether to keep separators in chunks
        """
        self.config = SplitterConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
        )

        # Default separators for recursive splitting
        self._default_separators = ["\n\n", "\n", ". ", " ", ""]

    def split_documents(
        self,
        documents: List[Document],
        strategy: str = "recursive"
    ) -> List[Document]:
        """
        Split documents using specified strategy.

        Args:
            documents: List of Document objects
            strategy: Splitting strategy ('recursive', 'token', 'sentence')

        Returns:
            List of chunked Document objects
        """
        strategy_map = {
            "recursive": self._split_recursive,
            "token": self._split_by_tokens,
            "sentence": self._split_by_sentences,
        }

        if strategy not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(strategy_map.keys())}"
            )

        return strategy_map[strategy](documents)

    def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: str = "recursive"
    ) -> List[Document]:
        """
        Split text string into documents.

        Args:
            text: Text to split
            metadata: Metadata to attach to all chunks
            strategy: Splitting strategy

        Returns:
            List of Document objects
        """
        doc = Document(page_content=text, metadata=metadata or {})
        return self.split_documents([doc], strategy=strategy)

    def _split_recursive(self, documents: List[Document]) -> List[Document]:
        """Split using recursive character splitter."""
        separators = self.config.separators or self._default_separators

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self.config.length_function,
            separators=separators,
            keep_separator=self.config.keep_separator,
            add_start_index=self.config.add_start_index,
        )

        return splitter.split_documents(documents)

    def _split_by_tokens(self, documents: List[Document]) -> List[Document]:
        """Split by token count (uses tiktoken)."""
        splitter = TokenTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        return splitter.split_documents(documents)

    def _split_by_sentences(self, documents: List[Document]) -> List[Document]:
        """
        Split by sentences, preserving sentence boundaries.

        This is useful when you want each chunk to contain complete sentences.
        """
        import re

        # Sentence boundary pattern
        sentence_pattern = r'(?<=[.!?])\s+'

        chunks = []

        for doc in documents:
            sentences = re.split(sentence_pattern, doc.page_content)
            sentences = [s.strip() for s in sentences if s.strip()]

            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence)

                # If adding this sentence exceeds chunk size, save current chunk
                if current_length + sentence_length > self.config.chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata=doc.metadata.copy()
                    ))

                    # Keep overlap
                    overlap_chunk = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) > self.config.chunk_overlap:
                            break
                        overlap_chunk.insert(0, s)
                        overlap_length += len(s)

                    current_chunk = overlap_chunk
                    current_length = overlap_length

                current_chunk.append(sentence)
                current_length += sentence_length

            # Don't forget the last chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                ))

        return chunks

    def split_with_custom_function(
        self,
        documents: List[Document],
        split_func: Callable[[str], List[str]]
    ) -> List[Document]:
        """
        Split documents using a custom splitting function.

        Args:
            documents: List of Document objects
            split_func: Function that takes text and returns list of chunks

        Returns:
            List of chunked Document objects
        """
        chunks = []

        for doc in documents:
            text_chunks = split_func(doc.page_content)

            for chunk_text in text_chunks:
                if chunk_text.strip():
                    chunks.append(Document(
                        page_content=chunk_text.strip(),
                        metadata=doc.metadata.copy()
                    ))

        return chunks


class SemanticSplitter:
    """
    Semantic text splitter that uses embeddings to find natural breakpoints.

    This splitter finds the most semantically coherent places to split text,
    rather than splitting at fixed character counts.

    Note: Requires an embeddings model to be provided.
    """

    def __init__(
        self,
        embeddings,
        chunk_size: int = 500,
        breakpoint_threshold: float = 0.5,
    ):
        """
        Initialize semantic splitter.

        Args:
            embeddings: Embeddings model instance
            chunk_size: Target chunk size
            breakpoint_threshold: Threshold for semantic similarity breaks
        """
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.breakpoint_threshold = breakpoint_threshold

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using semantic similarity.

        Finds natural breakpoints where semantic similarity drops below threshold.
        """
        import numpy as np

        chunks = []

        for doc in documents:
            # First, split into sentences
            sentences = self._split_into_sentences(doc.page_content)

            if len(sentences) <= 1:
                chunks.append(doc)
                continue

            # Get embeddings for all sentences
            sentence_embeddings = self.embeddings.embed_documents(sentences)

            # Calculate similarities between consecutive sentences
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                sim = self._cosine_similarity(
                    sentence_embeddings[i],
                    sentence_embeddings[i + 1]
                )
                similarities.append(sim)

            # Find breakpoints where similarity drops below threshold
            breakpoints = []
            for i, sim in enumerate(similarities):
                if sim < self.breakpoint_threshold:
                    breakpoints.append(i + 1)

            # Split at breakpoints
            start = 0
            for bp in breakpoints:
                chunk_text = " ".join(sentences[start:bp])
                if chunk_text.strip():
                    chunks.append(Document(
                        page_content=chunk_text.strip(),
                        metadata=doc.metadata.copy()
                    ))
                start = bp

            # Last chunk
            if start < len(sentences):
                chunk_text = " ".join(sentences[start:])
                if chunk_text.strip():
                    chunks.append(Document(
                        page_content=chunk_text.strip(),
                        metadata=doc.metadata.copy()
                    ))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

"""
Metadata Enhancer
=================

LLM-based metadata enrichment for document chunks.

Automatically extracts and assigns structured metadata:
- Characters (nhân vật)
- Locations (địa điểm)
- Time period (thời gian)
- Topic (chủ đề)
- Sentiment (cảm xúc)

Usage:
    from src.core.metadata_enhancer import MetadataEnhancer

    enhancer = MetadataEnhancer(llm)
    enhanced_chunks = enhancer.enhance(chunks)

    # Now each chunk has metadata like:
    # chunk.metadata["characters"] = ["Thạch Sanh", "Lý Thông"]
    # chunk.metadata["locations"] = ["hang đại bàng"]
    # chunk.metadata["time_period"] = "xưa"
    # chunk.metadata["topic"] = "Thạch Sanh giết đại bàng cứu công chúa"
    # chunk.metadata["sentiment"] = "bi tráng"
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    """Structured metadata extracted from a chunk."""

    characters: List[str] = Field(
        default_factory=list,
        description="List of character/person names found in the text. "
                    "Danh sách tên nhân vật/người trong văn bản."
    )
    locations: List[str] = Field(
        default_factory=list,
        description="List of locations/places mentioned. "
                    "Danh sách địa điểm được đề cập."
    )
    time_period: str = Field(
        default="unknown",
        description="Time period or temporal reference (e.g., 'xưa', 'thời Lê', 'hiện đại'). "
                    "Thời kỳ hoặc mốc thời gian."
    )
    topic: str = Field(
        default="",
        description="One-line topic summary of the chunk. "
                    "Tóm tắt chủ đề đoạn văn trong một câu."
    )
    sentiment: str = Field(
        default="neutral",
        description="Overall emotional tone (e.g., 'bi tráng', 'hài hước', 'buồn', 'vui', 'hồi hộp'). "
                    "Tông cảm xúc chung."
    )


class MetadataEnhancer:
    """
    LLM-based metadata enrichment for document chunks.

    Extracts characters, locations, time period, topic, and sentiment
    from chunk text using an LLM, with optional NER fallback.

    Example:
        from src.core.metadata_enhancer import MetadataEnhancer

        enhancer = MetadataEnhancer(llm)
        enhanced = enhancer.enhance(chunks)

        # Filter by metadata
        results = retriever.search("query", filter={"characters": {"$contains": "Thạch Sanh"}})
    """

    def __init__(
        self,
        llm,
        vietnamese_processor=None,
        batch_size: int = 5,
        use_ner_fallback: bool = True,
    ):
        """
        Initialize metadata enhancer.

        Args:
            llm: LLMManager instance for metadata extraction
            vietnamese_processor: Optional VietnameseProcessor for NER fallback
            batch_size: Number of chunks per LLM call (reduces API costs)
            use_ner_fallback: Whether to use NER as fallback when LLM fails
        """
        self.llm = llm
        self.batch_size = batch_size
        self.use_ner_fallback = use_ner_fallback

        # Lazy-init VietnameseProcessor
        self._processor = vietnamese_processor
        self._processor_initialized = False

    def _get_processor(self):
        """Get or initialize VietnameseProcessor."""
        if not self._processor_initialized:
            if self._processor is None and self.use_ner_fallback:
                try:
                    from src.core.vietnamese_processor import VietnameseProcessor
                    self._processor = VietnameseProcessor()
                except Exception:
                    logger.debug("VietnameseProcessor not available for NER fallback")
            self._processor_initialized = True
        return self._processor

    def enhance(self, chunks: List[Document]) -> List[Document]:
        """
        Add metadata to each chunk.

        Processes chunks in batches for efficiency. Each chunk gets:
        - characters: List of character names
        - locations: List of locations
        - time_period: Time reference
        - topic: One-line topic
        - sentiment: Emotional tone

        Args:
            chunks: List of Document objects to enhance

        Returns:
            List of enhanced Documents (new objects, originals unchanged)
        """
        if not chunks:
            return []

        enhanced = []

        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]

            try:
                batch_results = self._enhance_batch(batch)
                enhanced.extend(batch_results)
            except Exception as e:
                logger.warning(f"Batch metadata extraction failed: {e}, using NER fallback")
                # Fallback: use NER for each chunk individually
                for chunk in batch:
                    enhanced.append(self._enhance_with_ner(chunk))

        return enhanced

    def _enhance_batch(self, chunks: List[Document]) -> List[Document]:
        """Enhance a batch of chunks with a single LLM call."""
        # Build batch prompt
        chunk_texts = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content[:1000]  # Truncate for prompt
            chunk_texts.append(f"--- Chunk {i + 1} ---\n{text}")

        combined_text = "\n\n".join(chunk_texts)

        prompt = f"""Analyze the following text chunks and extract metadata for each.
Phân tích các đoạn văn sau và trích xuất metadata cho mỗi đoạn.

For each chunk, extract:
- characters: List of character/person names (nhân vật)
- locations: List of locations/places (địa điểm)
- time_period: Time period or reference (thời gian)
- topic: One-line topic summary (chủ đề)
- sentiment: Emotional tone (cảm xúc)

Text chunks / Các đoạn văn:
{combined_text}

Return a JSON array with one object per chunk, in the same order.
Trả về mảng JSON với một đối tượng cho mỗi đoạn, theo thứ tự tương ứng.

Example format:
[
  {{"characters": ["Thạch Sanh", "Lý Thông"], "locations": ["hang đại bàng"], "time_period": "xưa", "topic": "Thạch Sanh giết đại bàng", "sentiment": "bi tráng"}},
  {{"characters": [], "locations": [], "time_period": "unknown", "topic": "...", "sentiment": "neutral"}}
]

Return ONLY the JSON array, nothing else.
Chỉ trả về mảng JSON, không thêm gì khác."""

        response = self.llm.generate(prompt)

        # Parse response
        metadata_list = self._parse_batch_response(response, len(chunks))

        # Apply metadata to chunks
        enhanced = []
        for i, chunk in enumerate(chunks):
            new_metadata = chunk.metadata.copy()
            meta = metadata_list[i] if i < len(metadata_list) else ChunkMetadata()

            new_metadata["characters"] = meta.characters
            new_metadata["locations"] = meta.locations
            new_metadata["time_period"] = meta.time_period
            new_metadata["topic"] = meta.topic
            new_metadata["sentiment"] = meta.sentiment
            new_metadata["metadata_enhanced"] = True

            enhanced.append(Document(
                page_content=chunk.page_content,
                metadata=new_metadata,
            ))

        return enhanced

    def _parse_batch_response(self, response: str, expected_count: int) -> List[ChunkMetadata]:
        """Parse LLM response into list of ChunkMetadata."""
        try:
            # Clean response
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("\n", 1)[1]
            if json_str.endswith("```"):
                json_str = json_str.rsplit("```", 1)[0]
            json_str = json_str.strip()

            data = json.loads(json_str)

            if not isinstance(data, list):
                data = [data]

            results = []
            for item in data:
                try:
                    meta = ChunkMetadata(**item)
                    results.append(meta)
                except Exception:
                    results.append(ChunkMetadata())

            # Pad with defaults if too few
            while len(results) < expected_count:
                results.append(ChunkMetadata())

            return results[:expected_count]

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse metadata JSON: {e}")
            return [ChunkMetadata() for _ in range(expected_count)]

    def _enhance_with_ner(self, chunk: Document) -> Document:
        """Enhance a single chunk using NER fallback."""
        new_metadata = chunk.metadata.copy()

        processor = self._get_processor()
        if processor:
            try:
                entities = processor.extract_entities(chunk.page_content)

                characters = []
                locations = []
                for ent in entities:
                    if ent.get("type") in ("PERSON", "PER"):
                        characters.append(ent["text"])
                    elif ent.get("type") in ("LOCATION", "LOC", "GPE"):
                        locations.append(ent["text"])

                new_metadata["characters"] = list(set(characters))
                new_metadata["locations"] = list(set(locations))
            except Exception as e:
                logger.debug(f"NER extraction failed: {e}")
                new_metadata["characters"] = []
                new_metadata["locations"] = []
        else:
            new_metadata["characters"] = []
            new_metadata["locations"] = []

        new_metadata["time_period"] = "unknown"
        new_metadata["topic"] = ""
        new_metadata["sentiment"] = "neutral"
        new_metadata["metadata_enhanced"] = False  # Indicates NER-only

        return Document(
            page_content=chunk.page_content,
            metadata=new_metadata,
        )

    def enhance_single(self, chunk: Document) -> Document:
        """
        Enhance a single chunk.

        Args:
            chunk: Document to enhance

        Returns:
            Enhanced Document with metadata
        """
        return self.enhance([chunk])[0]

"""
Vietnamese Text Processor
=========================

Vietnamese NLP preprocessing for the RAG pipeline.

Features:
- Word segmentation (critical for Vietnamese - monosyllabic language)
- Sentence splitting (handles Vietnamese abbreviations)
- Named Entity Recognition for fairy tale characters
- Language detection (Vietnamese vs English)
- Text normalization (diacritics, Unicode)

Requires:
    pip install underthesea pyvi

Usage:
    processor = VietnameseProcessor()

    # Word segmentation
    tokens = processor.tokenize("Thạch Sanh đánh đại bàng")
    # -> ["Thạch_Sanh", "đánh", "đại_bàng"]

    # Sentence splitting
    sentences = processor.split_sentences("Thạch Sanh đánh đại bàng. Lý Thông lừa đảo.")

    # Language detection
    lang = processor.detect_language("Thạch Sanh là ai?")
    # -> "vi"
"""

import re
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Vietnamese abbreviations that end with "." but are NOT sentence boundaries
VIETNAMESE_ABBREVIATIONS = {
    "tp.", "t.p.", "p.", "q.", "h.", "x.", "tt.",  # Địa danh
    "ths.", "ts.", "pgs.", "gs.",                    # Học hàm
    "đc.", "dc.", "tel.", "sdt.",                    # Thông tin liên lạc
    "ko.", "kh.", "k.",                              # Viết tắt thông dụng
    "v.v.", "v.v...", "v..v.",                       # etc.
    "ts.", "đt.", "cn.", "t2.", "t3.", "t4.",        # Thời gian
    "t5.", "t6.", "t7.", "cn.",
}


class VietnameseProcessor:
    """
    Vietnamese text preprocessing pipeline.

    Provides word segmentation, sentence splitting, NER, and language detection
    specifically optimized for Vietnamese fairy tale content.
    """

    def __init__(self, use_underthesea: bool = True):
        """
        Initialize Vietnamese processor.

        Args:
            use_underthesea: If True, use underthesea (preferred).
                           If False, fall back to pyvi.
        """
        self._backend = None
        self._use_underthesea = use_underthesea
        self._init_backend()

    def _init_backend(self):
        """Initialize the NLP backend."""
        if self._use_underthesea:
            try:
                import underthesea
                self._backend = "underthesea"
                logger.info("Vietnamese processor initialized with underthesea")
                return
            except ImportError:
                logger.warning("underthesea not available, falling back to pyvi")

        try:
            import pyvi
            self._backend = "pyvi"
            logger.info("Vietnamese processor initialized with pyvi")
        except ImportError:
            logger.warning(
                "Neither underthesea nor pyvi available. "
                "Vietnamese processing will use basic regex. "
                "Install with: pip install underthesea pyvi"
            )
            self._backend = "regex"

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text with word segmentation.

        Vietnamese is monosyllabic but compound words span multiple syllables.
        Example: "Thạch Sanh" is one word, "đại bàng" is one word.

        Args:
            text: Input text

        Returns:
            List of word tokens
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        if self._backend == "underthesea":
            try:
                from underthesea import word_tokenize
                tokens = word_tokenize(text, format="text")
                # underthesea joins compound words with underscore
                return [t.strip() for t in tokens.split() if t.strip()]
            except Exception as e:
                logger.warning(f"underthesea tokenization failed: {e}")

        if self._backend == "pyvi":
            try:
                from pyvi import ViTokenizer
                tokens = ViTokenizer.tokenize(text)
                return [t.strip() for t in tokens.split() if t.strip()]
            except Exception as e:
                logger.warning(f"pyvi tokenization failed: {e}")

        # Fallback: basic regex (no word segmentation)
        return re.findall(r'\S+', text.lower())

    def split_sentences(self, text: str) -> List[str]:
        """
        Split Vietnamese text into sentences.

        Handles Vietnamese-specific abbreviations (TP., P., Q., ThS., TS., etc.)
        that end with "." but are NOT sentence boundaries.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # Protect abbreviations by replacing "." with a placeholder
        protected = text.lower()
        for abbr in VIETNAMESE_ABBREVIATIONS:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            protected = pattern.sub(abbr.replace(".", "§"), protected)

        # Split on sentence boundaries
        # Handles: . ! ? followed by whitespace or end of string
        # Also handles ellipsis (...) as a single boundary
        raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=\.\.\.)\s+', protected)

        # Restore abbreviations and clean up
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Restore dots in abbreviations
            for abbr in VIETNAMESE_ABBREVIATIONS:
                sent = sent.replace(abbr.replace(".", "§"), abbr)
            # Get the original case version from the input
            sentences.append(sent)

        # If we lost case info, try to recover from original text
        if sentences and text[0].isupper():
            result = []
            remaining = text
            for sent in sentences:
                # Find this sentence in the original text (case-insensitive)
                idx = remaining.lower().find(sent.lower())
                if idx >= 0:
                    result.append(remaining[idx:idx + len(sent)])
                    remaining = remaining[idx + len(sent):]
                else:
                    result.append(sent)
            return [s for s in result if s.strip()]

        return sentences

    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.

        Args:
            text: Input text

        Returns:
            Language code: "vi" for Vietnamese, "en" for English, "mixed" for both
        """
        if not text:
            return "unknown"

        # Vietnamese diacritics detection
        vi_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
        vi_char_count = sum(1 for c in text.lower() if c in vi_chars)
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha == 0:
            return "unknown"

        vi_ratio = vi_char_count / total_alpha

        if vi_ratio > 0.05:
            return "vi"
        elif vi_ratio > 0.01:
            return "mixed"
        else:
            return "en"

    def preprocess(self, text: str) -> str:
        """
        Preprocess Vietnamese text for embedding/analysis.

        Steps:
        1. Normalize Unicode
        2. Normalize whitespace
        3. Word segmentation (join compound words)

        Args:
            text: Raw input text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Normalize Unicode (NFC form - composed characters)
        import unicodedata
        text = unicodedata.normalize("NFC", text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Word segmentation for better embedding quality
        if self._backend in ("underthesea", "pyvi"):
            tokens = self.tokenize(text)
            # Join with space (compound words already have underscore)
            text = " ".join(tokens)

        return text

    def extract_entities(self, text: str) -> List[dict]:
        """
        Extract named entities from Vietnamese text.

        Optimized for fairy tale content: characters, locations, magical objects.

        Args:
            text: Input text

        Returns:
            List of entity dicts with keys: text, type, start, end
        """
        if not text:
            return []

        if self._backend == "underthesea":
            try:
                from underthesea import ner
                entities = ner(text)
                result = []
                for ent in entities:
                    if hasattr(ent, 'text') and hasattr(ent, 'type'):
                        result.append({
                            "text": ent.text,
                            "type": ent.type,
                        })
                    elif isinstance(ent, (list, tuple)) and len(ent) >= 4:
                        # underthesea returns (word, pos, chunk, ner_tag)
                        if ent[3] != 'O':  # NER tag is not 'O' (Outside)
                            result.append({
                                "text": ent[0],
                                "type": ent[3],
                            })
                return result
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")

        # Fallback: basic pattern matching for Vietnamese names
        return self._extract_fairy_tale_entities(text)

    def _extract_fairy_tale_entities(self, text: str) -> List[dict]:
        """
        Basic entity extraction for fairy tale content using patterns.

        Vietnamese fairy tale names often follow patterns:
        - Single syllable: Tấm, Cám, Sọ Dừa
        - Two syllables: Thạch Sanh, Thánh Gióng, Sơn Tinh
        - With title: Bà Chúa, Ông Tiên, Cô Gái
        """
        entities = []

        # Common fairy tale character patterns
        name_patterns = [
            r'(?:Thạch Sanh|Thánh Gióng|Sơn Tinh|Thủy Tinh|Tấm Cám|Sọ Dừa)',
            r'(?:Bà\s+\w+|Ông\s+\w+|Cô\s+\w+|Chú\s+\w+|Cụ\s+\w+)',
            r'(?:Công chúa|Hoàng tử|Vua|Nữ hoàng)(?:\s+\w+)?',
            r'(?:Lý Thông|Mai An Tiêm|Ngư Ông|Hai Bà Trưng)',
        ]

        for pattern in name_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "type": "PERSON",
                    "start": match.start(),
                    "end": match.end(),
                })

        # Location patterns
        location_patterns = [
            r'(?:Hà Nội|Huế|Đà Lạt|Sài Gòn|Cố Đô)',
            r'(?:Lạc Long|Âu Cơ|Bách Việt)',
            r'(?:hang\s+\w+|núi\s+\w+|sông\s+\w+|biển\s+\w+|làng\s+\w+)',
        ]

        for pattern in location_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "type": "LOCATION",
                    "start": match.start(),
                    "end": match.end(),
                })

        return entities


# Module-level singleton for convenience
_default_processor: Optional[VietnameseProcessor] = None


def get_vietnamese_processor() -> VietnameseProcessor:
    """Get or create the default Vietnamese processor singleton."""
    global _default_processor
    if _default_processor is None:
        _default_processor = VietnameseProcessor()
    return _default_processor

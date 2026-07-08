"""Cross-story RAG utilities for Vietnamese fairy tale collections."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from langchain_core.documents import Document


DEFAULT_MOTIF_KEYWORDS: Dict[str, Sequence[str]] = {
    "magical_object": ("magic", "magical", "than", "phep", "bua", "dan", "guom", "kiem", "ngoc", "noi com", "cay tre"),
    "talking_animal": ("talking animal", "biet noi", "con vat", "chim", "ca", "ran", "ho", "trau", "ngua", "dai bang"),
    "orphan_hero": ("orphan", "mo coi", "ngheo", "cut cui", "song mot minh", "nguoi em"),
    "betrayal": ("betrayal", "phan boi", "lua", "lua doi", "cuop cong", "ham hai", "do ky"),
    "transformation": ("transformation", "bien thanh", "hoa thanh", "lot xac", "than hoa", "bien hoa"),
    "reward_punishment": ("reward", "punishment", "thuong", "phat", "bao ung", "trung tri", "duoc thuong"),
    "hero_quest": ("hero", "quest", "chien dau", "giai cuu", "cuu cong chua", "diet", "vuot qua", "thu thach", "lap cong"),
    "kingdom_marriage": ("king", "kingdom", "vua", "hoang tu", "cong chua", "ket hon", "lay vo", "lam vua"),
    "sibling_rivalry": ("sibling", "anh em", "chi em", "nguoi anh", "nguoi em", "tam cam", "ganh ghe"),
    "nature_conflict": ("son tinh", "thuy tinh", "lua lut", "nui", "song", "bien", "mua", "han han"),
}

DEFAULT_MORAL_KEYWORDS: Dict[str, Sequence[str]] = {
    "kindness": ("kindness", "nhan hau", "tot bung", "giup do", "thuong nguoi"),
    "courage": ("courage", "dung cam", "gan da", "chien dau", "vuot kho"),
    "honesty": ("honesty", "trung thuc", "that tha", "khong noi doi"),
    "justice": ("justice", "cong ly", "chinh nghia", "bao ung", "cai thien thang cai ac"),
    "loyalty": ("loyalty", "trung thanh", "giu loi hua", "biet on"),
    "filial_piety": ("filial", "hieu thao", "cha me", "phu mau"),
    "greed_warning": ("greed", "tham lam", "ich ky", "do ky", "bi phat"),
    "diligence": ("diligence", "cham chi", "can cu", "lao dong"),
}

STOPWORD_NAMES = {"Ngay", "Xua", "Mot", "Trong", "Sau", "Khi", "Vi", "The", "Nhung", "Roi", "Den", "Va", "Cua", "Voi", "La", "Co", "Cau", "Chuyen"}


def strip_diacritics(value: str) -> str:
    value = value.replace("\u0111", "d").replace("\u0110", "D")
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_for_match(value: str) -> str:
    value = strip_diacritics(unicodedata.normalize("NFC", value or "")).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize_for_match(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_for_match(value))


def stable_story_id(title: str, text: str) -> str:
    digest = hashlib.sha1(f"{title}\n{text}".encode("utf-8", errors="replace")).hexdigest()
    return digest[:12]


def unique_keep_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip()
        key = normalize_for_match(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


@dataclass
class StoryRecord:
    """Structured representation of one story in a cross-story corpus."""

    story_id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    characters: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    motifs: List[str] = field(default_factory=list)
    morals: List[str] = field(default_factory=list)

    def to_document(self) -> Document:
        metadata = dict(self.metadata)
        metadata.update({
            "story_id": self.story_id,
            "title": self.title,
            "characters": list(self.characters),
            "locations": list(self.locations),
            "motifs": list(self.motifs),
            "morals": list(self.morals),
        })
        return Document(page_content=self.text, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "text": self.text,
            "metadata": dict(self.metadata),
            "characters": list(self.characters),
            "locations": list(self.locations),
            "motifs": list(self.motifs),
            "morals": list(self.morals),
        }


class CrossStoryRAG:
    """Deterministic story-level retrieval for Vietnamese fairy tale corpora."""

    def __init__(
        self,
        vietnamese_processor: Optional[Any] = None,
        motif_keywords: Optional[Mapping[str, Sequence[str]]] = None,
        moral_keywords: Optional[Mapping[str, Sequence[str]]] = None,
    ):
        self._processor = vietnamese_processor
        self.motif_keywords = dict(motif_keywords or DEFAULT_MOTIF_KEYWORDS)
        self.moral_keywords = dict(moral_keywords or DEFAULT_MORAL_KEYWORDS)
        self._stories: Dict[str, StoryRecord] = {}

    @property
    def stories(self) -> List[StoryRecord]:
        return list(self._stories.values())

    @property
    def count(self) -> int:
        return len(self._stories)

    def add_story(self, title: str, text: str, metadata: Optional[Mapping[str, Any]] = None, story_id: Optional[str] = None) -> StoryRecord:
        metadata_dict = dict(metadata or {})
        normalized_text = self._normalize_text(text)
        story_id = story_id or metadata_dict.get("story_id") or stable_story_id(title, normalized_text)
        record = StoryRecord(
            story_id=str(story_id),
            title=title.strip() or str(story_id),
            text=normalized_text,
            metadata=metadata_dict,
            characters=unique_keep_order(metadata_dict.get("characters", []) or self._extract_characters(title, normalized_text)),
            locations=unique_keep_order(metadata_dict.get("locations", []) or self._extract_locations(normalized_text)),
            motifs=unique_keep_order(metadata_dict.get("motifs", []) or self._detect_labels(normalized_text, self.motif_keywords)),
            morals=unique_keep_order(metadata_dict.get("morals", []) or self._detect_labels(normalized_text, self.moral_keywords)),
        )
        self._stories[record.story_id] = record
        return record

    def add_stories(self, stories: Iterable[Mapping[str, Any]]) -> List[StoryRecord]:
        return [self.add_story(str(item.get("title", "")), str(item.get("text", "")), item.get("metadata") or {}, item.get("story_id")) for item in stories]

    def add_documents(self, documents: Iterable[Document]) -> List[StoryRecord]:
        records = []
        for index, document in enumerate(documents, start=1):
            metadata = dict(document.metadata or {})
            title = metadata.get("title") or metadata.get("source") or f"story_{index}"
            records.append(self.add_story(title=title, text=document.page_content, metadata=metadata))
        return records
    def find_motifs(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        target_labels = self._query_labels(query, self.motif_keywords)
        results = []
        for story in self.stories:
            matched = [label for label in story.motifs if label in target_labels]
            score = self._story_score(query, story, fields=("title", "text", "motifs"))
            if matched:
                score += 2.0 + len(matched)
            if score > min_score:
                results.append(self._result(story, score, matched_motifs=matched))
        return self._rank(results, top_k)

    def find_moral_patterns(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        target_labels = self._query_labels(query, self.moral_keywords)
        results = []
        for story in self.stories:
            matched = [label for label in story.morals if label in target_labels]
            score = self._story_score(query, story, fields=("title", "text", "morals"))
            if matched:
                score += 2.0 + len(matched)
            if score > min_score:
                item = self._result(story, score)
                item["matched_morals"] = matched
                results.append(item)
        return self._rank(results, top_k)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        results = []
        for story in self.stories:
            score = self._story_score(query, story, fields=("title", "text", "motifs", "morals"))
            if score > min_score:
                results.append(self._result(story, score))
        return self._rank(results, top_k)

    def compare_characters(self, character_a: str, character_b: Optional[str] = None) -> Dict[str, Any]:
        a_stories = self._stories_for_character(character_a)
        b_stories = self._stories_for_character(character_b) if character_b else []
        result: Dict[str, Any] = {
            "character_a": character_a,
            "character_b": character_b,
            "character_a_stories": [self._story_summary(story) for story in a_stories],
            "character_b_stories": [self._story_summary(story) for story in b_stories],
        }
        if not character_b:
            co_characters: List[str] = []
            for story in a_stories:
                co_characters.extend(character for character in story.characters if normalize_for_match(character) != normalize_for_match(character_a))
            result["co_characters"] = unique_keep_order(co_characters)
            result["motifs"] = unique_keep_order(label for story in a_stories for label in story.motifs)
            result["morals"] = unique_keep_order(label for story in a_stories for label in story.morals)
            return result

        a_ids = {story.story_id for story in a_stories}
        b_ids = {story.story_id for story in b_stories}
        shared = [story for story in self.stories if story.story_id in (a_ids & b_ids)]
        a_motifs = set(label for story in a_stories for label in story.motifs)
        b_motifs = set(label for story in b_stories for label in story.motifs)
        a_morals = set(label for story in a_stories for label in story.morals)
        b_morals = set(label for story in b_stories for label in story.morals)
        result.update({
            "shared_stories": [self._story_summary(story) for story in shared],
            "shared_motifs": sorted(a_motifs & b_motifs),
            "only_a_motifs": sorted(a_motifs - b_motifs),
            "only_b_motifs": sorted(b_motifs - a_motifs),
            "shared_morals": sorted(a_morals & b_morals),
            "only_a_morals": sorted(a_morals - b_morals),
            "only_b_morals": sorted(b_morals - a_morals),
        })
        return result

    def as_documents(self) -> List[Document]:
        return [story.to_document() for story in self.stories]

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text or "")
        return re.sub(r"\s+", " ", text).strip()

    def _extract_characters(self, title: str, text: str) -> List[str]:
        values: List[str] = []
        values.extend(self._entities_by_type(text, {"PERSON", "PER", "B-PER", "I-PER"}))
        pattern = re.compile(r"\b[A-Z\u00C0-\u1EF9\u0110][\w\u00C0-\u1EF9\u0111]+(?:\s+[A-Z\u00C0-\u1EF9\u0110][\w\u00C0-\u1EF9\u0111]+){0,3}")
        values.extend(match.group(0) for match in pattern.finditer(title))
        values.extend(match.group(0) for match in pattern.finditer(text[:3000]))
        return unique_keep_order(value for value in values if value.split()[0] not in STOPWORD_NAMES and len(value) > 1)

    def _extract_locations(self, text: str) -> List[str]:
        values = self._entities_by_type(text, {"LOCATION", "LOC", "B-LOC", "I-LOC"})
        normalized = strip_diacritics(text)
        pattern = re.compile(r"\b(?:nui|song|bien|hang|lang|kinh do|vuong quoc)\s+[a-zA-Z\u00C0-\u1EF9\u0110\u0111]+", flags=re.IGNORECASE)
        values.extend(match.group(0) for match in pattern.finditer(normalized))
        return unique_keep_order(values)

    def _entities_by_type(self, text: str, accepted_types: Set[str]) -> List[str]:
        if self._processor is None:
            return []
        try:
            entities = self._processor.extract_entities(text)
        except Exception:
            return []
        values = []
        for entity in entities or []:
            entity_type = str(entity.get("type", "")).upper()
            if entity_type in accepted_types:
                values.append(str(entity.get("text", "")))
        return values

    def _detect_labels(self, text: str, catalog: Mapping[str, Sequence[str]]) -> List[str]:
        normalized = normalize_for_match(text)
        return [label for label, keywords in catalog.items() if any(normalize_for_match(keyword) in normalized for keyword in keywords)]

    def _query_labels(self, query: str, catalog: Mapping[str, Sequence[str]]) -> Set[str]:
        normalized = normalize_for_match(query)
        labels: Set[str] = set()
        for label, keywords in catalog.items():
            label_text = normalize_for_match(label.replace("_", " "))
            if label_text and label_text in normalized:
                labels.add(label)
                continue
            if any(normalize_for_match(keyword) in normalized for keyword in keywords):
                labels.add(label)
        return labels

    def _story_score(self, query: str, story: StoryRecord, fields: Sequence[str]) -> float:
        query_tokens = set(tokenize_for_match(query))
        if not query_tokens:
            return 0.0
        parts: List[str] = []
        if "title" in fields:
            parts.append(story.title)
        if "text" in fields:
            parts.append(story.text)
        if "motifs" in fields:
            parts.extend(story.motifs)
        if "morals" in fields:
            parts.extend(story.morals)
        haystack = " ".join(parts)
        haystack_tokens = set(tokenize_for_match(haystack))
        overlap = len(query_tokens & haystack_tokens)
        coverage = overlap / max(len(query_tokens), 1)
        exact_bonus = 1.0 if normalize_for_match(query) in normalize_for_match(haystack) else 0.0
        title_bonus = 0.5 if query_tokens & set(tokenize_for_match(story.title)) else 0.0
        return overlap + coverage + exact_bonus + title_bonus

    def _stories_for_character(self, character: Optional[str]) -> List[StoryRecord]:
        if not character:
            return []
        target = normalize_for_match(character)
        result = []
        for story in self.stories:
            character_keys = {normalize_for_match(value) for value in story.characters}
            if target in character_keys or target in normalize_for_match(story.text):
                result.append(story)
        return result

    def _story_summary(self, story: StoryRecord) -> Dict[str, Any]:
        return {
            "story_id": story.story_id,
            "title": story.title,
            "characters": list(story.characters),
            "motifs": list(story.motifs),
            "morals": list(story.morals),
        }

    def _result(self, story: StoryRecord, score: float, matched_motifs: Optional[List[str]] = None) -> Dict[str, Any]:
        item = self._story_summary(story)
        item.update({
            "score": round(float(score), 6),
            "locations": list(story.locations),
            "metadata": dict(story.metadata),
            "excerpt": self._excerpt(story.text),
        })
        if matched_motifs is not None:
            item["matched_motifs"] = matched_motifs
        return item

    def _excerpt(self, text: str, max_chars: int = 240) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "..."

    def _rank(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        results.sort(key=lambda item: (-item["score"], item["title"], item["story_id"]))
        return results[: max(top_k, 0)]

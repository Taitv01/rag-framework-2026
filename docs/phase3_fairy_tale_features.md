# Phase 3 Fairy Tale Features

This project now includes deterministic Vietnamese fairy tale utilities that can
run without an LLM, embedding model, or external API key.

## CrossStoryRAG

`src.rag.CrossStoryRAG` indexes complete stories and supports collection-level
analysis before deeper RAG retrieval:

- `find_motifs(query, top_k=5)`: find stories by motif, such as talking animals,
  magical objects, betrayal, transformation, hero quests, or reward/punishment.
- `compare_characters(character_a, character_b=None)`: inspect appearances,
  co-characters, shared stories, shared motifs, and motif differences.
- `find_moral_patterns(query, top_k=5)`: find stories by lesson/moral pattern,
  such as courage, kindness, justice, greed warning, or filial piety.
- `as_documents()`: convert indexed stories into LangChain `Document` objects.

Example:

```python
from src.rag import CrossStoryRAG

rag = CrossStoryRAG()
rag.add_story(
    "Thach Sanh",
    "Thach Sanh dung cay dan than va cuu cong chua khoi dai bang biet noi.",
    metadata={"characters": ["Thach Sanh", "Cong chua"]},
)

matches = rag.find_motifs("con vat biet noi")
comparison = rag.compare_characters("Thach Sanh", "Cong chua")
```

## FairyTaleDatasetBuilder

`src.data.FairyTaleDatasetBuilder` builds normalized, labeled records from raw
texts or directories:

- Unicode and whitespace normalization
- Stable `story_id` generation
- Character, location, motif, and moral labels
- JSON/JSONL export
- LangChain `Document` conversion

Example:

```python
from src.data import FairyTaleDatasetBuilder

builder = FairyTaleDatasetBuilder()
records = builder.build_from_directory("stories")
builder.export_jsonl(records, "dataset/fairy_tales.jsonl")
documents = builder.to_documents(records)
```

## Design Notes

These modules are intentionally deterministic. They provide high-signal labels
and story candidates before the expensive steps of embedding retrieval,
reranking, graph extraction, or LLM generation.

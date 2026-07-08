"""Tests for Vietnamese fairy tale Phase 3 modules."""

import json

from langchain_core.documents import Document


def test_cross_story_finds_talking_animal_motif():
    from src.rag import CrossStoryRAG

    rag = CrossStoryRAG()
    rag.add_story(
        "Thach Sanh",
        "Thach Sanh dung cay dan than va cuu cong chua khoi dai bang biet noi.",
        metadata={"characters": ["Thach Sanh", "Cong chua"]},
    )
    rag.add_story(
        "Tam Cam",
        "Tam bi me ke va Cam ham hai nhung cuoi cung cai thien thang cai ac.",
        metadata={"characters": ["Tam", "Cam"]},
    )

    results = rag.find_motifs("con vat biet noi", top_k=1)

    assert len(results) == 1
    assert results[0]["title"] == "Thach Sanh"
    assert "talking_animal" in results[0]["matched_motifs"]


def test_cross_story_compare_characters():
    from src.rag.cross_story_rag import CrossStoryRAG

    rag = CrossStoryRAG()
    rag.add_story(
        "Story A",
        "Thach Sanh chien dau va giai cuu cong chua.",
        metadata={"characters": ["Thach Sanh", "Cong chua"], "motifs": ["hero_quest"]},
    )
    rag.add_story(
        "Story B",
        "Ly Thong lua doi va cuop cong cua Thach Sanh.",
        metadata={"characters": ["Ly Thong", "Thach Sanh"], "motifs": ["betrayal"]},
    )

    comparison = rag.compare_characters("Thach Sanh", "Ly Thong")

    assert len(comparison["character_a_stories"]) == 2
    assert len(comparison["character_b_stories"]) == 1
    assert comparison["shared_stories"][0]["title"] == "Story B"
    assert "hero_quest" in comparison["only_a_motifs"]
    assert "betrayal" in comparison["shared_motifs"] or "betrayal" in comparison["only_b_motifs"]


def test_cross_story_moral_patterns():
    from src.rag.cross_story_rag import CrossStoryRAG

    rag = CrossStoryRAG()
    rag.add_story("Kind Story", "Nguoi em tot bung giup do moi nguoi va duoc thuong.")
    rag.add_story("Greedy Story", "Nguoi anh tham lam va ich ky nen bi phat.")

    results = rag.find_moral_patterns("tham lam bi phat", top_k=1)

    assert results[0]["title"] == "Greedy Story"
    assert "greed_warning" in results[0]["matched_morals"]


def test_cross_story_accepts_documents():
    from src.rag.cross_story_rag import CrossStoryRAG

    rag = CrossStoryRAG()
    docs = [
        Document(
            page_content="Son Tinh va Thuy Tinh tranh tai gay mua gio song nui.",
            metadata={"title": "Son Tinh Thuy Tinh"},
        )
    ]

    records = rag.add_documents(docs)

    assert rag.count == 1
    assert records[0].title == "Son Tinh Thuy Tinh"
    assert rag.as_documents()[0].metadata["story_id"] == records[0].story_id


def test_fairy_tale_dataset_builder_builds_records():
    from src.data import FairyTaleDatasetBuilder

    builder = FairyTaleDatasetBuilder()
    records = builder.build_from_texts(
        [
            {
                "title": "Thach Sanh",
                "text": "Thach Sanh ngheo nhung dung cam chien dau cuu cong chua.",
                "metadata": {"source": "memory"},
            }
        ]
    )

    assert len(records) == 1
    assert records[0].title == "Thach Sanh"
    assert records[0].word_count > 0
    assert "hero_quest" in records[0].motifs
    assert records[0].metadata["source"] == "memory"


def test_fairy_tale_dataset_builder_directory_and_export(tmp_path):
    from src.data import FairyTaleDatasetBuilder

    source_dir = tmp_path / "stories"
    source_dir.mkdir()
    (source_dir / "tam_cam.txt").write_text(
        "Tam bi Cam ham hai. Cai thien thang cai ac va ke tham lam bi phat.",
        encoding="utf-8",
    )

    builder = FairyTaleDatasetBuilder()
    records = builder.build_from_directory(source_dir)
    jsonl_path = builder.export_jsonl(records, tmp_path / "dataset" / "stories.jsonl")
    docs = builder.to_documents(records)

    assert len(records) == 1
    assert jsonl_path.exists()
    assert docs[0].metadata["title"] == records[0].title

    line = jsonl_path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["title"] == "tam cam"
    assert "greed_warning" in payload["morals"]

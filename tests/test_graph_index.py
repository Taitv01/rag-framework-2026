"""Knowledge graph index (GraphRAG): extraction, communities, storage, context (no models)."""

import importlib.util
import json
import re

import pytest

from src.rag.graph_index import extraction_units, parse_extraction
from tests.fakes import ScriptedChat, fake_embeddings_manager

HAS_QDRANT = importlib.util.find_spec("qdrant_client") and importlib.util.find_spec("langchain_qdrant")
PROVIDERS = ["faiss", pytest.param("qdrant", marks=pytest.mark.skipif(not HAS_QDRANT, reason="qdrant extra"))]

STORIES = {
    "thach_sanh.md": (
        "# Thạch Sanh\n\nThạch Sanh giết chằn tinh. Chàng cứu thái tử con vua Thủy Tề, "
        "được vua Thủy Tề đưa xuống thủy cung và tặng cây đàn thần."
    ),
    "con_rong.md": (
        "# Con Rồng cháu Tiên\n\nLạc Long Quân là con trai thần Long Nữ, thường sống ở thủy cung. "
        "Lạc Long Quân lấy nàng Âu Cơ, sinh ra bọc trăm trứng."
    ),
    "tam_cam.md": (
        "# Tấm Cám\n\nTấm bị mẹ con Cám hãm hại, hóa thành chim vàng anh. "
        "Cuối cùng mẹ con Cám bị trừng phạt."
    ),
}

EXTRACTIONS = {
    "thach_sanh.md": {
        "entities": [
            {"name": "Thạch Sanh", "type": "person", "description": "Dũng sĩ giết chằn tinh."},
            {"name": "Thủy Tề", "type": "creature", "description": "Vua dưới nước, cha của thái tử."},
            {"name": "đàn thần", "type": "object", "description": "Cây đàn thần kỳ vua Thủy Tề tặng."},
            {"name": "thủy cung", "type": "place", "description": "Cung điện dưới nước của vua Thủy Tề."},
        ],
        "relationships": [
            {"source": "Thạch Sanh", "target": "Thủy Tề", "description": "Thạch Sanh cứu con trai Thủy Tề."},
            {"source": "Thủy Tề", "target": "đàn thần", "description": "Thủy Tề tặng Thạch Sanh đàn thần."},
            {"source": "Thủy Tề", "target": "thủy cung", "description": "Thủy Tề trị vì thủy cung."},
            {"source": "Thạch Sanh", "target": "đàn thần", "description": "Thạch Sanh nhận đàn thần."},
        ],
    },
    "con_rong.md": {
        "entities": [
            {"name": "Lạc Long Quân", "type": "creature", "description": "Con trai thần Long Nữ."},
            {"name": "Âu Cơ", "type": "person", "description": "Vợ Lạc Long Quân."},
            {"name": "Thủy cung", "type": "place", "description": "Nơi Lạc Long Quân sống."},
        ],
        "relationships": [
            {"source": "Lạc Long Quân", "target": "Thủy cung", "description": "Lạc Long Quân sống ở thủy cung."},
            {"source": "Lạc Long Quân", "target": "Âu Cơ", "description": "Lạc Long Quân lấy Âu Cơ."},
            {"source": "Âu Cơ", "target": "Thủy cung", "description": "Âu Cơ nhớ chồng ở thủy cung."},
        ],
    },
    "tam_cam.md": {
        "entities": [
            {"name": "Tấm", "type": "person", "description": "Cô gái hiền bị hãm hại."},
            {"name": "Cám", "type": "person", "description": "Em cùng cha khác mẹ của Tấm."},
            {"name": "chim vàng anh", "type": "creature", "description": "Tấm hóa thành chim vàng anh."},
        ],
        "relationships": [
            {"source": "Cám", "target": "Tấm", "description": "Cám hãm hại Tấm."},
            {"source": "Tấm", "target": "chim vàng anh", "description": "Tấm hóa thành chim vàng anh."},
            {"source": "Cám", "target": "chim vàng anh", "description": "Cám giết chim vàng anh."},
        ],
    },
}


def is_extraction(prompt):
    return "knowledge graph for a search index" in prompt


def is_report(prompt):
    return "community report for a search index" in prompt


def report_names(prompt):
    """Entity names listed in a report prompt."""
    section = prompt.split("Entities / Thực thể:")[1].split("Relationships / Quan hệ:")[0]
    return re.findall(r"^- (.+?) \(", section, flags=re.MULTILINE)


def graph_reply(prompt):
    if is_extraction(prompt):
        for name, extraction in EXTRACTIONS.items():
            if STORIES[name].split("\n\n")[1][:40] in prompt:
                return "```json\n" + json.dumps(extraction, ensure_ascii=False) + "\n```"
    if is_report(prompt):
        return "Tiêu đề: nhóm " + ", ".join(report_names(prompt))
    if "document card for a search index" in prompt:
        return "Thẻ."
    return "Trả lời [S1]."


class WordOverlapReranker:
    """Cross-encoder stand-in: shared words between query and text."""

    def predict(self, pairs, **_kwargs):
        return [float(len(set(q.lower().split()) & set(t.lower().split()))) for q, t in pairs]


def write_corpus(directory, stories=STORIES):
    directory.mkdir(exist_ok=True)
    for name, text in stories.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def make_rag(monkeypatch, tmp_path, provider="faiss", reply=graph_reply, cls=None, **kwargs):
    monkeypatch.setattr("src.rag.advanced_rag.EmbeddingsManager", lambda **_: fake_embeddings_manager())
    from src.rag.advanced_rag import AdvancedRAG

    options = dict(
        vector_store_provider=provider,
        persist_directory=str(tmp_path / f"store_{provider}"),
        chunk_size=160,
        chunk_overlap=20,
        retrieval_k=1,
        use_reranking=True,
        reranker=WordOverlapReranker(),
        reranker_model="fake-reranker",
        use_document_cards=False,
        use_graph=True,
    )
    options.update(kwargs)
    rag = (cls or AdvancedRAG)(**options)
    chat = ScriptedChat(reply)
    rag.llm._llm = chat
    return rag, chat


def calls(chat, kind):
    return [prompt for prompt in chat.prompts if kind(prompt)]


def test_extraction_units_cut_between_paragraphs():
    text = "a" * 30 + "\n\n" + "b" * 30 + "\n\n" + "c" * 70
    assert extraction_units(text, max_chars=64) == ["a" * 30 + "\n\n" + "b" * 30, "c" * 64, "c" * 6]
    assert extraction_units("Một truyện ngắn.") == ["Một truyện ngắn."]


def test_parse_extraction_reads_fenced_json_and_rejects_the_rest():
    parsed = parse_extraction('```json\n{"entities": [{"name": "Tấm"}, {"name": ""}], '
                              '"relationships": [{"source": "Tấm", "target": ""}]}\n```')
    assert parsed == {"entities": [{"name": "Tấm"}], "relationships": []}
    assert parse_extraction("OK") is None
    assert parse_extraction('{"entities": "Tấm"}') is None


def test_entities_merge_across_documents_into_communities(monkeypatch, tmp_path):
    rag, chat = make_rag(monkeypatch, tmp_path)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    assert len(calls(chat, is_extraction)) == 3  # one per story, not per chunk
    graph = rag._graph
    # "thủy cung" and "Thủy cung" are one entity, known from both stories.
    assert rag.num_entities == 9
    water = graph.view.entity_profile("thủy cung")
    assert water.metadata["documents"] == ["con_rong.md", "thach_sanh.md"]
    assert graph.communities and all(len(c.entities) >= 3 for c in graph.communities)
    assert len(calls(chat, is_report)) == len(graph.communities) == rag.num_communities
    # The water world links the two stories: one community spans both.
    assert any(len(c.documents) == 2 for c in graph.communities)


def test_unchanged_text_reuses_graph_and_reports(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path / "docs")
    manifest = tmp_path / "manifest.json"
    rag, chat = make_rag(monkeypatch, tmp_path)
    rag.refresh_markdown_directory(corpus, manifest_path=manifest)
    extractions, reports = len(calls(chat, is_extraction)), len(calls(chat, is_report))

    rag.add_documents(corpus)  # same texts: no LLM call
    assert (len(calls(chat, is_extraction)), len(calls(chat, is_report))) == (extractions, reports)

    (corpus / "tam_cam.md").write_text(STORIES["tam_cam.md"] + " Tấm sống hạnh phúc.", encoding="utf-8")
    rag.refresh_markdown_directory(corpus, manifest_path=manifest)
    # Only the changed story is read again; communities whose text is unchanged keep their report.
    assert len(calls(chat, is_extraction)) == extractions + 1
    assert len(calls(chat, is_report)) == reports


def test_removed_documents_leave_the_graph(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path / "docs")
    manifest = tmp_path / "manifest.json"
    rag, _ = make_rag(monkeypatch, tmp_path)
    rag.refresh_markdown_directory(corpus, manifest_path=manifest)

    (corpus / "tam_cam.md").unlink()
    rag.refresh_markdown_directory(corpus, manifest_path=manifest)

    assert "tấm" not in rag._graph.view.entities
    assert all("tấm" not in c.entities for c in rag._graph.communities)
    stored = rag._graph.entity_store.get_all_documents()
    assert {p.metadata["relative_source"] for p in stored} == {"thach_sanh.md", "con_rong.md"}
    assert set(rag._graph.reports) == {c.id for c in rag._graph.communities}
    assert {r.metadata["chunk_id"] for r in rag._graph.report_store.get_all_documents()} == set(rag._graph.reports)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_graph_survives_a_restart_without_llm_calls(monkeypatch, tmp_path, provider):
    rag, _ = make_rag(monkeypatch, tmp_path, provider)
    rag.add_documents(write_corpus(tmp_path / "docs"))
    entities, edges = rag.num_entities, rag._graph.num_relationships
    communities = sorted(c.id for c in rag._graph.communities)
    rag.vector_store.close()

    restarted, chat = make_rag(monkeypatch, tmp_path, provider)

    assert (restarted.num_entities, restarted._graph.num_relationships) == (entities, edges)
    assert sorted(c.id for c in restarted._graph.communities) == communities
    assert restarted.num_communities == len(communities)
    assert chat.prompts == []
    restarted.vector_store.close()


def test_entities_join_as_one_graph_source_when_they_outrank_passages(monkeypatch, tmp_path):
    rag, chat = make_rag(monkeypatch, tmp_path)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    docs = rag.retrieve("đàn thần của Thủy Tề", k=1)
    graph = [doc for doc in docs if doc.metadata.get("graph_context")]
    assert len(graph) == 1 and docs[-1] is graph[0]
    assert "đàn thần" in graph[0].metadata["entities"]
    assert "Thủy Tề tặng Thạch Sanh đàn thần." in graph[0].page_content

    result = rag.query_detailed("đàn thần của Thủy Tề", k=1)
    prompt = chat.prompts[-1]
    assert "knowledge graph (entities and relationships extracted from thach_sanh.md" in prompt
    assert any(source["metadata"].get("graph_context") for source in result["relevant_docs"])


def test_no_graph_context_without_reranker_scores(monkeypatch, tmp_path):
    rag, _ = make_rag(monkeypatch, tmp_path)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    docs = rag.retrieve("đàn thần của Thủy Tề", k=1, use_reranking=False)
    assert not any(doc.metadata.get("graph_context") or doc.metadata.get("community_report") for doc in docs)


def test_community_report_joins_when_it_outranks_passages(monkeypatch, tmp_path):
    rag, _ = make_rag(monkeypatch, tmp_path, max_context_entities=0)
    rag.add_documents(write_corpus(tmp_path / "docs"))
    names = [rag._graph.view.entities[key]["name"] for key in rag._graph.communities[0].entities]

    docs = rag.retrieve("nhóm " + " ".join(names), k=1)

    reports = [doc for doc in docs if doc.metadata.get("community_report")]
    assert len(reports) == 1 and reports[0].metadata["chunk_id"] == rag._graph.communities[0].id


def test_filters_limit_graph_context_to_matching_documents(monkeypatch, tmp_path):
    rag, _ = make_rag(monkeypatch, tmp_path)
    rag.add_documents(write_corpus(tmp_path / "docs"))
    spanning = next(c for c in rag._graph.communities if len(c.documents) == 2)
    query = "nhóm " + " ".join(rag._graph.view.entities[key]["name"] for key in spanning.entities)

    reports = rag._graph.report_candidates(query, {"file_name": "con_rong.md"}, k=4)
    assert spanning.id not in {r.metadata["chunk_id"] for r in reports}
    assert spanning.id in {r.metadata["chunk_id"] for r in rag._graph.report_candidates(query, None, k=4)}

    profiles = rag._graph.entity_candidates("thủy cung", {"file_name": "con_rong.md"}, k=4)
    water = next(p for p in profiles if p.metadata["graph_entity"] == "thủy cung")
    assert water.metadata["documents"] == ["con_rong.md"]
    assert "Thủy Tề" not in water.metadata["graph_profile"]


def test_profiles_name_the_document_of_each_line_and_rerank_short(monkeypatch, tmp_path):
    rag, _ = make_rag(monkeypatch, tmp_path)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    water = rag._graph.view.entity_profile("thủy cung")
    # Known from two stories: each description and relationship says which one.
    assert "  con_rong.md: Nơi Lạc Long Quân sống." in water.page_content
    assert "  thach_sanh.md: Cung điện dưới nước của vua Thủy Tề." in water.page_content
    assert "thach_sanh.md: Thủy Tề trị vì thủy cung." in water.metadata["graph_profile"]
    # The reranker reads the short part; relationships are for the LLM.
    assert "Thủy Tề trị vì" not in water.page_content

    lone = rag._graph.view.entity_profile("tấm")
    assert "  Cô gái hiền bị hãm hại." in lone.page_content  # one story: no label


def test_queries_without_diacritics_find_named_entities(monkeypatch, tmp_path):
    rag, _ = make_rag(monkeypatch, tmp_path)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    candidates = rag._graph.entity_candidates("lac long quan lay ai", None, k=2)
    assert len(candidates) == 2
    assert candidates[0].metadata["graph_entity"] == "lạc long quân"  # named entities first


def test_a_dead_provider_stops_graph_calls_and_indexing_goes_on(monkeypatch, tmp_path):
    from src.rag.document_cards import MAX_CONSECUTIVE_CARD_FAILURES

    stories = {f"truyen_{i}.md": f"# Truyện {i}\n\nChuyện thứ {i}." for i in range(6)}

    def reply(prompt):
        raise RuntimeError("no API key")

    rag, chat = make_rag(monkeypatch, tmp_path, reply=reply)
    chunks = rag.add_documents(write_corpus(tmp_path / "docs", stories))

    assert chunks > 0 and rag.num_entities == 0 and rag.num_communities == 0
    assert len(chat.prompts) == MAX_CONSECUTIVE_CARD_FAILURES


def test_invalid_json_is_extracted_again_next_time(monkeypatch, tmp_path):
    answers = {"mode": "OK"}

    def reply(prompt):
        return answers["mode"] if is_extraction(prompt) else graph_reply(prompt)

    corpus = write_corpus(tmp_path / "docs")
    rag, chat = make_rag(monkeypatch, tmp_path, reply=reply)
    rag.add_documents(corpus)
    # Placeholder answers (AgentLLM's first pass) are not failures: every story is asked.
    assert len(calls(chat, is_extraction)) == 3 and rag.num_entities == 0

    answers["mode"] = None
    rag.llm._llm = chat = ScriptedChat(graph_reply)
    rag.add_documents(corpus)
    assert len(calls(chat, is_extraction)) == 3 and rag.num_entities == 9


def test_graphrag_keeps_a_knowledge_graph_view(monkeypatch, tmp_path):
    from src.rag.graph_rag import GraphRAG

    rag, chat = make_rag(monkeypatch, tmp_path, cls=GraphRAG, use_graph=True)
    # Texts without a source are still documents with a graph.
    rag.add_texts([STORIES["thach_sanh.md"], STORIES["con_rong.md"]])

    assert len(calls(chat, is_extraction)) == 2
    knowledge_graph = rag.get_knowledge_graph()
    assert "Thạch Sanh" in knowledge_graph.entities
    assert "Thủy Tề" in knowledge_graph.get_neighbors("Thạch Sanh")
    context = rag.extract_subgraph_context("Thạch Sanh được tặng gì?", max_hops=1)
    assert "Thạch Sanh -[related_to]-> đàn thần" in context["subgraph_triples"]
    communities = rag.get_communities()
    assert communities and communities[0].summary.startswith("Tiêu đề: nhóm")
    assert not rag.has_neo4j

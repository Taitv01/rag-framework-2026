"""Document cards: one LLM summary per document, kept beside the chunks (no models)."""

import importlib.util

import pytest
from langchain_core.documents import Document

from src.rag.document_cards import card_document, card_prompt, content_hash, group_by_document
from tests.fakes import ScriptedChat, fake_embeddings_manager

HAS_QDRANT = importlib.util.find_spec("qdrant_client") and importlib.util.find_spec("langchain_qdrant")
PROVIDERS = ["faiss", pytest.param("qdrant", marks=pytest.mark.skipif(not HAS_QDRANT, reason="qdrant extra"))]

STORIES = {
    "cay_khe.md": (
        "# Cây khế\n\nNgười anh chia gia tài, chỉ để lại cho người em một cây khế. Chim lạ ăn khế "
        "rồi chở người em ra đảo lấy vàng. Người anh may túi mười hai gang, rơi xuống biển."
    ),
    "thach_sanh.md": (
        "# Thạch Sanh\n\nThạch Sanh sống dưới gốc đa, gia tài chỉ có một lưỡi búa của cha. "
        "Lý Thông lừa chàng đi canh miếu. Mẹ con Lý Thông bị sét đánh chết, hóa thành bọ hung."
    ),
}

CARDS = {
    "cay_khe.md": "Tiêu đề: Cây khế\nMô-típ, chủ đề: kẻ tham lam bị trừng phạt; chim trả ơn",
    "thach_sanh.md": "Tiêu đề: Thạch Sanh\nMô-típ, chủ đề: kẻ ác bị trừng phạt; dũng sĩ",
}


def card_reply(prompt):
    if "document card for a search index" in prompt:
        for name, card in CARDS.items():
            if f"Tài liệu: {name}" in prompt:
                return card
    return "Trả lời [S1]."


class WordOverlapReranker:
    """Cross-encoder stand-in: shared words between query and text."""

    def predict(self, pairs, **_kwargs):
        return [float(len(set(query.lower().split()) & set(text.lower().split()))) for query, text in pairs]


def write_corpus(directory, stories=STORIES):
    directory.mkdir(exist_ok=True)
    for name, text in stories.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def make_rag(monkeypatch, tmp_path, provider="faiss", reply=card_reply, **kwargs):
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
        use_document_cards=True,
    )
    options.update(kwargs)
    rag = AdvancedRAG(**options)
    chat = ScriptedChat(reply)
    rag.llm._llm = chat
    return rag, chat


def card_calls(chat):
    return [p for p in chat.prompts if "document card for a search index" in p]


def test_card_prompt_names_the_file_not_its_path_and_cuts_long_documents():
    pages = [
        Document(page_content="Trang một.", metadata={"source": "/tmp/x/a.pdf", "file_name": "a.pdf", "page": 0}),
        Document(page_content="Trang hai " * 50, metadata={"source": "/tmp/x/a.pdf", "file_name": "a.pdf", "page": 1}),
    ]
    prompt = card_prompt(pages, max_chars=100)

    assert "Tài liệu: a.pdf" in prompt and "/tmp/x" not in prompt
    assert "Trang một." in prompt and prompt.endswith("[...]")
    assert list(group_by_document(pages)) == ["/tmp/x/a.pdf"]


def test_card_keeps_document_metadata_but_not_page_metadata():
    pages = [Document(page_content="Truyện.", metadata={"source": "s.md", "tenant": "a", "page": 3,
                                                        "start_index": 7})]
    card = card_document(pages, " Tiêu đề: S \n", model="m")

    assert card.page_content == "Tiêu đề: S"
    assert card.metadata["tenant"] == "a" and card.metadata["document_card"] is True
    assert "page" not in card.metadata and "start_index" not in card.metadata
    assert card.metadata["content_sha256"] == content_hash(pages)
    assert card.metadata["card_model"] == "m"


def test_cards_are_on_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr("src.rag.advanced_rag.EmbeddingsManager", lambda **_: fake_embeddings_manager())
    from src.rag.advanced_rag import AdvancedRAG

    rag = AdvancedRAG(vector_store_provider="faiss", persist_directory=str(tmp_path / "store"))
    rag.llm._llm = ScriptedChat(card_reply)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    assert rag.use_document_cards and rag.num_cards == 2


def test_cards_can_be_turned_off(monkeypatch, tmp_path):
    rag, chat = make_rag(monkeypatch, tmp_path, use_document_cards=False)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    assert rag.num_cards == 0 and card_calls(chat) == []
    assert all(not doc.metadata.get("document_card") for doc in rag.retrieve("kẻ ác bị trừng phạt", k=1))


def test_one_card_per_document_and_unchanged_text_reuses_it(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path / "docs")
    rag, chat = make_rag(monkeypatch, tmp_path)

    rag.add_documents(corpus)
    assert rag.num_cards == 2 and len(card_calls(chat)) == 2

    rag.add_documents(corpus)  # same texts: no new LLM call
    assert rag.num_cards == 2 and len(card_calls(chat)) == 2


def test_refresh_rewrites_changed_cards_and_drops_removed_ones(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path / "docs")
    manifest = tmp_path / "manifest.json"
    rag, chat = make_rag(monkeypatch, tmp_path)
    rag.refresh_markdown_directory(corpus, manifest_path=manifest)
    assert len(card_calls(chat)) == 2

    (corpus / "cay_khe.md").write_text(STORIES["cay_khe.md"] + " Người em sống giản dị.", encoding="utf-8")
    rag.refresh_markdown_directory(corpus, manifest_path=manifest)
    assert len(card_calls(chat)) == 3 and rag.num_cards == 2

    (corpus / "thach_sanh.md").unlink()
    rag.refresh_markdown_directory(corpus, manifest_path=manifest)
    assert rag.num_cards == 1
    stored = rag._card_store.get_all_documents()
    assert [card.metadata["relative_source"] for card in stored] == ["cay_khe.md"]


def test_reused_card_takes_the_new_metadata(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path / "docs")
    rag, chat = make_rag(monkeypatch, tmp_path)
    rag.add_documents(corpus, metadata={"tenant": "a"})
    rag.add_documents(corpus, metadata={"tenant": "b"})

    assert len(card_calls(chat)) == 2
    docs = rag.retrieve("kẻ tham lam bị trừng phạt", k=1, filter={"tenant": "b"})
    assert any(doc.metadata.get("document_card") for doc in docs)


def test_reuploading_an_unchanged_file_keeps_its_card(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path / "docs")
    rag, chat = make_rag(monkeypatch, tmp_path)

    rag.add_files({"cay_khe.md": corpus / "cay_khe.md"})
    rag.add_files({"cay_khe.md": corpus / "cay_khe.md"})

    assert len(card_calls(chat)) == 1 and rag.num_cards == 1


@pytest.mark.parametrize("provider", PROVIDERS)
def test_cards_survive_a_restart(monkeypatch, tmp_path, provider):
    rag, _ = make_rag(monkeypatch, tmp_path, provider)
    rag.add_documents(write_corpus(tmp_path / "docs"))
    rag.vector_store.close()

    restarted, chat = make_rag(monkeypatch, tmp_path, provider)

    assert restarted.num_cards == 2
    docs = restarted.retrieve("truyện nào có kẻ ác bị trừng phạt", k=1)
    assert any(doc.metadata.get("document_card") for doc in docs)
    assert card_calls(chat) == []
    restarted.vector_store.close()


def test_failed_card_does_not_stop_indexing(monkeypatch, tmp_path):
    def reply(prompt):
        if "Tài liệu: cay_khe.md" in prompt:
            raise RuntimeError("provider down")
        return card_reply(prompt)

    rag, _ = make_rag(monkeypatch, tmp_path, reply=reply)
    chunks = rag.add_documents(write_corpus(tmp_path / "docs"))

    assert chunks > 0 and rag.num_documents == 2
    assert [card.metadata["file_name"] for card in rag._cards.values()] == ["thach_sanh.md"]


def test_a_dead_provider_stops_card_calls_for_the_rest_of_the_batch(monkeypatch, tmp_path):
    from src.rag.document_cards import MAX_CONSECUTIVE_CARD_FAILURES

    stories = {f"truyen_{i}.md": f"# Truyện {i}\n\nChuyện thứ {i}." for i in range(6)}
    calls = []

    def reply(prompt):
        calls.append(prompt)
        raise RuntimeError("no API key")

    rag, _ = make_rag(monkeypatch, tmp_path, reply=reply)
    chunks = rag.add_documents(write_corpus(tmp_path / "docs", stories))

    assert chunks > 0 and rag.num_documents == 6 and rag.num_cards == 0
    assert len(calls) == MAX_CONSECUTIVE_CARD_FAILURES


def test_cards_join_only_when_they_outrank_passages(monkeypatch, tmp_path):
    rag, _ = make_rag(monkeypatch, tmp_path)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    detail = rag.retrieve("gia tài của Thạch Sanh chỉ có một lưỡi búa của cha", k=1)
    assert len(detail) == 1 and not detail[0].metadata.get("document_card")

    motif = rag.retrieve("kẻ tham lam bị trừng phạt", k=1)
    assert len(motif) == 2  # the k passages, then the card
    assert not motif[0].metadata.get("document_card")
    assert motif[1].metadata["document_card"] and motif[1].metadata["file_name"] == "cay_khe.md"


def test_cards_need_reranker_scores_and_respect_the_cap(monkeypatch, tmp_path):
    rag, _ = make_rag(monkeypatch, tmp_path, max_context_cards=1)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    capped = rag.retrieve("kẻ ác tham lam bị trừng phạt", k=2)
    assert sum(bool(doc.metadata.get("document_card")) for doc in capped) == 1

    unranked = rag.retrieve("kẻ ác tham lam bị trừng phạt", k=2, use_reranking=False)
    assert not any(doc.metadata.get("document_card") for doc in unranked)


def test_filters_apply_to_cards(monkeypatch, tmp_path):
    rag, _ = make_rag(monkeypatch, tmp_path)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    docs = rag.retrieve("kẻ ác bị trừng phạt", k=1, filter={"file_name": "thach_sanh.md"})

    assert {doc.metadata["file_name"] for doc in docs} == {"thach_sanh.md"}
    assert any(doc.metadata.get("document_card") for doc in docs)


def test_the_llm_is_told_which_sources_are_cards(monkeypatch, tmp_path):
    rag, chat = make_rag(monkeypatch, tmp_path)
    rag.add_documents(write_corpus(tmp_path / "docs"))

    result = rag.query_detailed("kẻ tham lam bị trừng phạt", k=1)

    prompt = chat.prompts[-1]
    assert "[S2] Source:" in prompt and "document card" in prompt
    assert result["relevant_docs"][1]["metadata"]["document_card"] is True

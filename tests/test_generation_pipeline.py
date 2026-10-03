"""One generation pipeline for query, query_detailed and stream (no models, scripted LLM)."""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from tests.fakes import fake_embeddings_manager

STORIES = {
    "thach_sanh.md": "# Thạch Sanh\n\nThạch Sanh sống dưới gốc đa, gia tài chỉ có một lưỡi búa của cha. "
                     "Chàng dùng búa chặt đầu chằn tinh, xác nó là con trăn khổng lồ.",
    "tam_cam.md": "# Tấm Cám\n\nBụt bảo Tấm thả cá bống xuống giếng và gọi bống lên ăn cơm. "
                  "Mẹ con Cám bắt bống làm thịt.",
}


class ScriptedChat:
    """Chat model whose reply depends on the prompt; records every call."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages, **_kwargs):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply(prompt))

    def stream(self, messages, **kwargs):
        for word in self.invoke(messages).content.split(" "):
            yield SimpleNamespace(content=word + " ")


def default_reply(prompt):
    if "search query optimizer" in prompt:
        return "Thạch Sanh giết chằn tinh bằng gì"
    if "document relevance grader" in prompt:
        return "S1"
    return "Thạch Sanh dùng búa [S1]."


@pytest.fixture
def make_rag(monkeypatch, tmp_path):
    def build(reply=default_reply, **kwargs):
        monkeypatch.setattr("src.rag.advanced_rag.EmbeddingsManager", lambda **_: fake_embeddings_manager())
        from src.rag.advanced_rag import AdvancedRAG

        corpus = tmp_path / "docs"
        corpus.mkdir(exist_ok=True)
        for name, text in STORIES.items():
            (corpus / name).write_text(text, encoding="utf-8")
        rag = AdvancedRAG(vector_store_provider="faiss", chunk_size=160, chunk_overlap=20,
                          retrieval_k=2, use_reranking=False, **kwargs)
        rag.add_documents(corpus)
        chat = ScriptedChat(reply)
        rag.llm._llm = chat
        return rag, chat

    return build


def test_default_pipeline_calls_the_llm_once(make_rag):
    rag, chat = make_rag()
    result = rag.query_detailed("Thạch Sanh giết chằn tinh bằng gì?")

    assert len(chat.prompts) == 1  # no rewrite (has diacritics), no grading
    assert result["answer"] == "Thạch Sanh dùng búa [S1]."
    assert [c["source_id"] for c in result["citations"]] == ["S1"]
    assert len(result["relevant_docs"]) == 2 and not result["abstained"]


def test_queries_without_diacritics_are_rewritten_first(make_rag):
    rag, chat = make_rag()
    result = rag.query_detailed("thach sanh giet chan tinh bang gi")

    assert len(chat.prompts) == 2
    assert result["transformed_query"] == "Thạch Sanh giết chằn tinh bằng gì"


def test_llm_grading_is_one_batched_call(make_rag):
    rag, chat = make_rag(grading="llm")
    result = rag.query_detailed("Thạch Sanh giết chằn tinh bằng gì?")

    grading_prompts = [p for p in chat.prompts if "document relevance grader" in p]
    assert len(grading_prompts) == 1 and "[S2]" in grading_prompts[0]
    assert result["relevant_docs_count"] == 1 and len(chat.prompts) == 2


def test_nothing_relevant_means_no_answer_from_the_llm(make_rag):
    from src.rag.advanced_rag import NO_CONTEXT_ANSWER

    rag, chat = make_rag(reply=lambda p: "NONE" if "grader" in p else "invented", grading="llm")
    result = rag.query_detailed("Truyện Cóc kiện trời kể gì?")

    assert result["answer"] == NO_CONTEXT_ANSWER and result["abstained"]
    assert result["relevant_docs"] == [] and len(chat.prompts) == 1  # the grading call only


def test_reranker_grading_drops_low_scores(make_rag):
    rag, _ = make_rag(grading="reranker", min_relevance_score=0.5)
    docs = [Document(page_content="a", metadata={"relevance_score": 0.9}),
            Document(page_content="b", metadata={"relevance_score": 0.1}),
            Document(page_content="c", metadata={})]
    assert [d.page_content for d in rag._grade("q", docs)] == ["a", "c"]


def test_citations_keep_only_cited_sources_and_flag_unknown_ones(make_rag):
    rag, _ = make_rag(reply=lambda p: "Bống ở giếng [S2], xem thêm [S9].")
    result = rag.query_detailed("Tấm thả cá bống ở đâu?")

    assert [c["source_id"] for c in result["citations"]] == ["S2"]
    assert result["invalid_citations"] == ["S9"]


def test_stream_and_query_share_the_pipeline(make_rag):
    rag, chat = make_rag()
    streamed = "".join(rag.stream("Thạch Sanh giết chằn tinh bằng gì?")).strip()
    answered = rag.query("Thạch Sanh giết chằn tinh bằng gì?")

    assert streamed == answered == "Thạch Sanh dùng búa [S1]."
    assert len(chat.prompts) == 2 and chat.prompts[0] == chat.prompts[1]


def test_cache_is_cleared_when_the_corpus_changes(make_rag, tmp_path):
    rag, chat = make_rag(use_cache=True)
    rag.query("Thạch Sanh giết chằn tinh bằng gì?")
    rag.query("Thạch Sanh giết chằn tinh bằng gì?")
    assert len(chat.prompts) == 1  # second answer from the cache

    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "so_dua.md").write_text("# Sọ Dừa\n\nSọ Dừa chăn bò rất giỏi.", encoding="utf-8")
    rag.add_documents(extra)
    rag.query("Thạch Sanh giết chằn tinh bằng gì?")
    assert len(chat.prompts) == 2

"""AgenticRAG bounds its loop and keeps sources; AdaptiveRAG routes share one index (no models)."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from tests.fakes import fake_embeddings_manager

STORIES = {
    "thach_sanh.md": "# Thạch Sanh\n\nThạch Sanh sống dưới gốc đa, gia tài chỉ có một lưỡi búa của cha. "
                     "Chàng dùng búa chặt đầu chằn tinh, xác nó là con trăn khổng lồ.",
    "tam_cam.md": "# Tấm Cám\n\nBụt bảo Tấm thả cá bống xuống giếng và gọi bống lên ăn cơm. "
                  "Mẹ con Cám bắt bống làm thịt.",
}
QUESTION = "Thạch Sanh giết chằn tinh bằng gì?"


class AgentChat:
    """Chat model scripted by prompt; bind_tools() returns a twin that always searches."""

    def __init__(self, relevant="no", answer="Thạch Sanh dùng búa [S1]."):
        self.relevant = relevant
        self.answer = answer
        self.prompts = []
        self.tool_queries = []

    def invoke(self, messages, **_kwargs):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        if "document relevance grader" in prompt:
            return AIMessage(content=self.relevant)
        if "search query optimizer" in prompt:
            return AIMessage(content=f"truy vấn {len(self.rewrite_prompts)}")
        return AIMessage(content=self.answer)

    def bind_tools(self, tools):
        chat = self

        class Bound:
            def invoke(self, messages, **_kwargs):
                query = messages[-1].content
                chat.tool_queries.append(query)
                call = {"name": tools[0].name, "args": {"query": query}, "id": f"call_{len(chat.tool_queries)}"}
                return AIMessage(content="", tool_calls=[call])

        return Bound()

    @property
    def rewrite_prompts(self):
        return [p for p in self.prompts if "search query optimizer" in p]

    @property
    def answer_prompts(self):
        return [p for p in self.prompts if "Bạn là trợ lý AI hữu ích" in p]


def make_agent(chat, **kwargs):
    from src.core.llm import LLMManager
    from src.rag.agentic_rag import AgenticRAG

    searched = []

    def search(query, k):
        searched.append((query, k))
        return [Document(page_content="Chàng dùng búa chặt đầu chằn tinh.",
                         metadata={"source": "thach_sanh.md", "parent_text": "x" * 50})]

    llm = LLMManager()
    llm._llm = chat
    return AgenticRAG(search=search, llm=llm, retrieval_k=3, **kwargs), searched


def test_agent_stops_rewriting_after_max_retries_and_answers_the_users_question():
    chat = AgentChat(relevant="no")
    agent, searched = make_agent(chat, max_retries=2)

    result = agent.query_with_trace(QUESTION)

    assert result["rewrites"] == 2
    assert chat.tool_queries == [QUESTION, "truy vấn 1", "truy vấn 2"]
    assert [k for _, k in searched] == [3, 3, 3]
    # The last rewrite is told which queries already failed.
    assert f"- {QUESTION}" in chat.rewrite_prompts[-1] and "- truy vấn 1" in chat.rewrite_prompts[-1]
    # One answer, for the original question rather than a rewritten query.
    assert len(chat.answer_prompts) == 1
    assert f"Question / Câu hỏi: {QUESTION}" in chat.answer_prompts[0]
    assert result["answer"] == "Thạch Sanh dùng búa [S1]."


def test_agent_answers_with_cited_sources_when_documents_are_relevant():
    chat = AgentChat(relevant="yes")
    agent, _ = make_agent(chat)

    result = agent.query_with_trace(QUESTION)

    assert result["rewrites"] == 0 and chat.rewrite_prompts == []
    assert "[S1] Source: thach_sanh.md" in chat.answer_prompts[0]
    assert result["sources"][0]["source_id"] == "S1"
    assert result["sources"][0]["source"] == "thach_sanh.md"
    assert "parent_text" not in result["sources"][0]["metadata"]
    assert agent.query(QUESTION) == "Thạch Sanh dùng búa [S1]."


def test_agent_gives_up_at_the_recursion_limit():
    from src.rag.agentic_rag import GAVE_UP_ANSWER

    chat = AgentChat(relevant="no")
    agent, _ = make_agent(chat, max_retries=10, recursion_limit=5)

    assert agent.query(QUESTION) == GAVE_UP_ANSWER
    assert chat.answer_prompts == []


def test_agent_on_a_shared_index_loads_no_store_and_refuses_documents():
    agent, _ = make_agent(AgentChat())

    assert agent.vector_store is None and agent.embeddings is None
    with pytest.raises(RuntimeError, match="shared index"):
        agent.add_documents("docs/")


@pytest.fixture
def adaptive(monkeypatch, tmp_path):
    built = []

    def embeddings_factory(**_kwargs):
        built.append(1)
        return fake_embeddings_manager()

    monkeypatch.setattr("src.rag.advanced_rag.EmbeddingsManager", embeddings_factory)
    from src.rag.adaptive_rag import AdaptiveRAG

    corpus = tmp_path / "docs"
    corpus.mkdir()
    for name, text in STORIES.items():
        (corpus / name).write_text(text, encoding="utf-8")

    rag = AdaptiveRAG(vector_store_provider="faiss", chunk_size=160, chunk_overlap=20,
                      retrieval_k=2, advanced_use_reranking=False)
    chunks = rag.add_documents(corpus)
    chat = AgentChat(relevant="yes")
    rag._advanced_rag.llm._llm = chat

    reranking = []
    search = rag._advanced_rag._search

    def spy(query, k, use_hybrid=None, use_reranking=None):
        reranking.append(use_reranking)
        return search(query, k, use_hybrid=use_hybrid, use_reranking=use_reranking)

    rag._advanced_rag._search = spy
    return rag, chat, chunks, built, reranking


def test_adaptive_routes_share_one_index(adaptive):
    rag, chat, chunks, built, reranking = adaptive

    assert built == [1]  # one embedding model for every route
    assert chunks > 0 and rag.num_chunks == chunks
    assert rag._agentic_rag.vector_store is None
    assert rag._agentic_rag.llm is rag._advanced_rag.llm

    assert "búa" in rag.query(QUESTION, force_route="simple")
    assert "búa" in rag.query(QUESTION, force_route="medium")
    complex_result = rag.query_with_sources(QUESTION, force_route="complex")

    # simple skips the cross-encoder; medium and the agent use the configured search
    assert reranking == [False, None, None]
    assert complex_result["route"] == "complex"
    assert complex_result["sources"] and complex_result["sources"][0]["source"].endswith("thach_sanh.md")


def test_adaptive_falls_back_to_a_simpler_route_and_never_answers_without_retrieval(adaptive):
    rag, chat, *_ = adaptive

    def broken(*_args, **_kwargs):
        raise RuntimeError("agent down")

    rag._agentic_rag.query = broken
    assert "búa" in rag.query(QUESTION, force_route="complex")

    rag._advanced_rag.query = broken
    calls = len(chat.prompts)
    answer = rag.query(QUESTION, force_route="complex")
    assert "unable to process" in answer
    assert len(chat.prompts) == calls  # no ungrounded LLM answer

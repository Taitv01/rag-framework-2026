"""
Tests for API SSE Streaming & Langfuse Tracing
==============================================
"""


import asyncio
import json
import threading

import httpx

from src import __version__
from unittest.mock import Mock
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.monitoring import LangfuseTracer


def test_langfuse_tracer_graceful_fallback():
    """Verify LangfuseTracer degrades gracefully without keys."""
    tracer = LangfuseTracer(enabled=False)
    assert tracer.enabled is False
    
    trace = tracer.start_trace("test_op", input_data="hello")
    assert trace["name"] == "test_op"
    
    duration = tracer.end_trace(trace, output="world")
    assert duration >= 0.0


def test_api_health_endpoint():
    """Verify health endpoint exposes the package version and tracing status."""
    app = create_app(rag_type="naive", vector_store_provider="faiss")
    
    # Mock heavy RAG components
    mock_rag = Mock()
    mock_rag.num_documents = 10
    app.state.rag = mock_rag
    
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["version"] == __version__
    assert "tracing_enabled" in data


def sse_events(text):
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


def test_api_query_stream_naive_retrieves_once_and_sends_the_answer():
    """Without token streaming the naive pipeline still sends sources, answer, done."""
    app = create_app(rag_type="naive", vector_store_provider="faiss")
    mock_rag = Mock()
    mock_rag.query_with_sources.return_value = {
        "answer": "Tiếng Việt RAG xử lý tốt.",
        "sources": [{"content": "Tiếng Việt RAG framework 2026.", "metadata": {"source": "test.txt"}}],
    }
    app.state.rag = mock_rag

    response = TestClient(app).post("/query/stream", json={"question": "Tiếng Việt RAG?", "k": 3})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = sse_events(response.text)
    assert [e["status"] for e in events] == ["sources", "generating", "done"]
    assert events[1]["chunk"] == "Tiếng Việt RAG xử lý tốt."
    mock_rag.query_with_sources.assert_called_once_with("Tiếng Việt RAG?", k=3, filter=None)
    mock_rag.retrieve.assert_not_called()


def stream_client(rag):
    app = create_app(rag_type="naive", vector_store_provider="faiss")
    app.state.rag = rag
    app.state.rag_type = "advanced"
    return app


def test_api_query_stream_sends_the_sources_of_the_prompt(make_rag, monkeypatch):
    rag, chat = make_rag(reply=lambda prompt: "Bống ở giếng [S1].")
    retrievals = []
    retrieve = rag._retrieve
    monkeypatch.setattr(rag, "_retrieve", lambda *a, **kw: retrievals.append(kw) or retrieve(*a, **kw))

    response = TestClient(stream_client(rag)).post(
        "/query/stream",
        json={"question": "Tấm gọi bống thế nào?", "k": 1, "filter": {"file_name": "tam_cam.md"}},
    )

    events = sse_events(response.text)
    statuses = [e["status"] for e in events]
    assert statuses[0] == "sources" and statuses[-1] == "done"
    assert set(statuses[1:-1]) == {"generating"}
    sources = events[0]["sources"]
    assert [s["source_id"] for s in sources] == ["S1"]
    assert sources[0]["metadata"]["file_name"] == "tam_cam.md"
    # Exactly the passage the LLM was given, retrieved once with the request's k and filter.
    assert "[S1] Source:" in chat.prompts[0] and sources[0]["content"][:40] in chat.prompts[0]
    assert len(retrievals) == 1 and retrievals[0]["k"] == 1
    assert retrievals[0]["filter"] == {"file_name": "tam_cam.md"}
    assert "".join(e["chunk"] for e in events if e["status"] == "generating").strip() == "Bống ở giếng [S1]."
    assert [c["source_id"] for c in events[-1]["citations"]] == ["S1"]


def test_api_query_stream_does_not_block_the_event_loop(make_rag):
    rag, chat = make_rag()
    scripted_stream = chat.stream
    health_answered = threading.Event()
    waits = []

    def slow_stream(messages, **kwargs):
        # A provider slow to send its first token: it is still waiting when
        # /health runs, unless the waiting blocks the event loop itself.
        waits.append(health_answered.wait(timeout=1.0))
        yield from scripted_stream(messages, **kwargs)

    chat.stream = slow_stream
    app = stream_client(rag)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            stream_task = asyncio.create_task(
                client.post("/query/stream", json={"question": "Thạch Sanh giết chằn tinh bằng gì?"})
            )
            await asyncio.sleep(0.05)
            health = await client.get("/health")
            health_answered.set()
            return health, await stream_task

    health, response = asyncio.run(run())

    assert health.status_code == 200 and response.status_code == 200
    assert waits == [True]  # /health was served while the LLM stream waited
    assert sse_events(response.text)[-1]["status"] == "done"


def test_stopping_early_closes_the_blocking_iterator():
    from src.api.app import _iterate_in_thread

    closed = []

    def tokens():
        try:
            yield from ["a", "b", "c"]
        finally:
            closed.append(True)

    async def consume_one():
        stream = _iterate_in_thread(tokens())
        first = await stream.__anext__()
        await stream.aclose()  # what Starlette does when the client goes away
        return first

    assert asyncio.run(consume_one()) == "a"
    assert closed == [True]

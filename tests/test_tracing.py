"""Pipeline steps: local timings, and Langfuse observations nested per request (no network)."""

import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.monitoring.tracing import record_steps, set_langfuse_client, step

USAGE = {"input_tokens": 120, "output_tokens": 8, "total_tokens": 128}


def with_usage(chat):
    """Make a ScriptedChat report token usage like a real provider."""
    invoke = chat.invoke

    def invoke_with_usage(messages, **kwargs):
        return SimpleNamespace(content=invoke(messages, **kwargs).content, usage_metadata=USAGE)

    chat.invoke = invoke_with_usage
    return chat


def test_query_detailed_reports_each_step_with_its_duration(make_rag):
    rag, chat = make_rag()
    with_usage(chat)

    steps = rag.query_detailed("thach sanh giet chan tinh bang gi")["steps"]

    names = [s["name"] for s in steps]
    assert names == ["rewrite", "llm", "retrieve", "generate", "llm"]
    assert all(s["ms"] >= 0 for s in steps)
    generation = steps[-1]
    assert generation["type"] == "generation" and generation["model"] == rag.llm.config.model
    assert generation["usage"] == {"input": 120, "output": 8, "total": 128}


def test_steps_are_free_without_a_consumer():
    with step("retrieve") as s:
        s.update(output={"documents": 3}, usage={"input": 1})
    with record_steps() as steps:
        with step("outer"):
            with step("inner", as_type="generation", model="m"):
                pass
    assert [(s["name"], s["type"]) for s in steps] == [("outer", "span"), ("inner", "generation")]


@pytest.fixture
def langfuse_spans():
    """A real Langfuse client whose spans land in memory instead of the network."""
    langfuse = pytest.importorskip("langfuse")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from src.monitoring import LangfuseTracer

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The SDK shares one exporter per public key: a fresh key per test.
    client = langfuse.Langfuse(
        public_key=f"pk-lf-{uuid.uuid4().hex}", secret_key="sk-lf-test", base_url="http://127.0.0.1:9",
        tracer_provider=provider, should_export_span=lambda span: False,
    )
    tracer = LangfuseTracer(client=client)
    try:
        yield tracer, exporter
    finally:
        set_langfuse_client(None)


def by_name(spans):
    found = {}
    for span in spans:
        found.setdefault(span.name, []).append(span)
    return found


def parent_name(span, spans):
    ids = {s.context.span_id: s.name for s in spans}
    return ids.get(span.parent.span_id) if span.parent else None


def attribute(span, name):
    from langfuse import LangfuseOtelSpanAttributes

    return span.attributes.get(getattr(LangfuseOtelSpanAttributes, name))


def traced_api(rag, tracer):
    from src.api.app import create_app

    app = create_app(rag_type="naive", vector_store_provider="faiss")
    app.state.rag = rag
    app.state.rag_type = "advanced"
    app.state.tracer = tracer
    return TestClient(app)


def test_query_trace_nests_every_step_with_model_and_tokens(make_rag, langfuse_spans):
    tracer, exporter = langfuse_spans
    rag, chat = make_rag()
    with_usage(chat)

    response = traced_api(rag, tracer).post("/query", json={"question": "Thạch Sanh giết chằn tinh bằng gì?"})

    assert response.status_code == 200
    spans = exporter.get_finished_spans()
    found = by_name(spans)
    assert set(found) == {"query", "retrieve", "generate", "llm"}
    assert {span.context.trace_id for span in spans} == {found["query"][0].context.trace_id}
    assert parent_name(found["retrieve"][0], spans) == "query"
    assert parent_name(found["generate"][0], spans) == "query"
    llm = found["llm"][0]
    assert parent_name(llm, spans) == "generate"
    assert attribute(llm, "OBSERVATION_TYPE") == "generation"
    assert attribute(llm, "OBSERVATION_MODEL") == rag.llm.config.model
    assert json.loads(attribute(llm, "OBSERVATION_USAGE_DETAILS")) == {"input": 120, "output": 8, "total": 128}
    assert attribute(found["retrieve"][0], "OBSERVATION_TYPE") == "retriever"
    assert "dùng búa" in attribute(found["query"][0], "OBSERVATION_OUTPUT")


def test_stream_trace_spans_the_worker_threads(make_rag, langfuse_spans):
    tracer, exporter = langfuse_spans
    rag, _ = make_rag()

    response = traced_api(rag, tracer).post("/query/stream", json={"question": "Thạch Sanh giết chằn tinh bằng gì?"})

    assert response.status_code == 200 and '"status": "done"' in response.text
    spans = exporter.get_finished_spans()
    found = by_name(spans)
    # Retrieval and the token stream ran in worker threads, still under the request's trace.
    assert parent_name(found["retrieve"][0], spans) == "query_stream"
    assert parent_name(found["llm"][0], spans) == "query_stream"
    assert "Thạch Sanh dùng búa [S1]." in attribute(found["llm"][0], "OBSERVATION_OUTPUT")
    assert len({span.context.trace_id for span in spans}) == 1

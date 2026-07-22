"""
Tests for API SSE Streaming & Langfuse Tracing
==============================================
"""

import pytest
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient
from langchain_core.documents import Document
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
    """Verify health endpoint contains version 1.1.0 and tracing status."""
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
    assert data["version"] == "1.1.0"
    assert "tracing_enabled" in data


def test_api_query_stream_sse_endpoint():
    """Verify /query/stream SSE streaming endpoint."""
    app = create_app(rag_type="naive", vector_store_provider="faiss")
    
    # Mock RAG streaming
    mock_rag = Mock()
    mock_rag.num_documents = 1
    mock_rag.retrieve.return_value = [
        Document(page_content="Tiếng Việt RAG framework 2026.", metadata={"source": "test.txt"})
    ]
    mock_rag.stream.return_value = ["Tiếng ", "Việt ", "RAG ", "xử lý ", "tốt."]
    app.state.rag = mock_rag
    
    client = TestClient(app)
    response = client.post("/query/stream", json={"question": "Tiếng Việt RAG?"})
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    content = response.text
    assert "data:" in content
    assert "sources" in content
    assert "generating" in content
    assert "done" in content

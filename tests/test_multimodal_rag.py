"""Tests for image/video-aware LLM, RAG, and FastAPI paths."""

import base64
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.api.app import create_app
from src.core.llm import LLMManager
from src.rag.advanced_rag import AdvancedRAG
from src.rag.naive_rag import NaiveRAG


def test_llm_manager_builds_openrouter_multimodal_content_blocks():
    manager = LLMManager(provider="openai", model="stealth/ox-alpha")
    manager._llm = Mock()
    manager._llm.invoke.return_value = SimpleNamespace(
        content=[{"type": "text", "text": "Đã phân tích media."}]
    )

    answer = manager.generate_multimodal(
        "Mô tả nội dung",
        [
            {"type": "image", "url": "data:image/png;base64,AAAA"},
            {"type": "video", "url": "https://example.com/clip.mp4"},
        ],
        system_prompt="Be precise",
    )

    messages = manager._llm.invoke.call_args.args[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == [
        {"type": "text", "text": "Mô tả nội dung"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        },
        {
            "type": "video_url",
            "video_url": {"url": "https://example.com/clip.mp4"},
        },
    ]
    assert answer == "Đã phân tích media."


def test_llm_manager_rejects_invalid_multimodal_input():
    manager = LLMManager(provider="openai", model="stealth/ox-alpha")
    manager._llm = Mock()

    with pytest.raises(ValueError, match="Unsupported media type"):
        manager.generate_multimodal(
            "Analyze",
            [{"type": "audio", "url": "https://example.com/audio.mp3"}],
        )


def test_naive_rag_multimodal_can_run_without_indexed_documents():
    rag = NaiveRAG.__new__(NaiveRAG)
    rag._chunks = []
    rag.llm = Mock()
    rag.llm.config = SimpleNamespace(model="stealth/ox-alpha")
    rag.llm.generate_multimodal.return_value = "Một con mèo trong ảnh."

    result = rag.query_multimodal(
        "Có gì trong ảnh?",
        [{"type": "image", "url": "data:image/png;base64,AAAA"}],
    )

    assert result["answer"] == "Một con mèo trong ảnh."
    assert result["sources"] == []
    assert result["media_count"] == 1


def test_advanced_rag_multimodal_includes_retrieved_context():
    rag = AdvancedRAG.__new__(AdvancedRAG)
    rag._chunks = [Document(page_content="indexed")]
    rag.retrieval_k = 5
    rag.llm = Mock()
    rag.llm.config = SimpleNamespace(model="stealth/ox-alpha")
    rag.llm.generate_multimodal.return_value = "Doanh thu trong biểu đồ tăng. [S1]"
    rag._retrieve = Mock(return_value=[
        Document(
            page_content="Doanh thu mục tiêu là 200 triệu.",
            metadata={"source": "report.md"},
        )
    ])
    rag._build_context = AdvancedRAG._build_context.__get__(rag, AdvancedRAG)
    rag._format_sources = AdvancedRAG._format_sources.__get__(rag, AdvancedRAG)
    rag._context_validator = None

    result = rag.query_multimodal(
        "Biểu đồ có đạt mục tiêu không?",
        [{"type": "image", "url": "https://example.com/chart.png"}],
    )

    prompt = rag.llm.generate_multimodal.call_args.args[0]
    assert "Doanh thu mục tiêu là 200 triệu" in prompt
    assert result["citations"][0]["source_id"] == "S1"


def _multimodal_test_app(**kwargs):
    app = create_app(
        rag_type="naive",
        vector_store_provider="faiss",
        **kwargs,
    )
    rag = Mock()
    rag.query_multimodal.return_value = {
        "answer": "Phân tích thành công",
        "sources": [],
        "citations": [],
        "model": "stealth/ox-alpha",
    }
    app.state.rag = rag
    return app, rag


def test_multimodal_upload_api_forwards_image_and_video_data_urls():
    app, rag = _multimodal_test_app()
    client = TestClient(app)

    response = client.post(
        "/query/multimodal",
        data={"question": "Mô tả media", "k": "3", "use_retrieval": "false"},
        files=[
            ("files", ("frame.png", b"\x89PNG\r\n", "image/png")),
            ("files", ("clip.mov", b"video-bytes", "video/quicktime")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Phân tích thành công"
    assert payload["model"] == "stealth/ox-alpha"
    assert payload["media"][1]["content_type"] == "video/mov"

    call = rag.query_multimodal.call_args.kwargs
    assert call["k"] == 3
    assert call["use_retrieval"] is False
    assert call["media"][0]["url"].startswith("data:image/png;base64,")
    assert call["media"][1]["url"].startswith("data:video/mov;base64,")
    encoded_image = call["media"][0]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded_image) == b"\x89PNG\r\n"


def test_multimodal_url_api_forwards_remote_video_without_echoing_url():
    app, rag = _multimodal_test_app()
    client = TestClient(app)

    response = client.post(
        "/query/multimodal/url",
        json={
            "question": "Tóm tắt video",
            "media": [{"type": "video", "url": "https://example.com/video.mp4"}],
            "use_retrieval": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["media"] == [{"type": "video", "source": "url"}]
    assert rag.query_multimodal.call_args.kwargs["media"] == [
        {"type": "video", "url": "https://example.com/video.mp4"}
    ]


def test_multimodal_api_rejects_unsupported_media_and_oversized_upload():
    app, rag = _multimodal_test_app(max_upload_size_mb=1)
    client = TestClient(app)

    unsupported = client.post(
        "/query/multimodal",
        data={"question": "Analyze"},
        files=[("files", ("notes.txt", b"text", "text/plain"))],
    )
    oversized = client.post(
        "/query/multimodal",
        data={"question": "Analyze"},
        files=[("files", ("large.png", b"x" * (1024 * 1024 + 1), "image/png"))],
    )

    assert unsupported.status_code == 415
    assert oversized.status_code == 413
    rag.query_multimodal.assert_not_called()


def test_multimodal_url_api_rejects_non_http_url():
    app, rag = _multimodal_test_app()
    client = TestClient(app)

    response = client.post(
        "/query/multimodal/url",
        json={
            "question": "Analyze",
            "media": [{"type": "image", "url": "file:///etc/passwd"}],
        },
    )

    assert response.status_code == 400
    rag.query_multimodal.assert_not_called()


def test_multimodal_url_api_rejects_embedded_credentials():
    app, rag = _multimodal_test_app()
    client = TestClient(app)

    response = client.post(
        "/query/multimodal/url",
        json={
            "question": "Analyze",
            "media": [
                {"type": "image", "url": "https://user:secret@example.com/image.png"}
            ],
        },
    )

    assert response.status_code == 400
    rag.query_multimodal.assert_not_called()


def test_ox_status_never_exposes_api_key():
    app, _ = _multimodal_test_app(
        ox_api_key="openrouter-secret",
        ox_model="stealth/ox-alpha",
    )
    client = TestClient(app)

    response = client.get("/ox/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "configured",
        "configured": True,
        "provider": "openrouter",
        "model": "stealth/ox-alpha",
        "modalities": ["text", "image", "video"],
    }
    assert "openrouter-secret" not in response.text


def test_ox_chat_uses_dedicated_model_and_rag_context():
    app, rag = _multimodal_test_app(ox_api_key="openrouter-secret")
    rag.num_chunks = 1
    rag.retrieve.return_value = [
        Document(
            page_content="Doanh thu mục tiêu là 200 triệu.",
            metadata={"source": "report.md"},
        )
    ]
    ox_llm = Mock()
    ox_llm.config = SimpleNamespace(model="stealth/ox-alpha")
    ox_llm.generate.return_value = "Biểu đồ đạt mục tiêu. [S1]"
    app.state.ox_llm = ox_llm
    client = TestClient(app)

    response = client.post(
        "/ox/chat",
        json={"question": "Mục tiêu doanh thu là bao nhiêu?", "k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "stealth/ox-alpha"
    assert payload["citations"][0]["source_id"] == "S1"
    assert "Doanh thu mục tiêu là 200 triệu" in ox_llm.generate.call_args.args[0]
    rag.retrieve.assert_called_once_with("Mục tiêu doanh thu là bao nhiêu?", k=3)


def test_ox_analyze_upload_forces_dedicated_ox_model():
    app, rag = _multimodal_test_app(ox_api_key="openrouter-secret")
    client = TestClient(app)

    response = client.post(
        "/ox/analyze",
        data={"question": "Ảnh này có gì?", "use_retrieval": "false"},
        files=[("files", ("frame.png", b"\x89PNG\r\n", "image/png"))],
    )

    assert response.status_code == 200
    call = rag.query_multimodal.call_args.kwargs
    assert call["llm"] is app.state.ox_llm
    assert call["use_retrieval"] is False
    assert call["media"][0]["type"] == "image"


def test_ox_routes_fail_fast_when_openrouter_key_is_missing():
    app, rag = _multimodal_test_app(ox_api_key=None)
    app.state.ox_configured = False
    client = TestClient(app)

    chat = client.post("/ox/chat", json={"question": "Xin chào"})
    analyze = client.post(
        "/ox/analyze/url",
        json={
            "question": "Mô tả video",
            "media": [{"type": "video", "url": "https://example.com/video.mp4"}],
        },
    )

    assert chat.status_code == 503
    assert analyze.status_code == 503
    assert chat.json()["detail"] == "Ox AI is not configured. Set OPENROUTER_API_KEY."
    rag.query_multimodal.assert_not_called()

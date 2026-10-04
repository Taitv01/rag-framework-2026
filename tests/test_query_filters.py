"""Metadata filters on queries, cache entries per filter, API query defaults (no models)."""

from unittest.mock import Mock

from fastapi.testclient import TestClient

QUESTION = "Bống và chằn tinh trong truyện nào?"


def sources_of(result):
    return {source["metadata"]["file_name"] for source in result["relevant_docs"]}


def test_filter_limits_retrieval_and_answers_to_matching_documents(make_rag):
    rag, chat = make_rag()

    assert sources_of(rag.query_detailed(QUESTION)) == {"thach_sanh.md", "tam_cam.md"}
    assert sources_of(rag.query_detailed(QUESTION, filter={"file_name": "tam_cam.md"})) == {"tam_cam.md"}
    assert {doc.metadata["file_name"] for doc in rag.retrieve(QUESTION, filter={"file_name": "thach_sanh.md"})} \
        == {"thach_sanh.md"}
    # A list matches any of its values.
    both = rag.query_detailed(QUESTION, filter={"file_name": ["tam_cam.md", "thach_sanh.md"]})
    assert sources_of(both) == {"thach_sanh.md", "tam_cam.md"}


def test_filter_matching_nothing_abstains_without_an_llm_call(make_rag):
    rag, chat = make_rag()

    result = rag.query_detailed(QUESTION, filter={"file_name": "khong_co.md"})

    assert result["abstained"] and result["relevant_docs"] == []
    assert chat.prompts == []


def test_cached_answers_are_reused_only_for_the_same_filter_and_k(make_rag):
    rag, chat = make_rag(use_cache=True)
    tam_cam = {"file_name": "tam_cam.md"}

    assert not rag.query_detailed(QUESTION, filter=tam_cam)["cache_hit"]
    assert rag.query_detailed(QUESTION, filter=tam_cam)["cache_hit"]
    assert not rag.query_detailed(QUESTION, filter={"file_name": "thach_sanh.md"})["cache_hit"]
    assert not rag.query_detailed(QUESTION)["cache_hit"]
    assert not rag.query_detailed(QUESTION, k=1, filter=tam_cam)["cache_hit"]
    assert len(chat.prompts) == 4


def api_with(rag, rag_type):
    from src.api.app import create_app

    app = create_app(rag_type="naive", vector_store_provider="faiss")
    app.state.rag = rag
    app.state.rag_type = rag_type
    return TestClient(app)


def test_query_api_follows_pipeline_defaults_and_passes_the_filter():
    rag = Mock()
    rag.query_detailed.return_value = {
        "answer": "Tôi không có đủ thông tin.", "relevant_docs": [], "citations": [],
        "transformed_query": None, "abstained": True,
    }
    client = api_with(rag, "advanced")

    response = client.post("/query", json={"question": QUESTION, "filter": {"file_name": "tam_cam.md"}})

    assert response.status_code == 200 and response.json()["abstained"] is True
    rag.query_detailed.assert_called_once_with(
        question=QUESTION, k=None, transform_query=None, grade_documents=None,
        filter={"file_name": "tam_cam.md"},
    )
    assert client.post("/query", json={"question": QUESTION, "k": 0}).status_code == 422
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_search_api_takes_an_optional_filter_body():
    rag = Mock()
    rag.retrieve.return_value = []
    client = api_with(rag, "advanced")

    assert client.post("/search", params={"query": QUESTION}).status_code == 200
    rag.retrieve.assert_called_with(QUESTION, k=5, filter=None)

    client.post("/search", params={"query": QUESTION, "k": 2}, json={"filter": {"file_name": "tam_cam.md"}})
    rag.retrieve.assert_called_with(QUESTION, k=2, filter={"file_name": "tam_cam.md"})

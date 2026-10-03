"""VectorStoreManager behaves the same on FAISS and Qdrant (no models, no server)."""

import importlib.util
import os
import uuid

import pytest
from langchain_core.documents import Document

from src.core.vector_store import VectorStoreManager, qdrant_point_id
from tests.fakes import fake_embeddings_manager

HAS_QDRANT = importlib.util.find_spec("qdrant_client") and importlib.util.find_spec("langchain_qdrant")
needs_qdrant = pytest.mark.skipif(not HAS_QDRANT, reason="qdrant extra not installed")

DOCS = [
    Document(page_content="Thạch Sanh dùng búa chặt đầu chằn tinh", metadata={"source": "thach_sanh.md", "root": "a"}),
    Document(page_content="Tấm gọi cá bống lên ăn cơm", metadata={"source": "tam_cam.md", "root": "a"}),
    Document(page_content="Sơn Tinh dời núi chặn nước lũ", metadata={"source": "son_tinh.md", "root": "b"}),
]
IDS = ["chunk-thach-sanh", "chunk-tam-cam", "chunk-son-tinh"]


def make_store(provider, tmp_path, name="default"):
    if provider == "faiss":
        return VectorStoreManager("faiss", fake_embeddings_manager(), collection_name=name,
                                  persist_directory=str(tmp_path / "faiss"))
    if provider == "qdrant-memory":
        return VectorStoreManager("qdrant", fake_embeddings_manager(), collection_name=name, url=":memory:")
    if provider == "qdrant-server":
        # A fresh collection per test on the shared server (CI service container).
        return VectorStoreManager("qdrant", fake_embeddings_manager(),
                                  collection_name=f"{name}_{uuid.uuid4().hex[:8]}",
                                  url=os.environ["QDRANT_TEST_URL"])
    return VectorStoreManager("qdrant", fake_embeddings_manager(), collection_name=name,
                              persist_directory=str(tmp_path / "qdrant"))


PROVIDERS = [
    "faiss",
    pytest.param("qdrant-memory", marks=needs_qdrant),
    pytest.param("qdrant-disk", marks=needs_qdrant),
    pytest.param("qdrant-server", marks=[
        needs_qdrant,
        pytest.mark.skipif(not os.getenv("QDRANT_TEST_URL"), reason="set QDRANT_TEST_URL to a Qdrant server"),
    ]),
]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_search_filter_and_get_all(provider, tmp_path):
    store = make_store(provider, tmp_path)
    store.add_documents(DOCS, ids=IDS)

    assert store.count() == 3
    assert store.similarity_search("chằn tinh búa", k=1)[0].metadata["source"] == "thach_sanh.md"
    filtered = store.similarity_search("chằn tinh búa", k=3, filter={"root": "b"})
    assert [d.metadata["source"] for d in filtered] == ["son_tinh.md"]
    assert sorted(d.metadata["source"] for d in store.get_all_documents()) == [
        "son_tinh.md", "tam_cam.md", "thach_sanh.md"]
    store.close()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_same_id_overwrites_and_delete(provider, tmp_path):
    store = make_store(provider, tmp_path)
    store.add_documents(DOCS, ids=IDS)
    store.add_documents([Document(page_content="Thạch Sanh gảy đàn thần", metadata={"source": "thach_sanh.md", "root": "a"})],
                        ids=["chunk-thach-sanh"])
    assert store.count() == 3
    assert "đàn thần" in store.similarity_search("đàn thần", k=1)[0].page_content

    store.delete(ids=["chunk-tam-cam", "never-added"])
    assert store.count() == 2
    store.delete(filter={"root": "a"})
    assert [d.metadata["source"] for d in store.get_all_documents()] == ["son_tinh.md"]
    store.close()


@pytest.mark.parametrize("provider", ["faiss", pytest.param("qdrant-disk", marks=needs_qdrant)])
def test_data_survives_a_restart(provider, tmp_path):
    store = make_store(provider, tmp_path)
    store.add_documents(DOCS, ids=IDS)
    store.persist()
    store.close()

    reopened = make_store(provider, tmp_path)
    assert reopened.count() == 3
    assert reopened.similarity_search("cá bống", k=1)[0].metadata["source"] == "tam_cam.md"
    reopened.close()


def test_qdrant_point_ids_are_stable_uuids():
    assert qdrant_point_id("chunk-a") == qdrant_point_id("chunk-a") != qdrant_point_id("chunk-b")
    assert qdrant_point_id("6f1d3c1e-7a51-4c55-9b0a-2f4d1b9e8c11") == "6f1d3c1e-7a51-4c55-9b0a-2f4d1b9e8c11"
    assert qdrant_point_id(7) == 7

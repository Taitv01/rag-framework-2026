"""Hybrid search details: BM25 filters, queries without diacritics, reranker reuse."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.core.retriever import RetrieverManager, fold_vietnamese, has_diacritics
from src.core.vector_store import VectorStoreManager
from tests.fakes import fake_embeddings_manager

pytest.importorskip("rank_bm25")

DOCS = [
    Document(page_content="Thạch Sanh dùng búa chặt đầu chằn tinh", metadata={"source": "thach_sanh.md", "root": "a"}),
    Document(page_content="Tấm gọi cá bống lên ăn cơm vàng cơm bạc", metadata={"source": "tam_cam.md", "root": "a"}),
    Document(page_content="Sơn Tinh dời núi chặn nước lũ của Thủy Tinh", metadata={"source": "son_tinh.md", "root": "b"}),
]


def make_retriever(**kwargs):
    store = VectorStoreManager("faiss", fake_embeddings_manager())
    store.add_documents(DOCS, ids=["a", "b", "c"])
    return RetrieverManager(store, documents=DOCS, k=2, use_hybrid=True, **kwargs)


def test_fold_vietnamese():
    assert fold_vietnamese("Thạch Sanh giết Chằn Tinh ở Đà Lạt") == "thach sanh giet chan tinh o da lat"
    assert has_diacritics("Sọ Dừa") and has_diacritics("đi") and not has_diacritics("so dua chan bo")


def test_query_without_diacritics_matches_through_bm25():
    retriever = make_retriever()
    bm25 = retriever._bm25_scores("thach sanh chan tinh")
    assert max(range(len(DOCS)), key=lambda i: bm25[i]) == 0
    # A query with diacritics keeps using the original word-segmented index.
    assert list(retriever._bm25_scores("Thạch Sanh")) == list(
        retriever._bm25.get_scores(retriever._tokenize_for_bm25("Thạch Sanh")))


def test_hybrid_filter_applies_to_bm25_results():
    retriever = make_retriever()
    results = retriever.hybrid_search("Thạch Sanh chằn tinh", k=3, filter={"root": "b"})
    assert [doc.metadata["source"] for doc in results] == ["son_tinh.md"]


class FakeCrossEncoder:
    loads = 0

    def __init__(self, name, device=None):
        FakeCrossEncoder.loads += 1

    def predict(self, pairs, **_kwargs):
        return [len(set(query.split()) & set(text.split())) for query, text in pairs]


def test_reranker_is_loaded_once_and_reused():
    FakeCrossEncoder.loads = 0
    with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(CrossEncoder=FakeCrossEncoder)}):
        first = make_retriever(use_reranking=True)
        second = make_retriever(use_reranking=True, reranker=first._reranker,
                                reranker_model=first.active_reranker_model)
    assert FakeCrossEncoder.loads == 1
    assert second._reranker is first._reranker
    assert second.active_reranker_model == first.config.reranker_model
    top = second.rerank("Tấm gọi cá bống", list(DOCS), k=1)[0]
    assert top.metadata["source"] == "tam_cam.md"


def test_failed_reranker_load_disables_reranking_without_switching_models():
    class Broken:
        def __init__(self, name, device=None):
            raise ValueError("Unrecognized processing class")

    with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(CrossEncoder=Broken)}):
        retriever = make_retriever(use_reranking=True)
    assert retriever._reranker is None and retriever.active_reranker_model is None

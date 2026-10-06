"""A question's embedding is computed once across collections (no models)."""

from src.core.embeddings import EmbeddingsManager, QueryCachedEmbeddings
from tests.fakes import HashEmbeddings


class CountingEmbeddings(HashEmbeddings):
    def __init__(self):
        super().__init__()
        self.queries = 0

    def embed_query(self, text):
        self.queries += 1
        return super().embed_query(text)


def test_a_repeated_query_is_embedded_once():
    inner = CountingEmbeddings()
    cached = QueryCachedEmbeddings(inner, size=2)

    vector = cached.embed_query("Thạch Sanh")
    vector.append(9.9)  # callers cannot change the cached vector
    assert len(cached.embed_query("Thạch Sanh")) == inner.dim and inner.queries == 1

    cached.embed_query("Tấm")
    cached.embed_query("Cám")  # size 2: "Thạch Sanh" is forgotten
    cached.embed_query("Thạch Sanh")
    assert inner.queries == 4

    assert len(cached.embed_documents(["a", "b"])) == 2 and inner.documents_embedded == 2
    assert cached.dim == inner.dim  # the model's own attributes stay reachable


def test_the_manager_caches_queries_of_its_model(monkeypatch):
    manager = EmbeddingsManager(provider="huggingface")
    monkeypatch.setattr(manager, "_create_embeddings", CountingEmbeddings)

    assert isinstance(manager.embeddings, QueryCachedEmbeddings)
    manager.embed_query("Sọ Dừa")
    manager.embeddings.embed_query("Sọ Dừa")
    assert manager.embeddings.inner.queries == 1

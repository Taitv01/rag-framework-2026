"""AdvancedRAG on a persistent store: restarts, refreshes and parent context (no models)."""

import importlib.util

import pytest

from tests.fakes import fake_embeddings_manager

HAS_QDRANT = importlib.util.find_spec("qdrant_client") and importlib.util.find_spec("langchain_qdrant")
PROVIDERS = ["faiss", pytest.param("qdrant", marks=pytest.mark.skipif(not HAS_QDRANT, reason="qdrant extra"))]

STORIES = {
    "thach_sanh.md": (
        "# Thạch Sanh\n\nThạch Sanh sống một mình dưới gốc đa. Cả gia tài chỉ có một lưỡi búa "
        "của cha để lại. Hằng ngày chàng vào rừng đốn củi đổi gạo. Ngọc Hoàng sai thiên thần "
        "xuống dạy chàng võ nghệ. Về sau chàng dùng búa chặt đầu chằn tinh hung dữ, xác nó hiện "
        "nguyên hình là con trăn khổng lồ bên cạnh bộ cung tên bằng vàng."
    ),
    "tam_cam.md": (
        "# Tấm Cám\n\nDì ghẻ hứa thưởng yếm đỏ cho đứa bắt đầy giỏ tép. Cám lừa chị xuống ao "
        "gội đầu rồi trút hết tép. Bụt bảo Tấm thả cá bống xuống giếng và gọi bống lên ăn cơm. "
        "Mẹ con Cám bắt bống làm thịt. Bụt bảo Tấm chôn xương bống dưới bốn chân giường."
    ),
}


def write_corpus(directory, names=STORIES):
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_text(STORIES[name], encoding="utf-8")
    return directory


def make_rag(monkeypatch, tmp_path, provider, **kwargs):
    monkeypatch.setattr("src.rag.advanced_rag.EmbeddingsManager", lambda **_: fake_embeddings_manager())
    from src.rag.advanced_rag import AdvancedRAG

    return AdvancedRAG(
        vector_store_provider=provider,
        persist_directory=str(tmp_path / f"store_{provider}"),
        chunk_size=160,
        chunk_overlap=20,
        retrieval_k=2,
        use_hybrid=True,
        use_reranking=False,
        **kwargs,
    )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_restart_keeps_the_index(monkeypatch, tmp_path, provider):
    corpus = write_corpus(tmp_path / "docs")
    rag = make_rag(monkeypatch, tmp_path, provider)
    rag.refresh_markdown_directory(corpus, manifest_path=tmp_path / "manifest.json")
    chunks, documents = rag.num_chunks, rag.num_documents
    rag.vector_store.close()

    restarted = make_rag(monkeypatch, tmp_path, provider)

    assert (restarted.num_chunks, restarted.num_documents) == (chunks, documents) == (chunks, 2)
    assert restarted._retriever is not None and restarted._retriever._bm25 is not None
    top = restarted.retrieve("chằn tinh trăn khổng lồ cung tên vàng", k=1)[0]
    assert top.metadata["source"].endswith("thach_sanh.md")
    restarted.vector_store.close()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_refresh_drops_removed_files_from_the_store(monkeypatch, tmp_path, provider):
    corpus = write_corpus(tmp_path / "docs")
    manifest = tmp_path / "manifest.json"
    rag = make_rag(monkeypatch, tmp_path, provider)
    rag.refresh_markdown_directory(corpus, manifest_path=manifest)

    (corpus / "tam_cam.md").unlink()
    result = rag.refresh_markdown_directory(corpus, manifest_path=manifest)

    stored = rag.vector_store.get_all_documents()
    assert result["removed"] == ["tam_cam.md"]
    assert rag.num_documents == 1
    assert len(stored) == rag.num_chunks
    assert all(doc.metadata["source"].endswith("thach_sanh.md") for doc in stored)
    rag.vector_store.close()


def test_parent_context_returns_whole_parents_once(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path / "docs")
    rag = make_rag(monkeypatch, tmp_path, "faiss", use_parent_context=True)
    rag.refresh_markdown_directory(corpus, manifest_path=tmp_path / "manifest.json")

    children = rag._retriever.search("bống giếng xương", k=6)
    parents = rag.retrieve("bống giếng xương", k=2)

    assert {child.metadata["parent_id"] for child in children} >= {p.metadata["parent_id"] for p in parents}
    assert len({p.metadata["parent_id"] for p in parents}) == len(parents) == 2
    assert all(len(p.page_content) >= len(c.page_content) for p in parents for c in children
               if c.metadata["parent_id"] == p.metadata["parent_id"])
    assert all("parent_text" not in p.metadata for p in parents)
    assert all("parent_text" not in s["metadata"] for s in rag._format_sources(children))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_refresh_embeds_only_changed_files(monkeypatch, tmp_path, provider):
    corpus = write_corpus(tmp_path / "docs")
    manifest = tmp_path / "manifest.json"
    rag = make_rag(monkeypatch, tmp_path, provider)
    rag.refresh_markdown_directory(corpus, manifest_path=manifest)
    embeddings = rag.vector_store.embeddings.embeddings
    thach_sanh_chunks = [c for c in rag._chunks if c.metadata["relative_source"] == "thach_sanh.md"]

    (corpus / "tam_cam.md").write_text(STORIES["tam_cam.md"] + " Tấm hóa thành chim vàng anh.", encoding="utf-8")
    before = embeddings.documents_embedded
    result = rag.refresh_markdown_directory(corpus, manifest_path=manifest)
    tam_cam_chunks = [c for c in rag._chunks if c.metadata["relative_source"] == "tam_cam.md"]

    assert result["updated"] == ["tam_cam.md"] and result["documents_loaded"] == 1
    assert embeddings.documents_embedded - before == len(tam_cam_chunks)
    assert [c for c in rag._chunks if c.metadata["relative_source"] == "thach_sanh.md"] == thach_sanh_chunks
    assert len(rag.vector_store.get_all_documents()) == rag.num_chunks
    assert any("chim vàng anh" in c.page_content for c in tam_cam_chunks)
    rag.vector_store.close()


def test_unchanged_folder_is_indexed_when_the_store_is_empty(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path / "docs")
    manifest = tmp_path / "manifest.json"
    first = make_rag(monkeypatch, tmp_path, "faiss")
    first.refresh_markdown_directory(corpus, manifest_path=manifest)

    # A new process on an in-memory store: the manifest says "unchanged", the index is empty.
    monkeypatch.setattr("src.rag.advanced_rag.EmbeddingsManager", lambda **_: fake_embeddings_manager())
    from src.rag.advanced_rag import AdvancedRAG

    fresh = AdvancedRAG(vector_store_provider="faiss", chunk_size=160, chunk_overlap=20, use_reranking=False)
    result = fresh.refresh_markdown_directory(corpus, manifest_path=manifest)

    assert result["unchanged"] == ["tam_cam.md", "thach_sanh.md"]
    assert fresh.num_documents == 2 and fresh.num_chunks == first.num_chunks

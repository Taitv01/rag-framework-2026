"""
Shared pytest setup.

Tests must never see the developer's real credentials: automatic loading of
`.env` / `.env.local` is disabled and secret-looking variables inherited from
the shell are removed before any `src` module is imported.

Importing `src.api.app` builds the default app. Tests point its vector store
at an in-memory FAISS index in a throwaway directory, so they neither write
into the repository nor load embedding models; Qdrant has its own tests.
"""

import os
import tempfile

_SECRET_SUFFIXES = ("_API_KEY", "_SECRET_KEY", "_PUBLIC_KEY", "_PASSWORD", "_TOKEN")
_PROVIDER_SETTINGS = (
    "API_KEYS",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "OPENROUTER_BASE_URL",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBEDDING_PROVIDER",
    "DEFAULT_EMBEDDING_MODEL",
    "QDRANT_URL",
)

os.environ["RAG_DISABLE_DOTENV"] = "1"
for _name in list(os.environ):
    if _name.endswith(_SECRET_SUFFIXES) or _name in _PROVIDER_SETTINGS:
        del os.environ[_name]

os.environ["DEFAULT_VECTOR_STORE"] = "faiss"
os.environ["PERSIST_DIRECTORY"] = tempfile.mkdtemp(prefix="rag-tests-")


# Imported after the environment is cleaned.
import pytest  # noqa: E402

from tests.fakes import STORIES, ScriptedChat, default_reply, fake_embeddings_manager  # noqa: E402


@pytest.fixture
def make_rag(monkeypatch, tmp_path):
    """AdvancedRAG over two short stories, fake embeddings, a scripted chat model."""
    def build(reply=default_reply, **kwargs):
        monkeypatch.setattr("src.rag.advanced_rag.EmbeddingsManager", lambda **_: fake_embeddings_manager())
        from src.rag.advanced_rag import AdvancedRAG

        corpus = tmp_path / "docs"
        corpus.mkdir(exist_ok=True)
        for name, text in STORIES.items():
            (corpus / name).write_text(text, encoding="utf-8")
        rag = AdvancedRAG(vector_store_provider="faiss", chunk_size=160, chunk_overlap=20,
                          retrieval_k=2, use_reranking=False, **kwargs)
        rag.add_documents(corpus)
        chat = ScriptedChat(reply)
        rag.llm._llm = chat
        return rag, chat

    return build

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

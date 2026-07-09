# Ultimate RAG Framework

Ultimate RAG Framework is a Python toolkit for building Retrieval-Augmented Generation systems. It supports simple RAG pipelines, advanced retrieval, graph and agentic workflows, Vietnamese processing, and now update-safe Markdown folder ingestion.

This repository is suitable for a living knowledge base: reports, MMO notes, operating documents, story/worldbuilding files, and any `.md` folder where numbers and facts change over time.

## Main Capabilities

- Markdown folder refresh with content hashes and a persistent manifest.
- Multi-format document loading: `.txt`, `.md`, `.pdf`, `.docx`, `.html`, `.json`, `.csv`.
- Naive RAG and Advanced RAG pipelines.
- Hybrid retrieval, reranking, parent-child chunking, and query rewriting hooks.
- Agentic, graph, adaptive, and cross-story RAG modules for larger workflows.
- Vietnamese-aware processing and examples.
- Vector store support for FAISS, Chroma, and Qdrant.
- `.env.local` support for local secrets without committing them.

## New Markdown Refresh Flow

Use `refresh_markdown_directory()` when your `.md` files are updated regularly and you need the vector index to reflect the latest data.

The refresh flow:

1. Scans the Markdown folder recursively.
2. Computes SHA-256 hashes for each source file.
3. Compares the current folder state with `.rag_markdown_manifest.json`.
4. Reloads and rechunks the folder only when files are added, updated, removed, or when `force=True`.
5. Replaces old chunks for that folder instead of appending duplicates.
6. Stores stable `document_id`, `chunk_id`, and `chunk_sha256` metadata.

The returned result includes `added`, `updated`, `removed`, `changed`, `documents_loaded`, `chunks_indexed`, `rebuilt`, and `manifest_path`.

## Installation

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -e ".[dev]"
```

If you only need runtime dependencies:

```powershell
py -m pip install -r requirements.txt
```

## Local Configuration

Create `.env.local` in the repo root for local secrets. This file is ignored by Git.

```dotenv
OPENAI_API_KEY=<your-openai-api-key>
```

The config loader reads `.env` first, then `.env.local`, so local values can override shared defaults.

## Quick Start: Markdown Knowledge Base

```python
from src.rag import NaiveRAG

rag = NaiveRAG()

markdown_dir = r"D:\E & D Cá nhân\MMO_Project_2026\RAG"
result = rag.refresh_markdown_directory(markdown_dir)

print(result)
answer = rag.query("Tóm tắt số liệu mới nhất trong các file Markdown.")
print(answer)
```

Run the same call again after editing, adding, or deleting `.md` files. If nothing changed, the method returns quickly and does not rebuild the index.

To force a rebuild:

```python
result = rag.refresh_markdown_directory(markdown_dir, force=True)
```

To use a custom manifest path:

```python
result = rag.refresh_markdown_directory(
    markdown_dir,
    manifest_path=".cache/rag_markdown_manifest.json",
)
```

## Advanced RAG Example

```python
from src.rag import AdvancedRAG

rag = AdvancedRAG(
    chunk_size=1000,
    chunk_overlap=200,
    retrieval_method="hybrid",
)

result = rag.refresh_markdown_directory(r"D:\E & D Cá nhân\MMO_Project_2026\RAG")
print(result)

response = rag.query("Các số liệu quan trọng nào vừa thay đổi?")
print(response)
```

## Loading Markdown Directly

For lower-level workflows, use the document loader directly:

```python
from src.core import DocumentLoader

loader = DocumentLoader()
docs = loader.load_markdown_directory(r"D:\E & D Cá nhân\MMO_Project_2026\RAG")

for doc in docs:
    print(doc.metadata["relative_source"], doc.metadata["source_sha256"])
```

Each loaded Markdown document includes metadata such as `source`, `source_root`, `relative_source`, `source_sha256`, `source_mtime`, and `source_mtime_ns`.

## Vector Store Notes

- FAISS refresh rebuilds the in-memory vector store from the current chunk list.
- Chroma and Qdrant refresh delete chunks by `source_root`, then add the new chunks.
- Stable chunk IDs prevent avoidable duplication when a supported vector store accepts explicit IDs.

## Project Layout

```text
src/core/                 Core loaders, splitters, embeddings, LLM, vector store, Markdown indexer
src/rag/                  Naive, advanced, agentic, graph, adaptive, and cross-story RAG pipelines
src/agents/               Retrieval, grading, hallucination, and query rewrite agents
src/story/                Long-form story and consistency tooling
src/evaluation/           RAG metrics and evaluation helpers
examples/                 Runnable examples and demos
tests/                    Unit and pipeline tests
```

Key files for Markdown refresh:

```text
src/core/document_loader.py
src/core/markdown_index.py
src/rag/naive_rag.py
src/rag/advanced_rag.py
tests/test_markdown_index.py
tests/test_document_loader.py
tests/test_config_env.py
```

## Verification

Recommended checks before pushing changes:

```powershell
py -m compileall src tests
py -m pytest tests/test_document_loader.py tests/test_markdown_index.py tests/test_config_env.py -q
py -m pytest tests/test_rag_pipeline.py -q -k "not embeddings_manager"
```

The skipped `embeddings_manager` path is intentionally excluded in the focused command when a full embedding model download or provider-backed test is not needed.

## Data And Secret Safety

- Do not commit `.env.local` or any real API key.
- The Markdown manifest stores hashes and metadata, not secret values.
- If Markdown files contain sensitive business data, keep the repository and vector store private.
- Review generated answers against source documents before using them for reporting or financial decisions.

## License

MIT. See `LICENSE` for details.

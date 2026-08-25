# Configuration Guide

## Overview

The Ultimate RAG Framework uses a centralized configuration system that supports:
- Environment variables
- .env files
- Default values
- Programmatic configuration

## Environment Variables

### LLM Configuration

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Ox Alpha via OpenRouter
OX_API_KEY=your_openrouter_api_key_here
OX_BASE_URL=https://openrouter.ai/api/v1

# Default LLM settings
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.7
```

`OX_API_KEY` can also be supplied as `OPENROUTER_API_KEY`. Keeping a dedicated
Ox key name is recommended because it prevents an OpenAI credential from being
used against the wrong endpoint.

### Embedding Configuration

```bash
# Default embedding settings
DEFAULT_EMBEDDING_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Vector Store Configuration

```bash
# Default vector store
DEFAULT_VECTOR_STORE=faiss
DEFAULT_COLLECTION_NAME=default

# Qdrant (production)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Weaviate (production)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_weaviate_api_key_here
```

### RAG Configuration

```bash
# Text splitting
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Retrieval
RETRIEVAL_K=5

# Features
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
```

### Caching

```bash
ENABLE_CACHE=true
CACHE_TTL=3600
REDIS_URL=redis://localhost:6379
```

### Logging

```bash
LOG_LEVEL=INFO
LANGCHAIN_VERBOSE=false
```

## .env File

Create a `.env` file in the project root:

```bash
# Copy the example
cp .env.example .env

# Edit with your values
OPENAI_API_KEY=sk-...
DEFAULT_LLM_MODEL=gpt-4o
CHUNK_SIZE=500
```

## Programmatic Configuration

### Using Config Class

```python
from src.utils.config import Config

# Load configuration
config = Config()

# Get values
api_key = config.get("OPENAI_API_KEY")
chunk_size = config.get_int("CHUNK_SIZE", default=500)
enable_hybrid = config.get_bool("ENABLE_HYBRID_SEARCH", default=True)

# Get configuration groups
llm_config = config.get_llm_config()
embedding_config = config.get_embedding_config()
rag_config = config.get_rag_config()
```

### Direct Initialization

```python
from src.rag import NaiveRAG

# Initialize with specific configuration
rag = NaiveRAG(
    llm_provider="openai",
    llm_model="gpt-4o",
    llm_api_key="sk-...",
    embedding_provider="huggingface",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_provider="faiss",
    chunk_size=500,
    chunk_overlap=50,
    retrieval_k=5,
)
```

## Configuration Options

### LLM Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `llm_provider` | str | "openai" | LLM provider |
| `llm_model` | str | "gpt-4o-mini" | Model name |
| `llm_api_key` | str | None | API key |
| `temperature` | float | 0.7 | Generation temperature |
| `max_tokens` | int | None | Max tokens to generate |
| `streaming` | bool | False | Enable streaming |

### Embedding Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `embedding_provider` | str | "huggingface" | Embedding provider |
| `embedding_model` | str | "all-MiniLM-L6-v2" | Model name |
| `batch_size` | int | 32 | Batch size for embedding |

### Vector Store Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `vector_store_provider` | str | "faiss" | Vector store backend |
| `collection_name` | str | "default" | Collection name |
| `persist_directory` | str | None | Persistence directory |

### RAG Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | int | 500 | Chunk size for splitting |
| `chunk_overlap` | int | 50 | Overlap between chunks |
| `retrieval_k` | int | 5 | Documents to retrieve |
| `use_hybrid` | bool | False | Enable hybrid search |
| `use_reranking` | bool | False | Enable re-ranking |

## Provider-Specific Configuration

### OpenAI

```python
from src.core.llm import LLMManager

llm = LLMManager(
    provider="openai",
    model="gpt-4o",  # or gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
    api_key="sk-...",
    temperature=0.7,
)
```

### Anthropic

```python
llm = LLMManager(
    provider="anthropic",
    model="claude-sonnet-4-20250514",  # or claude-3-5-sonnet, claude-3-haiku
    api_key="sk-ant-...",
    temperature=0.7,
)
```

### Ox Alpha (OpenRouter)

Ox Alpha uses OpenRouter's OpenAI-compatible chat endpoint. Create a key at
<https://openrouter.ai/settings/keys> and keep it in `.env.local`:

```powershell
py scripts/connect_ox.py
```

This uses OpenRouter's OAuth PKCE flow and never prints the generated key. For
manual configuration, use:

```dotenv
OX_API_KEY=<your-openrouter-api-key>
OX_BASE_URL=https://openrouter.ai/api/v1
OX_MAX_RETRIES=5
OX_TIMEOUT_SECONDS=180
DEFAULT_LLM_PROVIDER=ox
DEFAULT_LLM_MODEL=stealth/ox-alpha
```

```python
llm = LLMManager(
    provider="ox",
    model="stealth/ox-alpha",
    temperature=0.2,
)
```

The model is currently listed as free, with a 1,048,576-token context window.
Treat that as current provider state rather than a permanent guarantee. The
stealth provider retains prompts and completions; avoid sending secrets or
sensitive documents.

### Local (Ollama)

```python
llm = LLMManager(
    provider="ollama",
    model="llama3",  # or mistral, phi3, etc.
    temperature=0.7,
)
```

### HuggingFace Embeddings

```python
from src.core.embeddings import EmbeddingsManager

embeddings = EmbeddingsManager(
    provider="huggingface",
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
    # or "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
    device="cuda",  # or "cpu", "mps"
)
```

### OpenAI Embeddings

```python
embeddings = EmbeddingsManager(
    provider="openai",
    model_name="text-embedding-3-small",  # or text-embedding-3-large
    api_key="sk-...",
)
```

## Advanced Configuration

### Custom System Prompt

```python
rag = NaiveRAG(
    system_prompt="""You are a helpful AI assistant.

Context:
{context}

Question: {question}

Answer based ONLY on the context above."""
)
```

### Custom Chunking Strategy

```python
from src.core.text_splitter import TextSplitter

splitter = TextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=True,
)
```

### Production Configuration

```python
from src.rag import AdvancedRAG

rag = AdvancedRAG(
    llm_provider="openai",
    llm_model="gpt-4o",
    embedding_provider="huggingface",
    vector_store_provider="qdrant",  # Production vector store
    chunk_size=500,
    chunk_overlap=50,
    retrieval_k=5,
    use_hybrid=True,
    use_reranking=True,
)
```

## Best Practices

1. **Never commit API keys**: Use .env files or environment variables
2. **Use appropriate models**: Balance cost and quality
3. **Tune chunk size**: 200-500 tokens for most use cases
4. **Enable caching**: Reduce API calls and latency
5. **Monitor usage**: Track API costs and performance
6. **Test configurations**: Validate settings before production

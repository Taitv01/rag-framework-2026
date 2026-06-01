<![CDATA[<div align="center">

# 🚀 Ultimate RAG Framework

### Retrieval-Augmented Generation for AI Models (2026)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-34%20Passed-brightgreen?style=for-the-badge)](#testing)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-orange?style=for-the-badge)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-red?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)

<br />

A production-ready RAG framework supporting **multiple RAG paradigms** — from basic vector search to advanced agentic systems with intelligent retrieval decisions.

[Quick Start](#-quick-start) • [Features](#-features) • [Architecture](#-architecture) • [Examples](#-examples) • [Documentation](#-documentation)

</div>

---

## 📖 Table of Contents

- [Why Ultimate RAG?](#-why-ultimate-rag)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [RAG Patterns](#-rag-patterns)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Why Ultimate RAG?

In 2026, RAG has evolved far beyond simple "vector search + LLM" patterns. Modern AI systems need:

| Challenge | Solution |
|-----------|----------|
| 🔍 **Shallow retrieval** | Hybrid search (vector + BM25) for better recall |
| 📊 **Irrelevant context** | Cross-encoder re-ranking for precision |
| 🧠 **Complex reasoning** | Agentic RAG with LangGraph for multi-step decisions |
| 💬 **Conversational AI** | Query rewriting and conversation memory |
| ⚡ **Production scale** | Caching, error handling, and monitoring |

**Ultimate RAG Framework** provides all of these in a single, modular package.

---

## ✨ Features

### 🎨 Multiple RAG Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Naive RAG** | Basic vector search + LLM | Simple Q&A, chatbots |
| **Advanced RAG** | Hybrid search + re-ranking | Technical docs, research |
| **Agentic RAG** | Agent-based with LangGraph | Complex reasoning, multi-step |
| **Graph RAG** | Knowledge graph integration | Enterprise knowledge bases |
| **Multimodal RAG** | Images, video, audio support | Medical imaging, video search |

### 🔧 Core Components

- **📄 Document Loader** — PDF, DOCX, HTML, Markdown, CSV, JSON
- **✂️ Text Splitter** — Recursive, sentence-aware, semantic chunking
- **🔢 Embeddings** — HuggingFace, OpenAI, Cohere
- **💾 Vector Store** — FAISS, ChromaDB, Qdrant
- **🔍 Retriever** — Similarity, hybrid, MMR, re-ranking
- **🤖 LLM** — OpenAI, Anthropic, Ollama (local)

### 🚀 Production Features

- ✅ Configuration management (env vars + programmatic)
- ✅ Caching (in-memory + Redis)
- ✅ Error handling with retry (tenacity)
- ✅ Logging and monitoring
- ✅ Evaluation framework (faithfulness, relevance, precision, recall)

---

## 🏗️ Architecture

### Naive RAG Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Document   │────▶│   Vector    │────▶│     LLM     │
│    Loader    │     │    Store    │     │  Generator  │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Advanced RAG Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Query     │────▶│   Hybrid    │────▶│  Re-ranker  │
│  Transform   │     │   Search    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   Vector    │     │   Document  │
                    │   Search    │     │   Grader    │
                    └─────────────┘     └─────────────┘
```

### Agentic RAG Pipeline (LangGraph)

```
START → generate_query_or_respond → [tool_calls?] → retrieve → grade_documents
            ↑                                        ↓              ↓
            ←── rewrite_question ←── [irrelevant]    [relevant]
                                                                ↓
                                                          generate_answer → END
```

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ultimate-rag.git
cd ultimate-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Docker Installation

```bash
docker build -t ultimate-rag .
docker run -it --env-file .env ultimate-rag
```

---

## 🚀 Quick Start

### 1️⃣ Demo (No API Key Required)

```bash
python examples/00_demo_no_api.py
```

This runs a complete demo using local HuggingFace embeddings — no API key needed!

### 2️⃣ Basic RAG

```python
from src.rag import NaiveRAG

# Initialize
rag = NaiveRAG(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_provider="huggingface",
)

# Add documents
rag.add_documents(["document.pdf", "article.txt"])

# Query
answer = rag.query("What is the main topic?")
print(answer)

# Query with sources
result = rag.query_with_sources("What is Python?")
print(result["answer"])
print(result["sources"])
```

### 3️⃣ Advanced RAG (Hybrid Search + Re-ranking)

```python
from src.rag import AdvancedRAG

# Initialize with advanced features
rag = AdvancedRAG(
    llm_provider="openai",
    use_hybrid=True,      # Vector + BM25
    use_reranking=True,    # Cross-encoder re-ranking
)

# Add documents
rag.add_documents(["docs/"])

# Query with detailed results
result = rag.query_detailed("What is machine learning?")
print(result["answer"])
print(result["transformed_query"])
print(result["relevant_docs"])
```

### 4️⃣ Agentic RAG (LangGraph)

```python
from src.rag import AgenticRAG

# Initialize agent-based RAG
rag = AgenticRAG(
    llm_provider="openai",
    llm_model="gpt-4o",
)

# Add documents
rag.add_documents(["knowledge_base/"])

# Query (agent decides whether to retrieve)
answer = rag.query("What is Python?")

# Multi-turn conversation
answer = rag.query(
    "Tell me more about that",
    conversation_history=[
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."}
    ]
)
```

### 5️⃣ Production RAG

```python
from src.rag import AdvancedRAG
from src.utils.config import Config
from src.utils.cache import QueryCache
from src.utils.logger import setup_logger

# Load configuration
config = Config()

# Setup logging
logger = setup_logger("production_rag")

# Initialize with config
rag = AdvancedRAG(
    llm_provider=config.get("DEFAULT_LLM_PROVIDER"),
    llm_model=config.get("DEFAULT_LLM_MODEL"),
    use_hybrid=config.get_bool("ENABLE_HYBRID_SEARCH"),
    use_reranking=config.get_bool("ENABLE_RERANKING"),
)

# Add caching
cache = QueryCache(ttl=config.get_int("CACHE_TTL"))

# Query with caching
def cached_query(question):
    cached = cache.get(question)
    if cached:
        return cached

    result = rag.query_detailed(question)
    cache.set(question, result)
    return result
```

---

## 🎨 RAG Patterns

### Naive RAG

The simplest RAG pattern. Documents are chunked, embedded, and stored in a vector database. At query time, relevant chunks are retrieved and passed to the LLM.

**Best for:** Basic Q&A, document search, simple chatbots

```python
from src.rag import NaiveRAG
rag = NaiveRAG()
rag.add_documents(["docs/"])
answer = rag.query("What is Python?")
```

### Advanced RAG

Enhanced with hybrid search (vector + BM25) and cross-encoder re-ranking for better precision.

**Best for:** Technical documentation, research papers, complex Q&A

```python
from src.rag import AdvancedRAG
rag = AdvancedRAG(use_hybrid=True, use_reranking=True)
rag.add_documents(["docs/"])
answer = rag.query("What is Python?")
```

### Agentic RAG

Uses LangGraph to create an intelligent agent that decides whether to retrieve, grades document relevance, and rewrites queries when needed.

**Best for:** Conversational AI, complex reasoning, multi-step tasks

```python
from src.rag import AgenticRAG
rag = AgenticRAG()
rag.add_documents(["docs/"])
answer = rag.query("What is Python?")
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# LLM
OPENAI_API_KEY=sk-...
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini

# Embeddings
DEFAULT_EMBEDDING_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# RAG Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=5
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
```

### Programmatic Configuration

```python
from src.utils.config import Config

config = Config()

# Get configuration
llm_config = config.get_llm_config()
rag_config = config.get_rag_config()

# Get specific values
chunk_size = config.get_int("CHUNK_SIZE", default=500)
enable_hybrid = config.get_bool("ENABLE_HYBRID_SEARCH", default=True)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_rag_pipeline.py
```

### Test Results

```
============================= test session starts ==============================
platform win32 -- Python 3.14.2
collected 35 items

tests/test_document_loader.py ........                    [ 22%]
tests/test_rag_pipeline.py ......................          [ 85%]
tests/test_retriever.py .........                         [100%]

============================== 34 passed, 1 skipped ==============================
```

---

## 📁 Project Structure

```
ultimate-rag/
├── 📄 README.md                        # This file
├── 📄 pyproject.toml                   # Project configuration
├── 📄 requirements.txt                 # Dependencies
├── 📄 .env.example                     # Environment template
├── 📄 .gitignore                       # Git ignore rules
├── 📄 LICENSE                          # MIT License
├── 📄 CONTRIBUTING.md                  # Contribution guide
│
├── 📂 src/
│   ├── 📂 core/                        # Core components
│   │   ├── document_loader.py          # Multi-format document loading
│   │   ├── text_splitter.py            # Text chunking strategies
│   │   ├── embeddings.py               # Embedding model abstraction
│   │   ├── vector_store.py             # Vector database abstraction
│   │   ├── retriever.py                # Retrieval strategies
│   │   └── llm.py                      # LLM abstraction layer
│   │
│   ├── 📂 rag/                         # RAG implementations
│   │   ├── naive_rag.py                # Basic RAG
│   │   ├── advanced_rag.py             # Advanced with re-ranking
│   │   └── agentic_rag.py              # Agent-based RAG (LangGraph)
│   │
│   ├── 📂 agents/                      # Specialized agents
│   │   ├── retrieval_agent.py          # Smart retrieval decisions
│   │   ├── grading_agent.py            # Document relevance grading
│   │   └── query_rewriter.py           # Query optimization
│   │
│   ├── 📂 evaluation/                  # Evaluation framework
│   │   ├── metrics.py                  # RAG metrics
│   │   └── evaluator.py                # End-to-end evaluation
│   │
│   └── 📂 utils/                       # Utilities
│       ├── config.py                   # Configuration management
│       ├── cache.py                    # Caching layer
│       └── logger.py                   # Logging utilities
│
├── 📂 examples/                        # Example scripts
│   ├── 00_demo_no_api.py               # Demo (no API key needed)
│   ├── 01_naive_rag.py                 # Basic RAG example
│   ├── 02_advanced_rag.py              # Advanced RAG example
│   ├── 03_agentic_rag.py               # Agentic RAG example
│   └── 04_production_rag.py            # Production RAG example
│
├── 📂 notebooks/                       # Jupyter notebooks
│   └── quickstart.ipynb                # Quick start guide
│
├── 📂 tests/                           # Unit tests
│   ├── test_document_loader.py         # Document loader tests
│   ├── test_retriever.py               # Retriever tests
│   └── test_rag_pipeline.py            # RAG pipeline tests
│
└── 📂 docs/                            # Documentation
    ├── architecture.md                 # Architecture overview
    ├── configuration.md                # Configuration guide
    └── deployment.md                   # Deployment guide
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System architecture and design patterns |
| [Configuration](docs/configuration.md) | Configuration options and best practices |
| [Deployment](docs/deployment.md) | Deployment guide for various platforms |
| [Contributing](CONTRIBUTING.md) | How to contribute to the project |

---

## 🛠️ Supported Providers

### LLM Providers

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo | ✅ |
| **Anthropic** | Claude Sonnet 4, Claude 3.5 Sonnet, Claude 3 Haiku | ✅ |
| **Ollama** | Llama 3, Mistral, Phi-3 (local) | ✅ |

### Embedding Providers

| Provider | Models | Status |
|----------|--------|--------|
| **HuggingFace** | all-MiniLM-L6-v2, all-mpnet-base-v2, BGE | ✅ |
| **OpenAI** | text-embedding-3-small, text-embedding-3-large | ✅ |
| **Cohere** | embed-english-v3.0 | ✅ |

### Vector Store Providers

| Provider | Type | Status |
|----------|------|--------|
| **FAISS** | In-memory (prototyping) | ✅ |
| **ChromaDB** | Local persistent | ✅ |
| **Qdrant** | Production scalable | ✅ |

---

## 📊 Evaluation Metrics

The framework includes built-in evaluation metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| **Faithfulness** | Is the answer grounded in context? | 0.0 - 1.0 |
| **Answer Relevance** | Does the answer address the question? | 0.0 - 1.0 |
| **Context Precision** | Are retrieved documents relevant? | 0.0 - 1.0 |
| **Context Recall** | Are all relevant documents retrieved? | 0.0 - 1.0 |

```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator(llm)
report = evaluator.evaluate(query_func, test_data)
evaluator.print_report(report)
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) — LLM application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent orchestration
- [HuggingFace](https://huggingface.co) — Transformers and embeddings
- [FAISS](https://faiss.ai) — Vector similarity search
- [ChromaDB](https://www.trychroma.com) — Vector database
- [Qdrant](https://qdrant.tech) — Vector search engine

---

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/ultimate-rag/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/ultimate-rag/discussions)
- 📧 **Email**: your.email@example.com

---

## ⭐ Star History

If you find this project useful, please consider giving it a star on GitHub!

---

<div align="center">

**Made with ❤️ by the RAG Framework Team**

[⬆ Back to Top](#-ultimate-rag-framework)

</div>
]]>
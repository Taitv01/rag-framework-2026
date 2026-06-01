# 🚀 Ultimate RAG Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://langchain.com)

> **A comprehensive Retrieval-Augmented Generation (RAG) framework supporting multiple RAG paradigms for AI models (2026)**

## 📖 Overview

The Ultimate RAG Framework provides production-ready implementations of multiple RAG patterns, from basic vector search to advanced agentic systems. Built with the latest 2026 best practices, it supports:

- **Naive RAG** - Basic vector search + LLM generation
- **Advanced RAG** - Hybrid search with re-ranking
- **Agentic RAG** - Agent-based retrieval with LangGraph
- **Graph RAG** - Knowledge graph integration
- **Multimodal RAG** - Multi-format document support

## ✨ Features

### 🎯 Multiple RAG Patterns

| Pattern | Use Case | Complexity |
|---------|----------|------------|
| Naive RAG | Basic Q&A, chatbots | ⭐ |
| Advanced RAG | Technical docs, research | ⭐⭐ |
| Agentic RAG | Conversational AI, complex reasoning | ⭐⭐⭐ |
| Graph RAG | Enterprise knowledge, multi-hop questions | ⭐⭐⭐ |
| Multimodal RAG | Images, video, audio | ⭐⭐⭐ |

### 🔧 Core Components

- **Document Loader** - PDF, DOCX, HTML, Markdown, CSV, JSON
- **Text Splitter** - Recursive, sentence-aware, semantic chunking
- **Embeddings** - HuggingFace, OpenAI, Cohere support
- **Vector Store** - FAISS, ChromaDB, Qdrant backends
- **Retriever** - Similarity, hybrid, MMR, re-ranking
- **LLM** - OpenAI, Anthropic, Ollama support

### 🚀 Production Features

- ✅ Configuration management
- ✅ Caching (in-memory, Redis)
- ✅ Error handling with retry
- ✅ Logging and monitoring
- ✅ Evaluation framework

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip or poetry

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/ultimate-rag.git
cd ultimate-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### Docker Installation

```bash
# Build and run with Docker
docker build -t ultimate-rag .
docker run -it --env-file .env ultimate-rag

# Or use Docker Compose
docker-compose up -d
```

## 🚀 Quick Start

### 1. Naive RAG (Basic)

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
```

### 2. Advanced RAG (With Re-ranking)

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

### 3. Agentic RAG (LangGraph)

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

### 4. Production RAG

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

## 📚 Examples

### Basic Examples

```bash
# Naive RAG
python examples/01_naive_rag.py

# Advanced RAG
python examples/02_advanced_rag.py

# Agentic RAG
python examples/03_agentic_rag.py

# Production RAG
python examples/04_production_rag.py
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/
# - quickstart.ipynb
# - advanced_techniques.ipynb
# - evaluation_guide.ipynb
```

## 🏗️ Architecture

### RAG Pipeline Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Document   │────▶│   Text      │────▶│   Embeddings│
│   Loader     │     │   Splitter  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│     LLM      │◀────│   Retriever │◀────│ Vector Store│
│   Generator  │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Agentic RAG Flow

```
START → generate_query_or_respond → [tool_calls?] → retrieve → grade_documents
            ↑                                        ↓              ↓
            ←── rewrite_question ←── [irrelevant]    [relevant]
                                                                ↓
                                                          generate_answer → END
```

## 🔧 Configuration

### Environment Variables

```bash
# LLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini

# Embeddings
DEFAULT_EMBEDDING_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store
DEFAULT_VECTOR_STORE=faiss

# RAG
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
```

## 📊 Evaluation

### Built-in Metrics

- **Faithfulness** - Is the answer grounded in context?
- **Answer Relevance** - Does the answer address the question?
- **Context Precision** - Are retrieved documents relevant?
- **Context Recall** - Are all relevant documents retrieved?

### Evaluation Example

```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator(llm)

# Evaluate
report = evaluator.evaluate(
    query_func=rag.query,
    test_data=[
        {"question": "What is Python?", "expected": "Python is..."},
    ]
)

# Print report
evaluator.print_report(report)

# Export report
evaluator.export_report(report, "evaluation.json")
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_rag_pipeline.py
```

## 📁 Project Structure

```
ultimate-rag/
├── src/
│   ├── core/                    # Core components
│   │   ├── document_loader.py   # Multi-format document loading
│   │   ├── text_splitter.py     # Text chunking strategies
│   │   ├── embeddings.py        # Embedding model abstraction
│   │   ├── vector_store.py      # Vector database abstraction
│   │   ├── retriever.py         # Retrieval strategies
│   │   └── llm.py               # LLM abstraction layer
│   │
│   ├── rag/                     # RAG implementations
│   │   ├── naive_rag.py         # Basic RAG
│   │   ├── advanced_rag.py      # Advanced with re-ranking
│   │   └── agentic_rag.py       # Agent-based RAG
│   │
│   ├── agents/                  # Specialized agents
│   │   ├── retrieval_agent.py   # Smart retrieval decisions
│   │   ├── grading_agent.py     # Document relevance grading
│   │   └── query_rewriter.py    # Query optimization
│   │
│   ├── evaluation/              # Evaluation framework
│   │   ├── metrics.py           # RAG metrics
│   │   └── evaluator.py         # End-to-end evaluation
│   │
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration management
│       ├── cache.py             # Caching layer
│       └── logger.py            # Logging utilities
│
├── examples/                    # Example scripts
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/ultimate-rag.git
cd ultimate-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
black src/ tests/
isort src/ tests/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) - LLM framework
- [LlamaIndex](https://llamaindex.ai) - Data framework
- [HuggingFace](https://huggingface.co) - Transformers and embeddings
- [FAISS](https://faiss.ai) - Vector search
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Qdrant](https://qdrant.tech) - Vector search engine

## 📞 Support

- 📧 Email: your.email@example.com
- 💬 Discord: [Join our community](https://discord.gg/your-server)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/ultimate-rag/issues)
- 📖 Docs: [Documentation](https://github.com/yourusername/ultimate-rag#readme)

## 🗺️ Roadmap

- [ ] Graph RAG implementation
- [ ] Multimodal RAG support
- [ ] More vector store backends
- [ ] Streaming support
- [ ] REST API
- [ ] Web UI
- [ ] More evaluation metrics
- [ ] Performance benchmarks

---

<p align="center">
  Made with ❤️ by the RAG Framework Team
</p>

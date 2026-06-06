<div align="center">

# 🚀 Ultimate RAG Framework

### Retrieval-Augmented Generation for AI Models (2026)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-116%20Passed-brightgreen?style=for-the-badge)](#-testing)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.25%2B-FF6B6B?style=for-the-badge)](https://gradio.app)

<br />

A production-ready RAG framework supporting **multiple RAG paradigms** — from basic vector search to advanced agentic systems with intelligent retrieval decisions. Built-in Vietnamese language support and story writing system.

**🎯 Perfect for:** Chatbots • Knowledge Bases • Story Writing • Document Q&A • Research Assistants

[Quick Start](#-quick-start) • [Features](#-features) • [Installation](#-installation) • [Examples](#-usage-examples) • [API](#-api-reference) • [Docs](#-documentation)

</div>

---

## 📖 Table of Contents

- [Why Ultimate RAG?](#-why-ultimate-rag)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Story Writing System](#-story-writing-system)
- [API Reference](#-api-reference)
- [Web UI](#-web-ui)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Why Ultimate RAG?

| Problem | Solution |
|---------|----------|
| 🔍 **Shallow retrieval** | Hybrid search (vector + BM25) for better recall |
| 📊 **Irrelevant context** | Cross-encoder re-ranking for precision |
| 🧠 **Complex reasoning** | Agentic RAG with LangGraph for multi-step decisions |
| 📚 **Long documents** | Advanced chunking (semantic, proposition, contextual) |
| 💬 **Multi-turn chat** | Conversation memory with buffer/window/summary modes |
| 🇻🇳 **Vietnamese language** | Vietnamese NLP processor, optimized embeddings, bilingual prompts |
| 🔐 **Production security** | API key management, rate limiting, JWT authentication |
| 📈 **Monitoring** | Usage tracking, analytics, error monitoring |
| 📖 **Story writing** | Character, plot, world management with consistency checking |
| 🤖 **Hallucination detection** | Claim-level grounding verification for answer safety |
| 🌐 **Web search fallback** | Automatic web search when retrieval quality is poor |
| 📊 **Smart metadata** | Auto-assign characters, locations, time periods to chunks |

---

## ✨ Features

### 🎨 RAG Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Naive RAG** | Basic vector search + LLM | Simple Q&A, chatbots |
| **Advanced RAG** | Hybrid search + re-ranking + cache + HyDE | Technical docs, research |
| **Adaptive RAG** | Query router + auto-select pipeline | Cost-optimized production |
| **Agentic RAG** | Agent-based with LangGraph | Complex reasoning |
| **Graph RAG** | Knowledge graph (NetworkX + Neo4j) | Relationship queries |
| **Self-RAG** | Self-reflective with quality scoring | Quality-critical apps |
| **Corrective RAG** | Dynamic retrieval correction | Fact-checking |
| **HyDE** | Hypothetical Document Embedding | Better semantic matching |
| **RAPTOR** | Tree-organized recursive retrieval | Cross-document reasoning |

### 🇻🇳 Vietnamese Language Support

- **Vietnamese Processor** — Word segmentation (`underthesea`), sentence splitting, language detection, NER for fairy tales
- **Vietnamese Embeddings** — Default: `keepitreal/vietnamese-sbert` (768d)
- **Vietnamese Reranker** — `AITeamVN/Vietnamese_Reranker` (MRR@10: 86.72)
- **BM25 Vietnamese Tokenization** — Proper word segmentation for hybrid search
- **Bilingual Prompts** — All LLM prompts support both Vietnamese and English

### 🤖 Phase 3: Advanced Features

- **Metadata Enhancement** — LLM auto-extracts characters/locations/time/topic/sentiment per chunk
- **Hallucination Grader** — Claim-level verification that answers are grounded in documents
- **Neo4j Integration** — Persistent graph storage alongside NetworkX (supplement, not replace)
- **Web Search Fallback** — Automatic web search when retrieval fails (OFF by default, safety-first)

### 🔧 Core Components

- **📄 Document Loader** — PDF, DOCX, HTML, Markdown, CSV, JSON
- **✂️ Text Splitter** — Recursive, semantic, proposition, contextual headers, RAPTOR, late chunking
- **📊 Advanced Chunking** — Semantic, Proposition, Contextual Retrieval (Anthropic pattern), Parent-Child
- **💾 Semantic Cache** — Embedding similarity-based caching (reduces 50-70% LLM calls)
- **🔢 Embeddings** — HuggingFace, OpenAI, Cohere (6 Vietnamese models available)
- **💾 Vector Store** — FAISS, ChromaDB, Qdrant
- **🔍 Retriever** — Similarity, hybrid, MMR, re-ranking, multi-query RRF, HyDE, parent-child
- **🤖 LLM** — OpenAI, Anthropic, Ollama (local)

### 🚀 Production Features

- ✅ **RESTful API** — FastAPI endpoints with Swagger UI
- ✅ **Web UI** — Gradio interface with chat, upload, search, stats
- ✅ **Streaming** — Token-by-token response streaming
- ✅ **Memory** — Conversation history (buffer, window, summary modes)
- ✅ **Auth** — API key management, rate limiting, JWT
- ✅ **Monitoring** — Usage tracking, analytics, error monitoring
- ✅ **Document Management** — CRUD operations, versioning, bulk operations

---

## 📦 Installation

### Prerequisites

- Python 3.10+ (recommended: 3.11 or 3.12)
- pip

### Quick Install

```bash
# 1. Clone repository
git clone https://github.com/Taitv01/rag-framework-2026.git
cd rag-framework-2026

# 2. Create virtual environment
python -m venv venv

# 3. Activate venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Modular Install (Optional)

```bash
# Full install (includes web search + Neo4j)
pip install -e ".[all]"

# Web search only (DuckDuckGo, Tavily)
pip install -e ".[web]"

# Neo4j graph storage only
pip install -e ".[graph]"

# Dev tools (pytest, black, ruff)
pip install -e ".[dev]"
```

### Manual Package Install

```bash
# Core (required)
pip install langchain langchain-core langchain-community langchain-text-splitters
pip install faiss-cpu sentence-transformers
pip install pypdf python-docx beautifulsoup4 markdown rank-bm25
pip install fastapi uvicorn python-multipart
pip install pydantic python-dotenv rich tiktoken

# Vietnamese NLP (recommended)
pip install underthesea pyvi

# Web Search (optional)
pip install duckduckgo-search
pip install tavily-python

# Neo4j (optional)
pip install neo4j
```

### Environment Configuration

Edit the `.env` file:

```bash
# ===== LLM API Keys =====
OPENAI_API_KEY=sk-...
# Or use Anthropic:
# ANTHROPIC_API_KEY=sk-ant-...

# ===== Web Search (optional) =====
# Tavily (recommended for RAG): https://tavily.com
# TAVILY_API_KEY=tvly-...
# DuckDuckGo: free, no API key needed

# ===== Neo4j (optional) =====
# NEO4J_URI=bolt://localhost:7687
# NEO4J_PASSWORD=your_password

# ===== Default Settings =====
DEFAULT_EMBEDDING_MODEL=keepitreal/vietnamese-sbert
DEFAULT_RERANKER_MODEL=AITeamVN/Vietnamese_Reranker
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=5
```

---

## 🚀 Quick Start

### 1️⃣ Demo (No API Key Needed)

```bash
python examples/00_demo_no_api.py
```

Runs a complete demo using local HuggingFace embeddings — no API key required!

### 2️⃣ Basic RAG (3 Lines of Code)

```python
from src.rag import NaiveRAG

rag = NaiveRAG()
rag.add_documents(["document.pdf", "article.txt"])
answer = rag.query("What is the main topic?")
print(answer)
```

### 3️⃣ Advanced RAG (Optimized for Vietnamese)

```python
from src.rag import AdvancedRAG

rag = AdvancedRAG(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_model="keepitreal/vietnamese-sbert",  # Vietnamese embeddings
    use_hybrid=True,          # Vector + BM25
    use_reranking=True,       # Cross-encoder re-ranking
    use_cache=True,           # Semantic cache
    use_multi_query_rrf=True, # Multi-query + RRF fusion
)

rag.add_documents(["docs/"])
answer = rag.query("What is Thạch Sanh?")
print(answer)
```

### 4️⃣ Advanced RAG with All Phase 3 Features

```python
from src.rag import AdvancedRAG

rag = AdvancedRAG(
    llm_provider="openai",
    use_hybrid=True,
    use_reranking=True,
    # Phase 2 features
    use_cache=True,
    use_contextual_chunking=True,  # Anthropic contextual retrieval
    use_hyde=True,                 # Hypothetical Document Embedding
    use_multi_query_rrf=True,      # Multi-query RRF fusion
    # Phase 3 features
    use_metadata_enhancement=True, # Auto-extract characters/locations/time
    use_hallucination_check=True,  # Verify answer grounding
    use_web_search=True,           # Web search fallback (OFF by default)
    web_search_provider="duckduckgo",
)

rag.add_documents(["docs/"])
answer = rag.query("How did Thạch Sanh defeat the eagle?")
```

### 5️⃣ Graph RAG with Neo4j

```python
from src.rag import GraphRAG

# Without Neo4j (in-memory NetworkX)
rag = GraphRAG()
rag.add_documents(["docs/"])
answer = rag.query("What is the relationship between Thạch Sanh and Lý Thông?")

# With Neo4j (persistent storage)
rag = GraphRAG(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="your_password",
)
rag.add_documents(["docs/"])
answer = rag.query("Who is Thạch Sanh's enemy?")
```

### 6️⃣ REST API Server

```bash
python examples/06_api_server.py
# Visit: http://localhost:8000/docs
```

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "What is Thạch Sanh?",
    "k": 5
})
print(response.json()["answer"])
```

### 7️⃣ Web UI

```bash
python examples/07_web_ui.py
# Visit: http://localhost:7860
```

---

## 📚 Usage Examples

### Streaming Response

```python
from src.rag import AdvancedRAG
from src.core.memory import ConversationalRAG

rag = AdvancedRAG(use_hybrid=True)
rag.add_documents(["docs/"])

conv_rag = ConversationalRAG(rag, memory_type="buffer")

# Stream response token by token
for token in conv_rag.stream("Tell me about Thạch Sanh"):
    print(token, end="", flush=True)
```

### Conversation Memory

```python
from src.core.memory import ConversationalRAG

conv_rag = ConversationalRAG(rag, memory_type="window", max_history=10)

# Multi-turn conversation with context
answer1 = conv_rag.query("Who is Thạch Sanh?")
answer2 = conv_rag.query("What did he do?")          # Remembers context
answer3 = conv_rag.query("Compare with Thánh Gióng")  # Still remembers
```

### Metadata Enhancement

```python
from src.core.metadata_enhancer import MetadataEnhancer

enhancer = MetadataEnhancer(llm=rag.llm)
enhanced_chunks = enhancer.enhance(chunks)

# Each chunk now has structured metadata:
# chunk.metadata["characters"] = ["Thạch Sanh", "Lý Thông"]
# chunk.metadata["locations"] = ["eagle's cave"]
# chunk.metadata["time_period"] = "ancient"
# chunk.metadata["topic"] = "Thạch Sanh defeats the eagle to rescue the princess"
# chunk.metadata["sentiment"] = "heroic"
```

### Hallucination Grading

```python
from src.agents.hallucination_grader import HallucinationGrader

grader = HallucinationGrader(llm=rag.llm)

# Verify an answer against source documents
grade = grader.grade(
    answer="Thạch Sanh killed the eagle with a bow and arrow.",
    context="Thạch Sanh used his magic bow to defeat the eagle."
)

print(grade.is_grounded)        # False — "bow and arrow" vs "magic bow"
print(grade.grounded_score)     # 0.5
print(grade.unsupported_claims) # ["killed with a bow and arrow"]

# Safe generation with automatic verification
answer, grade = grader.safe_generate(question, context, max_retries=2)
```

### Web Search (Safety-First)

```python
from src.core.web_search import SafeWebSearcher, DuckDuckGoSearchProvider

provider = DuckDuckGoSearchProvider()
searcher = SafeWebSearcher(provider=provider, llm=rag.llm)

# Search + verify relevance with LLM
results = searcher.search("Thạch Sanh fairy tale", num_results=3)
docs = searcher.to_documents(results)

# All web results are clearly labeled:
# docs[0].metadata["source_type"] = "web"
# docs[0].metadata["url"] = "https://en.wikipedia.org/wiki/Thạch_Sanh"
```

### Advanced Chunking

```python
from src.core.advanced_chunking import (
    SemanticChunker,
    PropositionChunker,
    ContextualRetrievalChunker,
    ParentChildChunker,
)

# Semantic chunking — split by meaning
chunker = SemanticChunker(embeddings, threshold=0.8)
chunks = chunker.split(documents)

# Contextual Retrieval (Anthropic pattern) — prepend context to each chunk
chunker = ContextualRetrievalChunker(llm, chunk_size=500)
chunks = chunker.split(documents)
# Each chunk now starts with: "[Context: This chunk discusses...]"

# Parent-Child — search with small chunks, return large context
parent_chunks, child_chunks = ParentChildChunker(
    parent_size=2000, child_size=200
).split(documents)
```

### Retrieval Strategies

```python
from src.core.retriever import RetrieverManager

retriever = RetrieverManager(
    vector_store=store,
    embeddings=embeddings,
    documents=chunks,
    use_hybrid=True,
    use_reranking=True,
)

# Basic search
results = retriever.search("Thạch Sanh", k=5)

# Hybrid search (vector + BM25)
results = retriever.hybrid_search("Thạch Sanh fights eagle", k=5)

# Re-ranking with cross-encoder
results = retriever.search_with_reranking("Thạch Sanh story", k=5)

# Multi-query + RRF (Reciprocal Rank Fusion)
results = retriever.multi_query_rrf_search("Who is Thạch Sanh?", k=5, llm=llm)

# HyDE — search using hypothetical answer
results = retriever.hyde_search("Who is Thạch Sanh?", k=5, llm=llm)

# Parent-Child search
results = retriever.parent_child_search("Thạch Sanh", k=5, parent_chunks=parents)
```

---

## 📖 Story Writing System

```python
from src.story import (
    CharacterManager, Character,
    PlotManager, PlotPoint, PlotArc,
    WorldBuilder, Location,
    ConsistencyChecker,
    ChapterManager, TimelineManager,
)

# Create characters
char_mgr = CharacterManager()
char_mgr.add_character(Character(
    name="Thạch Sanh",
    personality="Honest, brave, filial",
    backstory="Orphan living under a tree",
    strengths=["Superhuman strength", "Archery skills"],
    weaknesses=["Too trusting"],
))

# Manage plot
plot_mgr = PlotManager()
plot_mgr.add_plot_point(PlotPoint(
    chapter=1,
    event="Thạch Sanh meets Lý Thông",
    importance="high",
    characters_involved=["Thạch Sanh", "Lý Thông"],
))

# Build world
world = WorldBuilder()
world.add_location(Location(
    name="Eagle's Cave",
    description="Deep dark cave where the princess is imprisoned",
    first_appearance=3,
))

# Check consistency
checker = ConsistencyChecker(llm)
issues = checker.check_chapter(chapter_text, chapter_number)
for issue in issues:
    print(f"[{issue.severity}] {issue.description}")
```

### Adaptive RAG (NEW — 2026)

```python
from src.rag import AdaptiveRAG

# Auto-selects the best pipeline based on query complexity
rag = AdaptiveRAG(
    llm_provider="openai",
    embedding_provider="huggingface",
)
rag.add_documents(["docs/"])

# Simple query → NaiveRAG (fast, cheap)
answer = rag.query("Thạch Sanh là ai?")

# Complex query → AgenticRAG (thorough)
answer = rag.query("So sánh nhân vật Thạch Sanh và Lý Thông qua các khía cạnh tính cách, hành động và kết cục")
```

---

## 🌐 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Query the RAG system |
| `POST` | `/documents` | Add text documents |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/ingest` | Upload and index files |
| `POST` | `/search` | Search documents without generation |

### Examples

```python
import requests

# Query
response = requests.post("http://localhost:8000/query", json={
    "question": "What is Thạch Sanh?",
    "k": 5,
    "transform_query": True,
    "grade_documents": True,
})
print(response.json()["answer"])

# Add documents
response = requests.post("http://localhost:8000/documents", json={
    "texts": ["Thạch Sanh is a character in Vietnamese fairy tales..."],
})
print(response.json()["status"])

# Upload files
files = [("files", open("document.pdf", "rb"))]
response = requests.post("http://localhost:8000/ingest", files=files)
```

Swagger UI: `http://localhost:8000/docs`

---

## 🖥️ Web UI

```bash
python examples/07_web_ui.py
# Visit: http://localhost:7860
```

Features:
- 💬 **Chat Interface** — Chat with your knowledge base
- 📄 **Document Upload** — Upload and index documents
- 🔍 **Search** — Search through indexed documents
- 📊 **Statistics** — View usage analytics

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `DEFAULT_EMBEDDING_MODEL` | `keepitreal/vietnamese-sbert` | Default embedding model |
| `DEFAULT_RERANKER_MODEL` | `AITeamVN/Vietnamese_Reranker` | Default reranker model |
| `CHUNK_SIZE` | `500` | Chunk size for text splitting |
| `CHUNK_OVERLAP` | `50` | Chunk overlap |
| `RETRIEVAL_K` | `5` | Number of documents to retrieve |
| `TAVILY_API_KEY` | — | Tavily web search API key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `LOG_LEVEL` | `INFO` | Logging level |

### Programmatic Configuration

```python
from src.utils.config import Config

config = Config()
llm_config = config.get_llm_config()
rag_config = config.get_rag_config()
```

---

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=src

# Run specific test module
pytest tests/test_phase3_features.py -v

# Run specific test
pytest tests/test_retriever.py -v
```

### Results

```
============================= 116 passed in 20.82s ==============================

tests/test_document_loader.py    :  8 tests  ✅
tests/test_new_features.py       : 39 tests  ✅
tests/test_phase3_features.py    : 33 tests  ✅
tests/test_rag_pipeline.py       : 20 tests  ✅
tests/test_retriever.py          : 16 tests  ✅
```

---

## 📁 Project Structure

```
rag-framework-2026/
├── src/
│   ├── core/                        # Core components
│   │   ├── document_loader.py       # Load documents (PDF, DOCX, HTML...)
│   │   ├── text_splitter.py         # Text splitting strategies
│   │   ├── embeddings.py            # Embedding model management
│   │   ├── vector_store.py          # Vector store management
│   │   ├── retriever.py             # Retrieval (hybrid, MMR, RRF, HyDE)
│   │   ├── llm.py                   # LLM provider management
│   │   ├── streaming.py             # Token streaming
│   │   ├── memory.py                # Conversation memory
│   │   ├── advanced_chunking.py     # Semantic, proposition, contextual, RAPTOR, late chunking
│   │   ├── vietnamese_processor.py  # Vietnamese NLP processing
│   │   ├── metadata_enhancer.py     # [Phase 3] Metadata enhancement
│   │   ├── graph_store.py           # [Phase 3] Neo4j backend
│   │   └── web_search.py            # [Phase 3] Web search fallback
│   │
│   ├── rag/                         # RAG implementations
│   │   ├── naive_rag.py             # Basic RAG
│   │   ├── advanced_rag.py          # Advanced RAG (hybrid, reranking, cache)
│   │   ├── adaptive_rag.py          # NEW: Query router + auto-select pipeline
│   │   ├── agentic_rag.py           # Agentic RAG (LangGraph)
│   │   ├── graph_rag.py             # Graph RAG (Knowledge Graph + Neo4j)
│   │   └── advanced_techniques.py   # Self-RAG, CRAG, HyDE, HyPE
│   │
│   ├── agents/                      # Specialized agents
│   │   ├── retrieval_agent.py       # Retrieval decision agent
│   │   ├── grading_agent.py         # Document grading agent
│   │   ├── query_rewriter.py        # Query optimization agent
│   │   └── hallucination_grader.py  # [Phase 3] Hallucination detection
│   │
│   ├── story/                       # Story writing system
│   │   ├── character_manager.py     # Character management
│   │   ├── plot_manager.py          # Plot management
│   │   ├── world_builder.py         # World building
│   │   ├── consistency_checker.py   # Consistency checking
│   │   ├── chapter_manager.py       # Chapter management
│   │   ├── timeline_manager.py      # Timeline management
│   │   └── writing_assistant.py     # AI writing assistant
│   │
│   ├── api/                         # REST API
│   │   └── app.py                   # FastAPI endpoints
│   │
│   ├── ui/                          # Web UI
│   │   └── __init__.py              # Gradio interface
│   │
│   ├── auth/                        # Authentication
│   │   ├── auth_manager.py          # JWT authentication
│   │   ├── api_key_manager.py       # API key management
│   │   └── rate_limiter.py          # Rate limiting
│   │
│   ├── monitoring/                  # Monitoring
│   │   └── metrics_collector.py     # Metrics collection
│   │
│   ├── documents/                   # Document management
│   │   └── manager.py               # CRUD operations
│   │
│   ├── evaluation/                  # Evaluation
│   │   ├── metrics.py               # RAGAS metrics
│   │   └── evaluator.py             # Evaluation pipeline
│   │
│   └── utils/                       # Utilities
│       ├── config.py                # Configuration
│       ├── cache.py                 # Cache (exact + semantic)
│       └── logger.py                # Logging
│
├── examples/                        # Example scripts (00-08)
├── tests/                           # Unit tests (116 tests)
├── notebooks/                       # Jupyter notebooks
├── docs/                            # Documentation
├── PHASES.md                        # Development roadmap
├── pyproject.toml                   # Project config
├── requirements.txt                 # Dependencies
└── README.md                        # ← You are here
```

---

## 🤝 Contributing

Contributions are welcome! See [Contributing Guide](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) — LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent orchestration
- [HuggingFace](https://huggingface.co) — Transformers & embeddings
- [FAISS](https://faiss.ai) — Vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework
- [Gradio](https://gradio.app) — Web UI framework
- [Anthropic](https://www.anthropic.com) — Contextual Retrieval technique

---

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Taitv01/rag-framework-2026/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/Taitv01/rag-framework-2026/discussions)

---

<div align="center">

**Made with ❤️ for Story Writers and AI Enthusiasts**

[⬆ Back to Top](#-ultimate-rag-framework)

</div>

<![CDATA[<div align="center">

# 🚀 Ultimate RAG Framework

### Retrieval-Augmented Generation for AI Models (2026)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-58%20Passed-brightgreen?style=for-the-badge)](#testing)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.25%2B-FF6B6B?style=for-the-badge)](https://gradio.app)

<br />

A production-ready RAG framework supporting **multiple RAG paradigms** — from basic vector search to advanced agentic systems with intelligent retrieval decisions.

**🎯 Perfect for:** Chatbots • Knowledge Bases • Story Writing • Document Q&A • Research Assistants

[Quick Start](#-quick-start) • [Features](#-features) • [Story Writing](#-story-writing) • [API](#-api) • [Documentation](#-documentation)

</div>

---

## 📖 Table of Contents

- [Why Ultimate RAG?](#-why-ultimate-rag)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Story Writing Guide](#-story-writing-guide)
- [RAG Patterns](#-rag-patterns)
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
| 🔐 **Production security** | API key management, rate limiting |
| 📈 **Monitoring** | Usage tracking, analytics, error monitoring |
| 📖 **Story Writing** | Character, plot, world management with consistency checking |

---

## ✨ Features

### 🎨 RAG Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Naive RAG** | Basic vector search + LLM | Simple Q&A, chatbots |
| **Advanced RAG** | Hybrid search + re-ranking | Technical docs, research |
| **Agentic RAG** | Agent-based with LangGraph | Complex reasoning |
| **Graph RAG** | Knowledge graph integration | Relationship queries |
| **Self-RAG** | Self-reflective RAG | Quality-critical apps |
| **Corrective RAG** | Dynamic correction | Fact-checking |
| **HyDE** | Hypothetical Document Embedding | Better semantic matching |

### 📖 Story Writing System

| Feature | Description |
|---------|-------------|
| **CharacterManager** | Character profiles, relationships, development arcs |
| **PlotManager** | Plot points, arcs, foreshadowing, subplots |
| **WorldBuilder** | Locations, lore, world rules, cultural elements |
| **ConsistencyChecker** | Character, plot, timeline, fact checking |
| **ChapterManager** | Chapter organization, word count |
| **TimelineManager** | Event timeline tracking |
| **WritingAssistant** | AI-powered content generation |

### 🔧 Core Components

- **📄 Document Loader** — PDF, DOCX, HTML, Markdown, CSV, JSON
- **✂️ Text Splitter** — Recursive, semantic, proposition, contextual headers
- **🔢 Embeddings** — HuggingFace, OpenAI, Cohere
- **💾 Vector Store** — FAISS, ChromaDB, Qdrant
- **🔍 Retriever** — Similarity, hybrid, MMR, re-ranking
- **🤖 LLM** — OpenAI, Anthropic, Ollama (local)

### 🚀 Production Features

- ✅ **RESTful API** — FastAPI endpoints
- ✅ **Web UI** — Gradio interface
- ✅ **Streaming** — Token-by-token response
- ✅ **Memory** — Conversation history management
- ✅ **Auth** — API key management, rate limiting
- ✅ **Monitoring** — Usage tracking, analytics
- ✅ **Document Management** — CRUD operations, versioning

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip

### Quick Install

```bash
# Clone repository
git clone https://github.com/Taitv01/rag-framework-2026.git
cd rag-framework-2026

# Create virtual environment
python -m vvenv venv

# Activate
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Docker Install

```bash
docker build -t ultimate-rag .
docker run -it --env-file .env ultimate-rag
```

---

## 🚀 Quick Start

### 1️⃣ Demo (No API Key Needed)

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

### 4️⃣ Graph RAG (Knowledge Graph)

```python
from src.rag import GraphRAG

# Initialize
rag = GraphRAG(llm_provider="openai")

# Add documents
rag.add_documents(["docs/"])

# Query with graph reasoning
answer = rag.query("What is the relationship between X and Y?")

# Get knowledge graph
kg = rag.get_knowledge_graph()
neighbors = kg.get_neighbors("Python")
```

### 5️⃣ REST API Server

```bash
# Start API server
python examples/06_api_server.py

# Visit API docs
open http://localhost:8000/docs
```

```python
import requests

# Query via API
response = requests.post("http://localhost:8000/query", json={
    "question": "What is Python?",
    "k": 5
})
print(response.json()["answer"])
```

### 6️⃣ Web UI

```bash
# Start Web UI
python examples/07_web_ui.py

# Visit UI
open http://localhost:7860
```

Features:
- 💬 Chat interface
- 📄 Document upload
- 🔍 Search interface
- 📊 Statistics dashboard

---

## 📖 Story Writing Guide

### Complete Workflow

```python
from src.story import (
    CharacterManager, Character,
    PlotManager, PlotPoint, PlotArc,
    WorldBuilder, Location, Lore,
    ConsistencyChecker,
    WritingAssistant
)
from src.rag import AdvancedRAG

# =========================================================================
# Step 1: Setup Knowledge Base
# =========================================================================

# Character Manager
char_manager = CharacterManager()

char_manager.add_character(Character(
    name="Nguyễn Văn A",
    age=25,
    gender="Nam",
    personality="Thông minh, quyết đoán, đôi khi nóng nảy",
    backstory="Mồ côi từ nhỏ, lớn lên ở trại trẻ mồ côi",
    appearance="Cao 1m75, tóc đen, mắt nâu",
    motivations=["Tìm lại gia đình"],
    fears=["Bị bỏ rơi"],
    strengths=["Trí nhớ tốt", "Can đảm"],
    weaknesses=["Nóng nảy"],
))

char_manager.add_character(Character(
    name="Trần Thị B",
    age=23,
    gender="Nữ",
    personality="Hiền lành, thông minh",
    backstory="Con gái gia đình giàu có",
))

# Add relationship
char_manager.add_relationship(
    "Nguyễn Văn A", "Trần Thị B",
    "friend", "Bạn thân từ nhỏ"
)

# Plot Manager
plot_manager = PlotManager()

plot_manager.add_plot_point(PlotPoint(
    chapter=1,
    event="A tìm thấy bức thư cũ",
    importance="high",
    characters_involved=["Nguyễn Văn A"],
))

plot_manager.create_plot_arc(PlotArc(
    name="Hành trình tìm lại gia đình",
    description="A khám phá bí mật về gia đình",
    start_chapter=1,
    characters_involved=["Nguyễn Văn A", "Trần Thị B"],
))

plot_manager.add_foreshadowing(
    chapter=1,
    hint="Bức thư có mùi lạ",
    resolution_chapter=10,
)

# World Builder
world_builder = WorldBuilder()

world_builder.add_location(Location(
    name="Hà Nội",
    description="Thủ đô ngàn năm văn hiến",
    climate="Nhiệt đới gió mùa",
    landmarks=["Hồ Gươm", "Phố Cổ"],
    first_appearance=1,
))

world_builder.add_location(Location(
    name="Đà Lạt",
    description="Thành phố mù sương",
    climate="Mát mẻ quanh năm",
    first_appearance=5,
))

# =========================================================================
# Step 2: Setup RAG with Story Context
# =========================================================================

rag = AdvancedRAG(
    llm_provider="openai",
    llm_model="gpt-4o",
    embedding_provider="huggingface",
    use_hybrid=True,
    use_reranking=True,
)

# Add story documents
rag.add_documents([
    "truyen/chuong_1.txt",
    "truyen/chuong_2.txt",
    # ... more chapters
])

# =========================================================================
# Step 3: Write New Chapter
# =========================================================================

def write_chapter(chapter_number, main_characters, location, plot_points):
    """Write new chapter with consistency checking."""

    # Get character context
    char_context = ""
    for name in main_characters:
        char_context += char_manager.get_character_context(name) + "\n\n"

    # Get plot context
    plot_context = plot_manager.get_plot_context()

    # Get world context
    world_context = world_builder.get_location_context(location)

    # Generate chapter
    prompt = f"""Viết chương {chapter_number} với thông tin sau:

Nhân vật:
{char_context}

Cốt truyện:
{plot_context}

Địa điểm:
{world_context}

Plot points cần đề cập:
{chr(10).join([f"- {p}" for p in plot_points])}

Yêu cầu:
1. Nhất quán với tính cách nhân vật
2. Mô tả địa điểm sinh động
3. Phát triển plot points
4. Hội thoại tự nhiên

Viết chương hoàn chỉnh:"""

    chapter_content = rag.llm.generate(prompt)

    # Update character development
    for name in main_characters:
        char_manager.add_development(
            name,
            chapter=chapter_number,
            event=f"Xuất hiện trong chương {chapter_number}",
        )

    # Update timeline
    timeline.add_event(TimelineEvent(
        chapter=chapter_number,
        time="Buổi sáng",
        event=plot_points[0] if plot_points else "Chương mới",
        location=location,
        characters_involved=main_characters,
    ))

    return chapter_content

# =========================================================================
# Step 4: Check Consistency
# =========================================================================

def check_consistency(chapter_content, chapter_number):
    """Check chapter for consistency issues."""

    issues = []

    # Check character consistency
    for name in char_manager.list_characters():
        char = char_manager.get_character(name)
        prompt = f"""Kiểm tra văn bản có nhất quán với nhân vật {name} không:

Thông tin nhân vật:
{char.get_profile_text()}

Văn bản:
{chapter_content[:1000]}

Trả về 'OK' nếu nhất quán, hoặc mô tả vấn đề."""

        response = rag.llm.generate(prompt)
        if "ok" not in response.lower():
            issues.append(f"Nhân vật {name}: {response}")

    # Check unresolved foreshadowing
    unresolved = plot_manager.get_unresolved_foreshadowing()
    for fs in unresolved:
        issues.append(f"Foreshadowing chưa giải quyết: {fs.hint}")

    return issues

# =========================================================================
# Step 5: Get Writing Suggestions
# =========================================================================

def get_suggestions(current_chapter):
    """Get suggestions for next chapter."""

    # Get unresolved foreshadowing
    unresolved = plot_manager.get_unresolved_foreshadowing()

    # Get active arcs
    active_arcs = plot_manager.get_active_arcs()

    prompt = f"""Dựa vào tình hình hiện tại:

Foreshadowing chưa giải quyết:
{chr(10).join([f"- {fs.hint}" for fs in unresolved])}

Plot arcs đang diễn ra:
{chr(10).join([f"- {arc.name}: {arc.description}" for arc in active_arcs])}

Gợi ý cho chương tiếp theo:
1. Nên giải quyết foreshadowing nào?
2. Plot arc nên phát triển thế nào?
3. Nhân vật nào nên xuất hiện?
4. Sự kiện gì nên xảy ra?"""

    return rag.llm.generate(prompt)
```

### Character Management

```python
# Get character context
context = char_manager.get_character_context("Nguyễn Văn A")
print(context)

# Get relationships
relationships = char_manager.get_relationships_context("Nguyễn Văn A")
print(relationships)

# Get development arc
arc = char_manager.get_development_arc("Nguyễn Văn A")
print(arc)

# Find related characters
related = char_manager.find_related_characters("Nguyễn Văn A")
print(related)
```

### Plot Management

```python
# Get plot context
context = plot_manager.get_plot_context()
print(context)

# Get unresolved foreshadowing
unresolved = plot_manager.get_unresolved_foreshadowing()
for fs in unresolved:
    print(f"Chương {fs.chapter_planted}: {fs.hint}")

# Resolve foreshadowing
plot_manager.resolve_foreshadowing(
    "fs_1",
    resolution="Bức thư bị ngấm thuốc độc",
    resolution_chapter=10
)
```

### World Building

```python
# Get world context
context = world_builder.get_world_context()
print(context)

# Get location details
location = world_builder.get_location_context("Đà Lạt")
print(location)

# Get secret lore
secrets = world_builder.get_secret_lore()
for lore in secrets:
    print(f"{lore.name}: {lore.content}")
```

### Consistency Checking

```python
# Check chapter
issues = consistency_checker.check_chapter(chapter_text, chapter_number)
for issue in issues:
    print(f"[{issue.severity}] {issue.description}")

# Check character
issues = consistency_checker.check_character_consistency("A")
for issue in issues:
    print(f"[{issue.severity}] {issue.description}")

# Get full report
report = consistency_checker.get_consistency_report()
print(report)
```

---

## 🎨 RAG Patterns

### Naive RAG

```python
from src.rag import NaiveRAG
rag = NaiveRAG()
rag.add_documents(["docs/"])
answer = rag.query("What is Python?")
```

### Advanced RAG

```python
from src.rag import AdvancedRAG
rag = AdvancedRAG(use_hybrid=True, use_reranking=True)
rag.add_documents(["docs/"])
answer = rag.query("What is Python?")
```

### Agentic RAG

```python
from src.rag import AgenticRAG
rag = AgenticRAG()
rag.add_documents(["docs/"])
answer = rag.query("What is Python?")
```

### Graph RAG

```python
from src.rag import GraphRAG
rag = GraphRAG()
rag.add_documents(["docs/"])
answer = rag.query("What is the relationship between X and Y?")
```

### Advanced Techniques

```python
from src.rag.advanced_techniques import SelfRAG, CorrectiveRAG, HyDE

# Self-RAG
rag = SelfRAG(llm=llm, retriever=retriever)
answer = rag.query("What is Python?")

# HyDE
hyde = HyDE(llm=llm, embeddings=embeddings)
results = hyde.search("What is Python?", documents)
```

---

## 🌐 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Query RAG system |
| `POST` | `/documents` | Add documents |
| `GET` | `/documents` | List documents |
| `POST` | `/ingest` | Upload files |
| `POST` | `/search` | Search documents |

### Query Example

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "What is Python?",
    "k": 5,
    "transform_query": True,
    "grade_documents": True
})

result = response.json()
print(result["answer"])
print(result["sources"])
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI.

---

## 🖥️ Web UI

### Features

- 💬 **Chat Interface** — Chat with your knowledge base
- 📄 **Document Upload** — Upload and index documents
- 🔍 **Search** — Search documents
- 📊 **Statistics** — View usage statistics

### Launch

```bash
python examples/07_web_ui.py
```

Visit `http://localhost:7860`

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

# Run specific test
pytest tests/test_new_features.py
```

### Test Results

```
============================= 58 passed, 1 skipped ==============================
```

---

## 📁 Project Structure

```
ultimate-rag/
├── src/
│   ├── core/                    # Core components
│   │   ├── document_loader.py
│   │   ├── text_splitter.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── llm.py
│   │   ├── streaming.py         # Token streaming
│   │   ├── memory.py            # Conversation memory
│   │   └── advanced_chunking.py # Semantic, proposition chunking
│   │
│   ├── rag/                     # RAG implementations
│   │   ├── naive_rag.py
│   │   ├── advanced_rag.py
│   │   ├── agentic_rag.py
│   │   ├── graph_rag.py
│   │   └── advanced_techniques.py
│   │
│   ├── story/                   # Story writing system
│   │   ├── character_manager.py
│   │   ├── plot_manager.py
│   │   ├── world_builder.py
│   │   ├── consistency_checker.py
│   │   ├── chapter_manager.py
│   │   ├── timeline_manager.py
│   │   └── writing_assistant.py
│   │
│   ├── api/                     # REST API
│   │   └── app.py
│   │
│   ├── ui/                      # Web UI
│   │   └── __init__.py
│   │
│   ├── auth/                    # Authentication
│   │   ├── auth_manager.py
│   │   ├── api_key_manager.py
│   │   └── rate_limiter.py
│   │
│   ├── monitoring/              # Monitoring
│   │   └── metrics_collector.py
│   │
│   ├── documents/               # Document management
│   │   └── manager.py
│   │
│   ├── agents/                  # Specialized agents
│   │   ├── retrieval_agent.py
│   │   ├── grading_agent.py
│   │   └── query_rewriter.py
│   │
│   ├── evaluation/              # Evaluation
│   │   ├── metrics.py
│   │   └── evaluator.py
│   │
│   └── utils/                   # Utilities
│       ├── config.py
│       ├── cache.py
│       └── logger.py
│
├── examples/                    # Example scripts
│   ├── 00_demo_no_api.py
│   ├── 01_naive_rag.py
│   ├── 02_advanced_rag.py
│   ├── 03_agentic_rag.py
│   ├── 04_production_rag.py
│   ├── 05_graph_rag.py
│   ├── 06_api_server.py
│   ├── 07_web_ui.py
│   └── 08_story_writing.py
│
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
├── docs/                        # Documentation
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Quick Steps

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) — LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent orchestration
- [HuggingFace](https://huggingface.co) — Transformers and embeddings
- [FAISS](https://faiss.ai) — Vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework
- [Gradio](https://gradio.app) — Web UI framework

---

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Taitv01/rag-framework-2026/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/Taitv01/rag-framework-2026/discussions)

---

<div align="center">

**Made with ❤️ for Story Writers and AI Enthusiasts**

[⬆ Back to Top](#-ultimate-rag-framework)

</div>
]]>
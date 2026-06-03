<div align="center">

# 🚀 Ultimate RAG Framework

### Retrieval-Augmented Generation cho AI Models (2026)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-116%20Passed-brightgreen?style=for-the-badge)](#-testing)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.25%2B-FF6B6B?style=for-the-badge)](https://gradio.app)

<br />

Framework RAG toàn diện, hỗ trợ **nhiều paradigm RAG** — từ tìm kiếm vector cơ bản đến hệ thống agentic thông minh với khả năng tự ra quyết định.

**🎯 Phù hợp cho:** Chatbot • Cơ sở tri thức • Viết truyện • Hỏi đáp tài liệu • Trợ lý nghiên cứu

[Quick Start](#-quick-start) • [Tính năng](#-tính-năng) • [Cài đặt](#-cài-đặt) • [Ví dụ](#-ví-dụ-sử-dụng) • [API](#-api) • [Tài liệu](#-tài-liệu)

</div>

---

## 📖 Mục lục

- [Tại sao chọn Ultimate RAG?](#-tại-sao-chọn-ultimate-rag)
- [Tính năng](#-tính-năng)
- [Cài đặt](#-cài-đặt)
- [Quick Start](#-quick-start)
- [Ví dụ sử dụng](#-ví-dụ-sử-dụng)
- [Hệ thống viết truyện](#-hệ-thống-viết-truyện)
- [API Reference](#-api-reference)
- [Web UI](#-web-ui)
- [Cấu hình](#-cấu-hình)
- [Testing](#-testing)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Đóng góp](#-đóng-góp)
- [License](#-license)

---

## 🎯 Tại sao chọn Ultimate RAG?

| Vấn đề | Giải pháp |
|---------|-----------|
| 🔍 **Tìm kiếm nông** | Hybrid search (vector + BM25) cho recall tốt hơn |
| 📊 **Ngữ cảnh không liên quan** | Cross-encoder re-ranking cho precision cao |
| 🧠 **Lý luận phức tạp** | Agentic RAG với LangGraph cho quyết định đa bước |
| 📚 **Tài liệu dài** | Advanced chunking (semantic, proposition, contextual) |
| 💬 **Chat đa lượt** | Conversation memory (buffer/window/summary) |
| 🇻🇳 **Tiếng Việt** | Vietnamese NLP processor, embedding tối ưu, song ngữ |
| 🔐 **Bảo mật production** | API key, rate limiting, JWT authentication |
| 📈 **Monitoring** | Theo dõi usage, analytics, error tracking |
| 📖 **Viết truyện** | Quản lý nhân vật, cốt truyện, thế giới, kiểm tra nhất quán |
| 🤖 **Kiểm tra ảo giác** | Hallucination Grader — xác minh câu trả lời grounded |
| 🌐 **Tìm kiếm web** | Web search fallback khi retrieval không đủ |
| 📊 **Metadata thông minh** | Tự động gán nhân vật, địa điểm, thời gian cho chunks |

---

## ✨ Tính năng

### 🎨 Các mô hình RAG

| Mô hình | Mô tả | Sử dụng |
|---------|-------|---------|
| **Naive RAG** | Tìm kiếm vector cơ bản + LLM | Q&A đơn giản, chatbot |
| **Advanced RAG** | Hybrid search + re-ranking + cache + HyDE | Tài liệu kỹ thuật, nghiên cứu |
| **Agentic RAG** | Agent-based với LangGraph | Lý luận phức tạp |
| **Graph RAG** | Knowledge graph (NetworkX + Neo4j) | Truy vấn mối quan hệ |
| **Self-RAG** | Tự phản ánh, tự đánh giá | Ứng dụng cần chất lượng cao |
| **Corrective RAG** | Tự sửa lỗi retrieval | Fact-checking |
| **HyDE** | Hypothetical Document Embedding | Matching ngữ nghĩa tốt hơn |

### 🇻🇳 Hỗ trợ tiếng Việt

- **Vietnamese Processor** — Tách từ (`underthesea`), tách câu, phát hiện ngôn ngữ, NER cho truyện cổ tích
- **Vietnamese Embeddings** — Default: `keepitreal/vietnamese-sbert` (768d)
- **Vietnamese Reranker** — `AITeamVN/Vietnamese_Reranker` (MRR@10: 86.72)
- **BM25 Vietnamese** — Tokenization tiếng Việt cho hybrid search
- **Song ngữ** — Tất cả prompt đều VI/EN

### 🤖 Phase 3: Nâng cao

- **Metadata Enhancement** — LLM tự gán nhân vật/địa điểm/thời gian/chủ đề/cảm xúc cho chunks
- **Hallucination Grader** — Kiểm tra từng claim trong câu trả lời có grounded trong tài liệu
- **Neo4j Integration** — Persistent graph storage alongside NetworkX
- **Web Search Fallback** — Tìm kiếm web khi retrieval thất bại (OFF by default, safety-first)

### 🔧 Thành phần cốt lõi

- **📄 Document Loader** — PDF, DOCX, HTML, Markdown, CSV, JSON
- **✂️ Text Splitter** — Recursive, semantic, proposition, contextual headers
- **📊 Advanced Chunking** — Semantic, Proposition, Contextual Retrieval (Anthropic), Parent-Child
- **💾 Semantic Cache** — Cache theo embedding similarity (giảm 50-70% LLM calls)
- **🔢 Embeddings** — HuggingFace, OpenAI, Cohere
- **💾 Vector Store** — FAISS, ChromaDB, Qdrant
- **🔍 Retriever** — Similarity, hybrid, MMR, re-ranking, multi-query RRF, HyDE, parent-child
- **🤖 LLM** — OpenAI, Anthropic, Ollama (local)

### 🚀 Production Features

- ✅ **RESTful API** — FastAPI endpoints
- ✅ **Web UI** — Gradio interface
- ✅ **Streaming** — Token-by-token response
- ✅ **Memory** — Conversation history management
- ✅ **Auth** — API key management, rate limiting, JWT
- ✅ **Monitoring** — Usage tracking, analytics, error monitoring
- ✅ **Document Management** — CRUD operations, versioning

---

## 📦 Cài đặt

### Yêu cầu

- Python 3.10+ (khuyến nghị 3.11 hoặc 3.12)
- pip

### Cài đặt nhanh

```bash
# 1. Clone repository
git clone https://github.com/Taitv01/rag-framework-2026.git
cd rag-framework-2026

# 2. Tạo virtual environment
python -m venv venv

# 3. Kích hoạt venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Cài dependencies
pip install -r requirements.txt

# 5. Copy file cấu hình
cp .env.example .env
# Chỉnh sửa .env với API keys của bạn
```

### Cài đặt theo module (tùy chọn)

```bash
# Cài đầy đủ (bao gồm web search + Neo4j)
pip install -e ".[all]"

# Chỉ cài web search (DuckDuckGo, Tavily)
pip install -e ".[web]"

# Chỉ cài Neo4j graph storage
pip install -e ".[graph]"

# Cài dev tools (pytest, black, ruff)
pip install -e ".[dev]"
```

### Cài đặt thủ công từng package

```bash
# Core (bắt buộc)
pip install langchain langchain-core langchain-community langchain-text-splitters
pip install faiss-cpu sentence-transformers
pip install pypdf python-docx beautifulsoup4 markdown rank-bm25
pip install fastapi uvicorn python-multipart
pip install pydantic python-dotenv rich tiktoken

# Vietnamese NLP (khuyến nghị)
pip install underthesea pyvi

# Web Search (tùy chọn)
pip install duckduckgo-search
pip install tavily-python

# Neo4j (tùy chọn)
pip install neo4j
```

### Cấu hình môi trường

Chỉnh sửa file `.env`:

```bash
# ===== LLM API Keys =====
OPENAI_API_KEY=sk-...
# Hoặc dùng Anthropic:
# ANTHROPIC_API_KEY=sk-ant-...

# ===== Web Search (tùy chọn) =====
# Tavily (khuyến nghị cho RAG): https://tavily.com
# TAVILY_API_KEY=tvly-...
# DuckDuckGo: miễn phí, không cần API key

# ===== Neo4j (tùy chọn) =====
# NEO4J_URI=bolt://localhost:7687
# NEO4J_PASSWORD=your_password

# ===== Cấu hình mặc định =====
DEFAULT_EMBEDDING_MODEL=keepitreal/vietnamese-sbert
DEFAULT_RERANKER_MODEL=AITeamVN/Vietnamese_Reranker
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=5
```

---

## 🚀 Quick Start

### 1️⃣ Demo không cần API Key

```bash
python examples/00_demo_no_api.py
```

Chạy demo hoàn chỉnh bằng HuggingFace embeddings local — không cần API key!

### 2️⃣ Basic RAG (3 dòng code)

```python
from src.rag import NaiveRAG

rag = NaiveRAG()
rag.add_documents(["document.pdf", "article.txt"])
answer = rag.query("Nội dung chính là gì?")
print(answer)
```

### 3️⃣ Advanced RAG (tối ưu cho tiếng Việt)

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
answer = rag.query("Thạch Sanh là ai?")
print(answer)
```

### 4️⃣ Advanced RAG với tất cả tính năng Phase 3

```python
from src.rag import AdvancedRAG

rag = AdvancedRAG(
    llm_provider="openai",
    use_hybrid=True,
    use_reranking=True,
    # Phase 2
    use_cache=True,
    use_contextual_chunking=True,  # Anthropic contextual retrieval
    use_hyde=True,                 # Hypothetical Document Embedding
    use_multi_query_rrf=True,      # Multi-query RRF
    # Phase 3
    use_metadata_enhancement=True, # Tự gán metadata nhân vật/địa điểm/thời gian
    use_hallucination_check=True,  # Kiểm tra ảo giác
    use_web_search=True,           # Web search fallback (OFF by default)
    web_search_provider="duckduckgo",
)

rag.add_documents(["docs/"])
answer = rag.query("Thạch Sanh đánh đại bàng như thế nào?")
```

### 5️⃣ Graph RAG với Neo4j

```python
from src.rag import GraphRAG

# Không cần Neo4j (dùng NetworkX in-memory)
rag = GraphRAG()
rag.add_documents(["docs/"])
answer = rag.query("Mối quan hệ giữa Thạch Sanh và Lý Thông?")

# Với Neo4j (persistent storage)
rag = GraphRAG(
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="your_password",
)
rag.add_documents(["docs/"])
answer = rag.query("Ai là kẻ thù của Thạch Sanh?")
```

### 6️⃣ REST API Server

```bash
python examples/06_api_server.py
# Truy cập: http://localhost:8000/docs
```

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "Thạch Sanh là ai?",
    "k": 5
})
print(response.json()["answer"])
```

### 7️⃣ Web UI

```bash
python examples/07_web_ui.py
# Truy cập: http://localhost:7860
```

---

## 📚 Ví dụ sử dụng

### Streaming Response

```python
from src.rag import AdvancedRAG
from src.core.memory import ConversationalRAG

rag = AdvancedRAG(use_hybrid=True)
rag.add_documents(["docs/"])

conv_rag = ConversationalRAG(rag, memory_type="buffer")

# Stream response
for token in conv_rag.stream("Kể về Thạch Sanh"):
    print(token, end="", flush=True)
```

### Conversation Memory

```python
from src.core.memory import ConversationalRAG

conv_rag = ConversationalRAG(rag, memory_type="window", max_history=10)

# Multi-turn conversation
answer1 = conv_rag.query("Thạch Sanh là ai?")
answer2 = conv_rag.query("Anh ta đã làm gì?")  # Nhớ context
answer3 = conv_rag.query("So sánh với Thánh Gióng")  # Vẫn nhớ
```

### Metadata Enhancement

```python
from src.core.metadata_enhancer import MetadataEnhancer

enhancer = MetadataEnhancer(llm=rag.llm)
enhanced_chunks = enhancer.enhance(chunks)

# Mỗi chunk giờ có metadata:
# chunk.metadata["characters"] = ["Thạch Sanh", "Lý Thông"]
# chunk.metadata["locations"] = ["hang đại bàng"]
# chunk.metadata["time_period"] = "xưa"
# chunk.metadata["topic"] = "Thạch Sanh giết đại bàng cứu công chúa"
# chunk.metadata["sentiment"] = "bi tráng"
```

### Hallucination Grading

```python
from src.agents.hallucination_grader import HallucinationGrader

grader = HallucinationGrader(llm=rag.llm)

# Kiểm tra câu trả lời
grade = grader.grade(
    answer="Thạch Sanh giết đại bàng bằng cung tên.",
    context="Thạch Sanh dùng cây đàn đánh đại bàng."
)

print(grade.is_grounded)        # False
print(grade.grounded_score)     # 0.3
print(grade.unsupported_claims) # ["giết đại bàng bằng cung tên"]

# Safe generate với auto-verification
answer, grade = grader.safe_generate(question, context, max_retries=2)
```

### Web Search (an toàn)

```python
from src.core.web_search import SafeWebSearcher, DuckDuckGoSearchProvider

provider = DuckDuckGoSearchProvider()
searcher = SafeWebSearcher(provider=provider, llm=rag.llm)

# Tìm kiếm + verify relevance
results = searcher.search("Thạch Sanh là ai?", num_results=3)
docs = searcher.to_documents(results)

# Mỗi kết quả web đều có metadata:
# docs[0].metadata["source_type"] = "web"
# docs[0].metadata["url"] = "https://vi.wikipedia.org/wiki/Thạch_Sanh"
```

### Advanced Chunking

```python
from src.core.advanced_chunking import (
    SemanticChunker,
    PropositionChunker,
    ContextualRetrievalChunker,
    ParentChildChunker,
)

# Semantic chunking — chia theo ngữ nghĩa
chunker = SemanticChunker(embeddings, threshold=0.8)
chunks = chunker.split(documents)

# Contextual Retrieval (Anthropic pattern) — thêm context cho mỗi chunk
chunker = ContextualRetrievalChunker(llm, chunk_size=500)
chunks = chunker.split(documents)
# Mỗi chunk giờ bắt đầu: "[Context: Đoạn này kể về...]"

# Parent-Child — tìm bằng child, trả về parent
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
results = retriever.hybrid_search("Thạch Sanh đánh đại bàng", k=5)

# Re-ranking với cross-encoder
results = retriever.search_with_reranking("câu chuyện Thạch Sanh", k=5)

# Multi-query + RRF
results = retriever.multi_query_rrf_search("Thạch Sanh là ai?", k=5, llm=llm)

# HyDE — tìm bằng câu trả lời giả thuyết
results = retriever.hyde_search("Thạch Sanh là ai?", k=5, llm=llm)

# Parent-Child search
results = retriever.parent_child_search("Thạch Sanh", k=5, parent_chunks=parents)
```

---

## 📖 Hệ thống viết truyện

```python
from src.story import (
    CharacterManager, Character,
    PlotManager, PlotPoint, PlotArc,
    WorldBuilder, Location,
    ConsistencyChecker,
    ChapterManager, TimelineManager,
)

# Tạo nhân vật
char_mgr = CharacterManager()
char_mgr.add_character(Character(
    name="Thạch Sanh",
    personality="Thật thà, dũng cảm, hiếu thảo",
    backstory="Con mồ côi, sống dưới gốc cây",
    strengths=["Sức mạnh phi thường", "Tài bắn cung"],
    weaknesses=["Quá thật thà"],
))

# Quản lý cốt truyện
plot_mgr = PlotManager()
plot_mgr.add_plot_point(PlotPoint(
    chapter=1,
    event="Thạch Sanh gặp Lý Thông",
    importance="high",
    characters_involved=["Thạch Sanh", "Lý Thông"],
))

# Xây dựng thế giới
world = WorldBuilder()
world.add_location(Location(
    name="Hang đại bàng",
    description="Hang tối sâu hun hút, nơi giam giữ công chúa",
    first_appearance=3,
))

# Kiểm tra nhất quán
checker = ConsistencyChecker(llm)
issues = checker.check_chapter(chapter_text, chapter_number)
```

---

## 🌐 API Reference

### Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Truy vấn RAG |
| `POST` | `/documents` | Thêm documents |
| `GET` | `/documents` | Liệt kê documents |
| `POST` | `/ingest` | Upload files |
| `POST` | `/search` | Tìm kiếm documents |

### Ví dụ

```python
import requests

# Query
response = requests.post("http://localhost:8000/query", json={
    "question": "Thạch Sanh là ai?",
    "k": 5,
    "transform_query": True,
    "grade_documents": True,
})
print(response.json()["answer"])

# Add documents
response = requests.post("http://localhost:8000/documents", json={
    "texts": ["Thạch Sanh là nhân vật trong truyện cổ tích Việt Nam..."],
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
# Truy cập: http://localhost:7860
```

Tính năng:
- 💬 **Chat Interface** — Chat với knowledge base
- 📄 **Document Upload** — Upload và index documents
- 🔍 **Search** — Tìm kiếm documents
- 📊 **Statistics** — Xem thống kê usage

---

## ⚙️ Cấu hình

### Biến môi trường

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `OPENAI_API_KEY` | — | API key cho OpenAI |
| `ANTHROPIC_API_KEY` | — | API key cho Anthropic |
| `DEFAULT_EMBEDDING_MODEL` | `keepitreal/vietnamese-sbert` | Model embedding mặc định |
| `DEFAULT_RERANKER_MODEL` | `AITeamVN/Vietnamese_Reranker` | Model re-ranking mặc định |
| `CHUNK_SIZE` | `500` | Kích thước chunk |
| `CHUNK_OVERLAP` | `50` | Độ chồng chunk |
| `RETRIEVAL_K` | `5` | Số documents retrieve |
| `TAVILY_API_KEY` | — | API key cho Tavily web search |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `LOG_LEVEL` | `INFO` | Mức log |

### Programmatic

```python
from src.utils.config import Config

config = Config()
llm_config = config.get_llm_config()
rag_config = config.get_rag_config()
```

---

## 🧪 Testing

```bash
# Chạy tất cả tests
pytest -v

# Chạy với coverage
pytest --cov=src

# Chạy test cụ thể
pytest tests/test_phase3_features.py -v

# Chạy test theo module
pytest tests/test_retriever.py -v
pytest tests/test_rag_pipeline.py -v
```

### Kết quả

```
============================= 116 passed in 20.82s ==============================

tests/test_document_loader.py    :  8 tests  ✅
tests/test_new_features.py       : 39 tests  ✅
tests/test_phase3_features.py    : 33 tests  ✅
tests/test_rag_pipeline.py       : 20 tests  ✅
tests/test_retriever.py          : 16 tests  ✅
```

---

## 📁 Cấu trúc dự án

```
rag-framework-2026/
├── src/
│   ├── core/                        # Thành phần cốt lõi
│   │   ├── document_loader.py       # Đọc tài liệu (PDF, DOCX, HTML...)
│   │   ├── text_splitter.py         # Chia văn bản
│   │   ├── embeddings.py            # Quản lý embedding models
│   │   ├── vector_store.py          # Quản lý vector stores
│   │   ├── retriever.py             # Tìm kiếm (hybrid, MMR, RRF, HyDE)
│   │   ├── llm.py                   # Quản lý LLM providers
│   │   ├── streaming.py             # Token streaming
│   │   ├── memory.py                # Conversation memory
│   │   ├── advanced_chunking.py     # Semantic, proposition, contextual chunking
│   │   ├── vietnamese_processor.py  # Xử lý tiếng Việt
│   │   ├── metadata_enhancer.py     # [Phase 3] Metadata enhancement
│   │   ├── graph_store.py           # [Phase 3] Neo4j backend
│   │   └── web_search.py            # [Phase 3] Web search fallback
│   │
│   ├── rag/                         # Các mô hình RAG
│   │   ├── naive_rag.py             # Basic RAG
│   │   ├── advanced_rag.py          # Advanced RAG (hybrid, reranking, cache)
│   │   ├── agentic_rag.py           # Agentic RAG (LangGraph)
│   │   ├── graph_rag.py             # Graph RAG (Knowledge Graph + Neo4j)
│   │   └── advanced_techniques.py   # Self-RAG, CRAG, HyDE, HyPE
│   │
│   ├── agents/                      # Agents chuyên biệt
│   │   ├── retrieval_agent.py       # Agent quyết định retrieval
│   │   ├── grading_agent.py         # Agent đánh giá tài liệu
│   │   ├── query_rewriter.py        # Agent tối ưu query
│   │   └── hallucination_grader.py  # [Phase 3] Kiểm tra ảo giác
│   │
│   ├── story/                       # Hệ thống viết truyện
│   │   ├── character_manager.py     # Quản lý nhân vật
│   │   ├── plot_manager.py          # Quản lý cốt truyện
│   │   ├── world_builder.py         # Xây dựng thế giới
│   │   ├── consistency_checker.py   # Kiểm tra nhất quán
│   │   ├── chapter_manager.py       # Quản lý chương
│   │   ├── timeline_manager.py      # Quản lý timeline
│   │   └── writing_assistant.py     # Trợ lý viết
│   │
│   ├── api/                         # REST API
│   │   └── app.py                   # FastAPI endpoints
│   │
│   ├── ui/                          # Web UI
│   │   └── __init__.py              # Gradio interface
│   │
│   ├── auth/                        # Xác thực
│   │   ├── auth_manager.py          # JWT authentication
│   │   ├── api_key_manager.py       # API key management
│   │   └── rate_limiter.py          # Rate limiting
│   │
│   ├── monitoring/                  # Monitoring
│   │   └── metrics_collector.py     # Thu thập metrics
│   │
│   ├── documents/                   # Quản lý tài liệu
│   │   └── manager.py               # CRUD operations
│   │
│   ├── evaluation/                  # Đánh giá
│   │   ├── metrics.py               # RAGAS metrics
│   │   └── evaluator.py             # Evaluation pipeline
│   │
│   └── utils/                       # Utilities
│       ├── config.py                # Configuration
│       ├── cache.py                 # Cache (exact + semantic)
│       └── logger.py                # Logging
│
├── examples/                        # Ví dụ (00-08)
├── tests/                           # Unit tests (116 tests)
├── notebooks/                       # Jupyter notebooks
├── docs/                            # Tài liệu
├── PHASES.md                        # Lộ trình phát triển
├── pyproject.toml                   # Project config
├── requirements.txt                 # Dependencies
└── README.md                        # ← Bạn đang đọc
```

---

## 🤝 Đóng góp

Chào mừng đóng góp! Xem [Contributing Guide](CONTRIBUTING.md).

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Mở Pull Request

---

## 📄 License

MIT License — xem [LICENSE](LICENSE).

---

## 🙏 Cảm ơn

- [LangChain](https://langchain.com) — LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent orchestration
- [HuggingFace](https://huggingface.co) — Transformers & embeddings
- [FAISS](https://faiss.ai) — Vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework
- [Gradio](https://gradio.app) — Web UI framework
- [Anthropic](https://www.anthropic.com) — Contextual Retrieval technique

---

## 📞 Hỗ trợ

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Taitv01/rag-framework-2026/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/Taitv01/rag-framework-2026/discussions)

---

<div align="center">

**Made with ❤️ for Vietnamese Story Writers and AI Enthusiasts**

[⬆ Back to Top](#-ultimate-rag-framework)

</div>

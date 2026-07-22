# 🚀 Ultimate RAG Framework (2026 Edition)

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.1.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Vietnamese NLP](https://img.shields.io/badge/NLP-Vietnamese_Aware-red.svg)](https://huggingface.co/BAAI/bge-m3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Ultimate RAG Framework** là bộ công cụ Python toàn diện cho việc xây dựng, đánh giá và triển khai các hệ thống **Retrieval-Augmented Generation (RAG)** hiện đại. 

Hệ thống được tối ưu hóa đặc biệt cho **Tiếng Việt**, hỗ trợ đa dạng pipeline RAG (Naive, Advanced, Agentic, Graph, Adaptive, Cross-Story), cơ chế **tự động cập nhật file Markdown theo hash SHA-256**, cùng khả năng **Real-time SSE Streaming** và **Langfuse Tracing** đạt chuẩn Production.

---

## 🌟 Tính Năng Nổi Bật (Key Capabilities)

### 1. 🇻🇳 Tối Ưu Hóa Ngôn Ngữ Tiếng Việt (Vietnamese NLP Engine)
- **Embedding Models**: Hỗ trợ mặc định `BAAI/bge-m3` (1024d), `keepitreal/vietnamese-sbert` (768d), `bkai-foundation-models/vietnamese-bi-encoder`.
- **Vietnamese Reranker**: Tích hợp `AITeamVN/Vietnamese_Reranker` & `Qwen/Qwen3-Reranker-0.6B`.
- **Word Segmentation**: Tách từ Tiếng Việt chuyên sâu với `underthesea` (xử lý từ ghép "Thạch_Sanh", "đại_bàng").

### 2. ⚡ Real-Time SSE Streaming & Production API
- **Server-Sent Events (SSE)**: API `/query/stream` phản hồi câu trả lời theo thời gian thực (chunk-by-chunk) cho trải nghiệm người dùng tương tác tức thì.
- **RESTful Endpoints**: `/query`, `/query/stream`, `/documents`, `/ingest`, `/search`, `/health`, `/ready`.
- **Rate Limiting & Authentication**: Giới hạn tần suất truy cập sliding-window và xác thực API-Key.

### 3. 📂 Thư Mục Markdown Refresh Tự Động (SHA-256 Incremental Ingestion)
- Tự động quét thư mục `.md`, tính toán hash SHA-256 cho từng file.
- **Chỉ cập nhật/thêm mới/xóa các file bị thay đổi** mà không cần index lại toàn bộ cơ sở dữ liệu.
- Phù hợp cho kho tri thức sống (living knowledge base): báo cáo, ghi chú MMO, tài liệu vận hành, kịch bản sáng tác.

### 4. 🧩 Đa Dạng RAG Pipelines (Multi-Pipeline Architecture)
- **Naive RAG**: Pipeline RAG cơ bản, siêu nhanh.
- **Advanced RAG**: Hybrid Search (Dense Vector + BM25) + Reranking + Parent-Child Chunking + Semantic Cache + HyDE.
- **Agentic RAG**: AI Agents tự động đánh giá độ liên quan của tài liệu, chấm điểm ảo giác (hallucination) và viết lại câu hỏi (query rewrite).
- **Graph RAG**: Kết hợp Đồ thị tri thức (Knowledge Graph) để truy xuất quan hệ nhân vật, địa điểm, sự kiện.
- **Adaptive RAG**: Tự động phân loại độ phức tạp câu hỏi (Simple/Medium/Complex) để định tuyến pipeline tối ưu nhất.
- **Cross-Story RAG**: Tìm kiếm motif, so sánh nhân vật và phân tích bài học đạo đức trong kho truyện cổ tích/tiểu thuyết.

### 5. 📚 Thư Viện Tri Thức Thông Minh & OCR Chuyển Đổi (Smart Library & OCR Engine)
- **Tự động phân loại tài liệu (Auto-Classification)**: Tự động phân bổ file vào các danh mục thư viện (`Cổ tích / Kịch bản`, `Báo cáo / Tài chính`, `Học tập / Nghiên cứu`, `Vận hành / MMO`, `Tài liệu Scan / OCR`, `Chung`).
- **Trích xuất chữ từ ảnh & PDF Scan (OCR Engine)**: Đọc file ảnh (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`) và PDF dạng scan ảnh bằng PyTesseract / EasyOCR / Pillow với thuật toán làm sạch dấu Tiếng Việt.
- **Library Manifest & Tagging**: Quản lý kho tri thức tại `library/` với file `.rag_library_manifest.json` theo dõi metadata, tóm tắt tự động, và tìm kiếm theo tag.

### 6. 📊 Production Tracing & Monitoring (Langfuse & Docker)
- **Langfuse Integration**: Đo lường chi phí token, độ trễ từng bước truy xuất, versioning prompt.
- **Graceful Fallback**: Tự động chuyển sang ghi log nội bộ mượt mà khi không cấu hình khóa Langfuse.
- **Containerization**: `docker-compose.yml` sẵn sàng với 3 dịch vụ: FastAPI Server (`rag-api`), Qdrant Vector Store (`rag-qdrant`), và Redis Cache (`rag-redis`).

---

## 🏗️ Cấu Trúc Dự Án (Project Layout)

```text
D:\RAG/
├── src/
│   ├── api/             # FastAPI REST Server & SSE Streaming endpoints
│   ├── core/            # Loaders, Splitters, Embeddings, Retrievers, Vietnamese NLP, Markdown Indexer
│   ├── rag/             # Naive, Advanced, Agentic, Graph, Adaptive, và Cross-Story RAG pipelines
│   ├── agents/          # Retrieval, Grading, Hallucination, và Query Rewrite Agents
│   ├── story/           # Consistency Checker, Character Manager, World Builder cho truyện dài
│   ├── monitoring/      # Langfuse Tracer & Metrics Collector
│   ├── evaluation/      # RAG metrics & RAGAS / DeepEval pipeline
│   └── utils/           # Semantic Cache, Config loader, Logging
├── docs/                # Tài liệu chi tiết & hướng dẫn triển khai
├── examples/            # Ví dụ mã nguồn có thể chạy trực tiếp
├── tests/               # Bộ test suite tự động (150+ test cases PASS)
├── Dockerfile           # Docker multi-stage build definition
├── docker-compose.yml   # Multi-service deployment (API + Qdrant + Redis)
├── PHASES.md            # Lộ trình phát triển qua từng giai đoạn
├── pyproject.toml       # Quản lý dependencies & cấu hình dự án
└── README.md            # Tài liệu hướng dẫn sử dụng
```

---

## ⚙️ Cài Đặt & Cấu Hình (Installation & Setup)

### 1. Khởi Tạo Môi Trường Python

```powershell
# Tạo môi trường ảo
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cập nhật pip & cài đặt dependencies
py -m pip install --upgrade pip
py -m pip install -e ".[dev]"
```

### 2. Cấu Hình Biến Môi Trường (`.env.local`)

Tạo file `.env.local` tại thư mục gốc dự án (file này được Git phớt lờ để bảo vệ bí mật):

```dotenv
# API Keys
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Tracing (Tùy chọn)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Server & Auth Settings
ENABLE_API_AUTH=false
API_KEYS=rag_secret_key_123
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

---

## 💡 Hướng Dẫn Sử Dụng Nhanh (Quick Start)

### 1. Cập Nhật Thư Mục Markdown (Incremental Ingestion)

```python
from src.rag import AdvancedRAG

# Khởi tạo Advanced RAG với Hybrid Search
rag = AdvancedRAG(retrieval_method="hybrid")

# Quét và cập nhật thư mục Markdown (chỉ index file thêm mới hoặc sửa đổi)
result = rag.refresh_markdown_directory(r"D:\path\to\your\markdown_folder")
print("Kết quả refresh:", result)

# Truy vấn dữ liệu
answer = rag.query("Các số liệu mới nhất trong tài liệu là gì?")
print("Trả lời:", answer)
```

### 2. Sử Dụng Cross-Story RAG Cho Truyện Cổ Tích / Truyện Dài

```python
from src.rag import CrossStoryRAG

cross_rag = CrossStoryRAG()

# Tìm kiếm motif cổ tích trong kho dữ liệu
motifs = cross_rag.find_motifs("con vật biết nói", top_k=2)
print("Motifs tìm thấy:", motifs)

# So sánh 2 nhân vật
comparison = cross_rag.compare_characters("Thạch Sanh", "Lý Thông")
print("So sánh nhân vật:", comparison)
```

### 3. Đọc File & Tự Động Phân Bổ Vào Thư Viện Tri Thức (Smart Library & OCR)

```python
from src.rag import NaiveRAG

rag = NaiveRAG()

# Đọc file/ảnh scan/PDF, tự động OCR và tự động phân bổ vào danh mục thư viện (library/<category>/...)
result = rag.ingest_to_library("path/to/scanned_document.png")
print("Đã phân bổ vào danh mục:", result["records"][0]["category"])
print("Đường dẫn trong thư viện:", result["records"][0]["library_path"])

# Tìm kiếm trong thư viện theo danh mục hoặc từ khóa
docs_co_tich = rag.library_manager.get_documents_by_category("cổ_tích_kịch_bản")
print("Thống kê danh mục thư viện:", rag.library_manager.list_library_categories())
```

---

## 🌐 Triển Khai API Server & Real-Time Streaming

### 1. Chạy API Server Trực Tiếp (Local)

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

* **Swagger UI Docs**: `http://localhost:8000/docs`
* **Health Check**: `http://localhost:8000/health`

### 2. Gọi API Streaming với SSE (cURL Example)

```bash
curl -X POST "http://localhost:8000/query/stream" \
     -H "Content-Type: application/json" \
     -d '{"question": "Tóm tắt quy trình xử lý Tiếng Việt trong RAG"}'
```

### 3. Chạy Bằng Docker Compose (Production)

```powershell
# Khởi động full-stack API + Qdrant + Redis
docker-compose up -d --build
```

---

## 🧪 Kiểm Thử Hệ Thống (Verification & Testing)

Chạy bộ kiểm thử tự động với `pytest`:

```powershell
# Kiểm tra biên dịch code
py -m compileall src tests

# Chạy toàn bộ unit test suite
py -m pytest -v
```

---

## 📜 Giấy Phép & Đóng Góp (License)

Dự án được phát hành theo giấy phép **[MIT License](LICENSE)**.

- **GitHub Repository**: [https://github.com/Taitv01/rag-framework-2026.git](https://github.com/Taitv01/rag-framework-2026.git)

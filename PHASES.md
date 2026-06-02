# 🗺️ Vietnamese Fairy Tales RAG - Development Roadmap

> Lộ trình phát triển Framework RAG cho Truyện Cổ Tích Việt Nam

---

## 📊 Tổng quan tiến độ

| Phase | Tên | Trạng thái | Files thay đổi |
|-------|-----|-----------|----------------|
| Phase 1 | Vietnamese Language Foundation | ✅ Hoàn thành | 15 files |
| Phase 2 | RAG Quality Improvements | ✅ Hoàn thành | 6 files |
| Phase 3 | Vietnamese Fairy Tale Features | ⏳ Chưa bắt đầu | - |
| Phase 4 | Production Readiness | ⏳ Chưa bắt đầu | - |
| Phase 5 | Advanced Features | ⏳ Chưa bắt đầu | - |

---

## ✅ Phase 1: Vietnamese Language Foundation (Hoàn thành)

### Mục tiêu
Khắc phục hoàn toàn vấn đề English-only, đưa framework về trạng thái có thể sử dụng thực tế cho tiếng Việt.

### Các thay đổi chính

#### 1.1 Vietnamese Embedding Models
- **File**: `src/core/embeddings.py`
- Default model: `all-MiniLM-L6-v2` → `keepitreal/vietnamese-sbert` (768d)
- Thêm vào catalog: `AITeamVN/Vietnamese_Embedding` (1024d), `bkai-foundation-models/vietnamese-bi-encoder` (768d), `BAAI/bge-m3` (1024d), `paraphrase-multilingual-MiniLM-L12-v2` (384d)
- Cohere default: `embed-english-v3.0` → `embed-multilingual-v3.0`

#### 1.2 Vietnamese Reranker
- **File**: `src/core/retriever.py`
- Default reranker: `ms-marco-MiniLM` → `AITeamVN/Vietnamese_Reranker` (MRR@10: 86.72)
- Fallback mechanism nếu model không load được

#### 1.3 Vietnamese Text Processor
- **File**: `src/core/vietnamese_processor.py` (NEW)
- Word segmentation với `underthesea` (compound words: "Thạch_Sanh", "đại_bàng")
- Sentence splitting xử lý viết tắt Việt Nam (TP., P., Q., ThS., TS.)
- Language detection (Vietnamese vs English)
- NER cho truyện cổ tích (nhân vật, địa điểm)
- Text preprocessing (Unicode normalization, whitespace)

#### 1.4 BM25 Vietnamese Tokenization
- **File**: `src/core/retriever.py`
- `_tokenize_for_bm25()`: dùng `underthesea` word segmentation
- Fallback regex nếu underthesea không có sẵn

#### 1.5 Vietnamese Query Variations
- **File**: `src/core/retriever.py`
- Template song ngữ (VI/EN) thay vì English-only
- "Thạch Sanh là ai?" → "Thạch Sanh là gì?", "Giải thích về Thạch Sanh"

#### 1.6 Bilingual LLM Prompts
- **Files**: `advanced_rag.py`, `agentic_rag.py`, `graph_rag.py`, `memory.py`, `advanced_chunking.py`
- Tất cả prompt đều song ngữ Việt/Anh
- Grading chấp nhận "yes"/"có"/"đúng"

#### 1.7 Bug Fixes
| Bug | File | Fix |
|-----|------|-----|
| FAISS ID collision | `vector_store.py` | `str(i)` → `str(uuid.uuid4())` |
| GraphRAG exception swallowing | `graph_rag.py` | Thêm `logger.warning/error` |
| FastAPI async blocking | `api/app.py` | `async def` → `def` cho sync endpoints |
| AgenticRAG message index | `agentic_rag.py` | `_get_question_from_state()` |
| ParentChildChunker return type | `advanced_chunking.py` | Handle tuple trong AdaptiveChunker |
| File cleanup on Windows | `api/app.py` | `finally` block + `shutil.rmtree` |

#### 1.8 Dependencies
- `requirements.txt` + `pyproject.toml`: thêm `underthesea>=6.8.0`, `pyvi>=0.1.1`
- `.env.example`: cập nhật default models

---

## ✅ Phase 2: RAG Quality Improvements (Hoàn thành)

### Mục tiêu
Cải thiện chất lượng retrieval thông qua các kỹ thuật tiên tiến nhất 2025-2026.

### Các thay đổi chính

#### 2.1 Semantic Cache
- **File**: `src/utils/cache.py`
- Cache query-answer theo embedding similarity (không phải exact match)
- Threshold configurable (default 0.95)
- TTL + LRU eviction
- `get()`, `get_with_score()`, `put()`, `clear()`, `stats()`
- **Impact**: Giảm 50-70% LLM calls cho câu hỏi tương tự

#### 2.2 Contextual Retrieval (Anthropic Pattern)
- **File**: `src/core/advanced_chunking.py`
- `ContextualRetrievalChunker`: LLM tạo context ngắn cho mỗi chunk
- Context prepended: `[Context: Đoạn này kể về Thạch Sanh giết đại bàng...]\n{chunk_text}`
- **Impact**: ~70% improvement in faithfulness (theo benchmark Anthropic)

#### 2.3 Multi-Query + RRF
- **File**: `src/core/retriever.py`
- `multi_query_rrf_search()`: LLM tạo 3-5 query variations
- `rrf_fusion()`: Reciprocal Rank Fusion kết hợp results
- RRF score = Σ(1/(k + rank_i))
- **Impact**: 26% recall improvement

#### 2.4 HyDE (Hypothetical Document Embeddings)
- **File**: `src/core/retriever.py`
- `hyde_search()`: LLM tạo câu trả lời giả thuyết, embed câu trả lời thay vì query
- Bridge embedding gap giữa query ngắn và document dài

#### 2.5 Parent-Child Retrieval
- **File**: `src/core/retriever.py`
- `parent_child_search()`: tìm kiếm bằng child chunks nhỏ, trả về parent chunks lớn
- Small chunks = precise matching, Parent chunks = rich context

#### 2.6 Streaming
- **File**: `src/rag/advanced_rag.py`
- `AdvancedRAG.stream()`: generator-based streaming
- Tích hợp với `ConversationalRAG.stream()`
- Cache check trước khi stream

#### 2.7 AdvancedRAG Integration
- **File**: `src/rag/advanced_rag.py`
- Thêm parameters: `use_cache`, `cache_ttl`, `cache_threshold`, `use_contextual_chunking`, `use_hyde`, `use_multi_query_rrf`, `num_query_variations`
- `_retrieve()`: internal method orchestrating HyDE → Multi-Query RRF → Standard search
- `cache_stats` property

---

## ⏳ Phase 3: Vietnamese Fairy Tale Features (Kế hoạch)

### Mục tiêu
Xây dựng các tính năng chuyên biệt cho truyện cổ tích Việt Nam.

#### 3.1 Fairy Tale Knowledge Graph Schema
- **File**: `src/rag/graph_rag.py`
- Ontology chuyên biệt: NhanVat, VatThe, DiaDiem, SuKien
- Relationship types: CHIEN_DAU, SO_HUU, GIAI_CUU, BIEN_THANH
- Entity extraction tối ưu cho truyện cổ tích

#### 3.2 Cross-Story Query System
- **File**: `src/rag/cross_story_rag.py` (NEW)
- `find_motifs()`: tìm motif trong nhiều truyện ("con vật biết nói")
- `compare_characters()`: so sánh nhân vật giữa các truyện
- `find_moral_patterns()`: tìm truyện theo bài học đạo đức

#### 3.3 Vietnamese Fairy Tale Dataset Builder
- **File**: `src/data/fairy_tale_builder.py` (NEW)
- Crawl + normalize truyện cổ tích từ nhiều nguồn
- Chuẩn hóa Unicode, diacritics
- Gán nhãn: nhân vật, địa điểm, motif, bài học
- Export format HuggingFace datasets

#### 3.4 Story Analysis & Improvement Engine
- **File**: `src/story/analyzer.py` (NÂNG CẤP)
- Generalize từ `analyze_ho_guom.py` thành tool tổng quát
- Phân tích: lặp từ, ý tưởng trùng, câu dài, nhịp điệu, phát triển nhân vật
- So sánh 2 phiên bản
- Gợi ý cải thiện cụ thể

#### 3.5 Vietnamese OCR Integration
- **File**: `src/core/document_loader.py`
- Tích hợp `vietnamese_ocr` cho sách scan
- Post-processing với `bmd1905/vietnamese-correction-v2`
- Xử lý dấu, chính tả

---

## ⏳ Phase 4: Production Readiness (Kế hoạch)

### Mục tiêu
Đưa framework lên production-grade với monitoring, evaluation, và deployment.

#### 4.1 Async FastAPI + Streaming
- **File**: `src/api/app.py`
- `asyncio.to_thread()` cho tất cả RAG operations
- SSE streaming cho `/query` endpoint
- WebSocket support cho real-time chat

#### 4.2 Docker + docker-compose
- **Files**: `Dockerfile`, `docker-compose.yml` (NEW)
- Services: app, qdrant, redis
- Health checks, volume mounts
- Multi-stage build

#### 4.3 RAGAS Evaluation Pipeline
- **File**: `src/evaluation/pipeline.py` (NÂNG CẤP)
- Automated evaluation với RAGAS metrics
- Faithfulness > 0.8, Answer Relevancy > 0.8
- CI/CD quality gates
- Regression testing

#### 4.4 Langfuse Integration
- **File**: `src/monitoring/langfuse_tracer.py` (NEW)
- Tracing cho tất cả LLM calls
- Cost tracking
- Prompt versioning
- User feedback collection

#### 4.5 Production Monitoring
- **File**: `src/monitoring/dashboard.py` (NEW)
- Grafana dashboards
- SLO tracking: p99 latency < 3s, cache hit > 30%, error rate < 1%
- Alerting

---

## ⏳ Phase 5: Advanced Features (Kế hoạch)

### Mục tiêu
Các tính năng nâng cao, đổi mới.

#### 5.1 Multimodal RAG (Text + Images)
- OCR cho sách truyện scan
- Image description bằng Vision model (GPT-4o, Claude)
- Index images as searchable text

#### 5.2 Agentic Workflow với LangGraph
- Agent có thể: search, analyze, generate, check accuracy, translate
- Multi-step reasoning
- Tool use

#### 5.3 Bilingual Support (VI ↔ EN)
- Translate truyện cổ tích cho international audience
- Bilingual subtitle generation
- Cultural notes translation

#### 5.4 Fine-tuned Vietnamese LLM
- Fine-tune PhoGPT-4B-Chat trên corpus truyện cổ tích
- Domain-specific vocabulary
- Style transfer: viết theo phong cách cổ tích

#### 5.5 Interactive Storytelling App
- User chọn nhân vật, bối cảnh → AI viết truyện mới
- Consistency checker đảm bảo logic
- World builder mở rộng universe
- Gradio/Streamlit UI

---

## 📋 Dependencies cho từng Phase

### Phase 3 (cần thêm)
```
# Vietnamese fairy tale specific
beautifulsoup4>=4.12.0  # (already have)
requests>=2.31.0        # for crawling
```

### Phase 4 (cần thêm)
```
# Production
docker
redis>=5.0.0
langfuse>=2.0.0
grafana-api>=1.0.0
```

### Phase 5 (cần thêm)
```
# Multimodal
pytesseract>=0.3.10
Pillow>=10.3.0  # (already have)

# Fine-tuning
peft>=0.10.0
trl>=0.8.0
datasets>=2.18.0
```

---

## 🎯 Success Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| Vietnamese embedding quality | MRR@10 > 80 | Phase 1 ✅ |
| BM25 Vietnamese tokenization | Working | Phase 1 ✅ |
| Semantic cache hit rate | > 30% | Phase 2 ✅ |
| Contextual retrieval faithfulness | > 0.8 | Phase 2 ✅ |
| Multi-query recall improvement | > 20% | Phase 2 ✅ |
| Cross-story query accuracy | > 70% | Phase 3 |
| RAGAS faithfulness | > 0.8 | Phase 4 |
| API p99 latency | < 3s | Phase 4 |
| Error rate | < 1% | Phase 4 |

---

## 🔗 Links

- **Repository**: https://github.com/Taitv01/rag-framework-2026
- **Vietnamese Embedding**: https://huggingface.co/keepitreal/vietnamese-sbert
- **Vietnamese Reranker**: https://huggingface.co/AITeamVN/Vietnamese_Reranker
- **Anthropic Contextual Retrieval**: https://www.anthropic.com/news/contextual-retrieval
- **RAGAS Docs**: https://docs.ragas.io/en/latest/

# Kế hoạch nâng cấp RAG (10/2026)

Kế hoạch dựa trên việc đọc code thực tế (pipelines, retriever, vector store, API)
và chạy test suite. Nguyên tắc: **sửa nền trước, đo trước khi tối ưu**, chỉ thêm
kỹ thuật mới khi số đo cho thấy cần.

| Giai đoạn | Nội dung | Ước lượng | Trạng thái |
|-----------|----------|-----------|------------|
| 0 | Ổn định nền | 1–2 ngày | ✅ Hoàn thành |
| 1 | Đo trước khi tối ưu (golden set + eval baseline) | 2–3 ngày | ⏳ |
| 2 | Sửa lõi retrieval | 3–5 ngày | ⏳ |
| 3 | Hợp nhất pipeline generation | 3–4 ngày | ⏳ |
| 4 | API & vận hành | 2–3 ngày | ⏳ |
| 5 | Nâng cao (tùy chọn, theo số đo) | — | ⏳ |

---

## Hiện trạng: vấn đề phát hiện

### Lỗi chặn production
1. **Qdrant backend hỏng.** `VectorStoreManager._create_qdrant_store` import
   `Qdrant` từ `langchain_community.vectorstores`, lớp này đã bị gỡ khỏi
   langchain-community 0.4.x, trong khi `docker-compose.yml` mặc định dùng Qdrant.
2. **Restart là mất index.** Mặc định là FAISS trong RAM, `persist()` không được
   gọi ở đâu. BM25, chunk và parent cũng chỉ nằm trong RAM. FAISS không hỗ trợ xoá,
   nên mỗi lần refresh phải embed lại toàn bộ.
3. **`/query/stream` chặn event loop.** Generator đồng bộ `rag.stream()` được lặp
   ngay trong async generator. Retrieve chạy 2 lần (sources gửi cho client có thể
   khác nguồn thực sự dùng) và `k` bị bỏ qua.

### Lỗi logic RAG
4. **Parent-child mới làm một nửa** (`AdvancedRAG._index_loaded_documents`):
   child chunk khoảng 250 ký tự được đưa thẳng vào LLM, parent không bao giờ được
   lấy ra. `_parent_idx` là chỉ số vị trí, sẽ lệch sau `_drop_source_root`.
5. **Web fallback không bao giờ chạy.** `_grade_documents` trả về `docs[:1]` khi
   mọi doc đều không liên quan, nên `_is_retrieval_quality_poor` luôn ra false và
   hệ thống không bao giờ trả lời "không có thông tin".
6. **Ba đường query lệch nhau.** `query`, `query_detailed`, `stream` mỗi cái tự cài
   pipeline riêng; chỉ `query()` có web search và kiểm tra hallucination.
7. **Filter bị rò.** `RetrieverManager.hybrid_search` không áp filter cho BM25.
   HyDE và multi-query RRF bỏ qua hybrid, rerank lẫn filter.
8. **AgenticRAG không giới hạn vòng lặp.** `max_retries` được lưu nhưng không dùng,
   không có `recursion_limit`. Tool retrieve chỉ dùng vector và làm mất metadata,
   nên mất luôn citation.
9. **AdaptiveRAG nhân 3 chi phí.** Ba pipeline riêng, mỗi cái tự load embedding và
   tự index lại toàn bộ tài liệu.

### Hiệu năng & chất lượng
10. **Khoảng 7 lời gọi LLM mỗi query** (k=5): viết lại câu hỏi, chấm từng doc
    tuần tự, sinh câu trả lời. Reranker bị load lại sau mỗi lần ingest
    (`_refresh_retriever` tạo `RetrieverManager` mới).
11. **Semantic cache** không bị xoá khi corpus đổi và không có key theo
    filter/tenant. Redis có trong compose nhưng code không dùng.
12. **Có metric, chưa có dữ liệu đo.** Đã có P@k, MRR, nDCG nhưng không có golden
    set tiếng Việt, nên mọi tối ưu chưa kiểm chứng được.

### Vệ sinh dự án (đã xử lý ở Giai đoạn 0)
13. `.env.local` được load với `override=True` nên đè env của test: 3 test fail
    và assertion in ra API key thật.
14. Default model không thống nhất (`keepitreal/vietnamese-sbert` và `BAAI/bge-m3`).
    Pipeline truyền model cứng của OpenAI cho mọi provider.
15. `pip install -e .` không chạy được (hatchling không biết đóng gói `src`).
    Dependency nặng và không dùng nằm trong phần lõi.

---

## Giai đoạn 0: Ổn định nền ✅

- [x] Commit phần multimodal/Ox Alpha đang dở lên nhánh `upgrade/phase-0-stabilize`.
- [x] Cô lập test khỏi secret cục bộ: `RAG_DISABLE_DOTENV=1` và `tests/conftest.py`
      xoá các biến môi trường giống secret.
- [x] Một nguồn duy nhất cho default model: `DEFAULT_LLM_MODELS` /
      `DEFAULT_EMBEDDING_MODELS` trong `src/utils/config.py`. Pipeline mặc định
      `llm_model=None`, `embedding_model=None` để provider tự chọn model phù hợp.
- [x] Tách dependency: lõi nhẹ (không torch) và các extras `local-models`, `qdrant`,
      `chroma`, `ocr`, `monitoring`, `eval`, `api`, `ui`, `web`, `graph`, `dev`.
      Gỡ các package không dùng (`langchain` meta, `networkx`, `markdown`,
      `FlagEmbedding`, `litellm`, `instructor`, `rich`, `pyjwt`...).
- [x] Sửa `pip install -e .` (`[tool.hatch.build.targets.wheel] packages = ["src"]`).
- [x] Cố định rule ruff (`E4, E7, E9, F`) và sửa toàn bộ lỗi lint, trong đó có bug
      `plot_context` không được đưa vào prompt (`WritingAssistant`).
- [x] CI GitHub Actions: ruff, compileall, pytest trên Python 3.11–3.13.

**Kết quả:** 194/194 test pass (trước đó 189 pass / 3 fail), ruff sạch,
`.venv` dựng lại từ `pip install -e ".[dev,api]"` không có torch.

## Giai đoạn 1: Đo trước khi tối ưu
- Golden set 50–100 câu từ corpus thật, gồm cả câu **không có đáp án**.
- `scripts/eval.py`: recall@k, nDCG, faithfulness, latency p50/p95 và số lời gọi
  LLM mỗi query; xuất baseline JSON để so sánh mọi thay đổi về sau.
- ✅ Xong khi: có baseline số liệu.

## Giai đoạn 2: Sửa lõi retrieval
- `langchain-qdrant` (`QdrantVectorStore`). FAISS dùng `save_local`/`load_local`
  và truyền đúng ids.
- Lưu chunk store xuống đĩa, hoặc dùng hybrid native của Qdrant
  (dense + sparse BM25/BGE-M3) để bỏ BM25 trong RAM.
- Parent-child thật: `parent_id` dạng hash ổn định, đưa parent vào LLM.
- Áp filter cho BM25; HyDE và multi-query đi qua cùng luồng hybrid + rerank; dùng RRF.
- Reranker chỉ load một lần, chạy theo batch. Cập nhật tăng dần thật theo
  `chunk_id` và manifest.
- ✅ Xong khi: Qdrant chạy trong compose, restart không mất index, recall/nDCG
  không thấp hơn baseline.

## Giai đoạn 3: Hợp nhất pipeline generation
- Một `RAGPipeline` (retrieve → grade → generate → verify) cho mọi đường query
  và API.
- Chấm độ liên quan bằng điểm reranker + ngưỡng, hoặc 1 lời gọi LLM structured
  cho cả batch.
- Không đủ ngữ cảnh thì trả lời "không có thông tin"; web fallback chạy đúng lúc.
- Kiểm tra citation `[S#]`. Cache có key theo câu hỏi, filter và phiên bản corpus;
  Redis tùy chọn.
- AgenticRAG: áp `max_retries` và `recursion_limit`, dùng chung retriever.
  AdaptiveRAG: các route dùng chung một index.
- ✅ Xong khi: số lời gọi LLM mỗi query giảm từ ~7 xuống 2–3, faithfulness không
  thấp hơn baseline.

## Giai đoạn 4: API & vận hành
- Stream bất đồng bộ thật, retrieve một lần, gửi đúng sources đã dùng.
- Langfuse trace cho từng bước (retrieve, rerank, grade, generate) kèm token và
  chi phí.
- Ingest/OCR chạy nền; filter metadata trong `QueryRequest`.

## Giai đoạn 5: Nâng cao (theo số đo)
- BGE-M3 sparse/multi-vector; contextual retrieval cho corpus truyện.
- GraphRAG có community summaries và lưu graph xuống đĩa.
- Cập nhật catalog LLM; CI chặn merge khi chất lượng tụt quá ngưỡng.

## Quyết định còn mở
1. Vector store chính: Qdrant (khuyến nghị) hay FAISS local?
2. Mục đích dùng ưu tiên: truyện cổ tích/sáng tác hay kho tài liệu
   (báo cáo, MMO)? Quyết định golden set và cách chia chunk.

# Kế hoạch nâng cấp RAG (10/2026)

Kế hoạch dựa trên việc đọc code thực tế (pipelines, retriever, vector store, API)
và chạy test suite. Nguyên tắc: **sửa nền trước, đo trước khi tối ưu**, chỉ thêm
kỹ thuật mới khi số đo cho thấy cần.

| Giai đoạn | Nội dung | Ước lượng | Trạng thái |
|-----------|----------|-----------|------------|
| 0 | Ổn định nền | 1–2 ngày | ✅ Hoàn thành |
| 1 | Đo trước khi tối ưu (golden set + eval baseline) | 2–3 ngày | ✅ Hoàn thành |
| 2 | Sửa lõi retrieval | 3–5 ngày | ✅ Hoàn thành (compose chờ CI kiểm chứng) |
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

## Giai đoạn 1: Đo trước khi tối ưu ✅

- [x] Golden set truyện cổ tích/sáng tác (`evals/fairy_tales/`): 10 tài liệu,
      95 câu (87 có đáp án, 8 không có), mỗi câu có nguồn và cụm bằng chứng
      nguyên văn. Hướng dẫn: [`evals/fairy_tales/README.md`](../evals/fairy_tales/README.md).
- [x] Sửa lỗi `RAGEvaluator` đếm trùng chunk cùng nguồn (recall/nDCG từng vượt 1).
- [x] `src/evaluation/benchmark.py` + `scripts/eval.py` (`retrieval`, `answer`,
      `compare`): recall/MRR/nDCG theo nguồn, `evidence_recall`, latency p50/p95,
      số lời gọi LLM và token mỗi query, tỉ lệ từ chối đúng, faithfulness (LLM judge).
- [x] Baseline retrieval: [`evals/fairy_tales/baselines/retrieval.json`](../evals/fairy_tales/baselines/retrieval.json).
- [x] Sửa lỗi phát hiện khi đo: reranker mặc định `AITeamVN/Vietnamese_Reranker`
      cần `sentencepiece`. Thiếu gói này, hệ thống lặng lẽ chuyển sang reranker
      tiếng Anh `ms-marco-MiniLM`.
- [x] Baseline câu trả lời: [`evals/fairy_tales/baselines/answer.json`](../evals/fairy_tales/baselines/answer.json).
      Theo quyết định của chủ dự án, pipeline chạy bằng **chính model đang gọi RAG**
      (`AgentLLM`, ở đây là `claude-opus-5-5`), không dùng API hay model nào khác.
      Toàn bộ 749 câu trả lời của model được lưu trong `agent_answers.json`, nên
      lần chạy có thể phát lại.

### Baseline retrieval (commit `098d5f1`)

AdvancedRAG mặc định: chunk cha 500, chunk con 250, `k=5`, `BAAI/bge-m3`.
Đo trên 87 câu có đáp án. Độ trễ đo trên GTX 1650 (embedding chạy CPU,
reranker chạy GPU).

| Cấu hình | recall@5 | MRR | nDCG | evidence_recall | p50 | p95 |
|----------|---------:|----:|-----:|----------------:|----:|----:|
| vector | 0.976 | 0.966 | 0.954 | 0.816 | 107 ms | 122 ms |
| vector + rerank | 0.965 | 0.946 | 0.940 | 0.817 | 780 ms | 891 ms |
| hybrid | 0.980 | 0.972 | 0.959 | 0.810 | 99 ms | 112 ms |
| hybrid + rerank (mặc định) | 0.965 | 0.952 | 0.944 | 0.829 | 778 ms | 892 ms |

`evidence_recall` theo nhóm câu hỏi:

| Nhóm | Số câu | vector | vector + rerank | hybrid | hybrid + rerank |
|------|------:|------:|------:|------:|------:|
| fact | 42 | 0.937 | 0.905 | 0.937 | 0.905 |
| sequence | 12 | 0.833 | 0.854 | 0.833 | 0.854 |
| creative (kịch bản Hồ Gươm) | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_source | 9 | 0.556 | 0.574 | 0.611 | 0.574 |
| character | 7 | 0.810 | 0.952 | 0.810 | 0.952 |
| motif | 7 | 0.500 | 0.524 | 0.571 | 0.524 |
| moral | 6 | 0.583 | 0.750 | 0.583 | 0.750 |
| no_diacritics | 5 | 0.400 | 0.200 | 0.200 | 0.400 |
| paraphrase | 4 | 0.750 | 0.750 | 0.750 | 0.750 |
| multi_hop | 2 | 0.333 | 0.458 | 0.333 | 0.458 |

### Nhận định từ số đo

1. **Chỉ số theo nguồn đã bão hoà** (~0.97) vì corpus chỉ có 10 truyện. Từ nay
   dùng `evidence_recall` làm thước đo chính. Trần của chỉ số này là 1.0: mọi cụm
   bằng chứng đều nằm trọn trong một chunk, nên các lần trượt là lỗi truy xuất thật.
2. **Đúng truyện nhưng sai đoạn.** 16 câu thiếu ít nhất một cụm bằng chứng ở cả 4 cấu hình. Nhiều
   câu lấy được 5/5 chunk của đúng truyện nhưng không có đoạn chứa đáp án
   (q022, q025, q036, q037, q055). Chunk con ~180 ký tự quá nhỏ. Đây là lý do ưu
   tiên **parent-child thật** ở Giai đoạn 2.
3. **Reranker gần như không đáng tiền ở cấu hình hiện tại.** Nó chỉ thêm +1,9
   điểm `evidence_recall` (hybrid 0.810 → 0.829) nhưng chậm gấp khoảng 8 lần. Nó
   giúp câu về nhân vật và bài học, nhưng làm giảm câu `fact`.
4. **BM25 không đóng góp gì** (hybrid 0.810 so với vector 0.816) với trọng số và
   cách tách từ hiện tại.
5. **Nhóm yếu nhất:** câu gõ không dấu (0.2–0.4), multi-hop, motif và
   multi-source (0.5–0.6). Với câu hỏi trải trên 3–4 truyện, 5 chunk nhỏ không đủ
   phủ.
6. **Chi phí LLM đo được:** 7 lời gọi mỗi câu hỏi (1 viết lại câu hỏi, 5 chấm
   tài liệu, 1 sinh câu trả lời).
7. **Lỗi âm thầm xuống cấp.** Khi reranker không nạp được (thiếu `sentencepiece`,
   hoặc GPU hết bộ nhớ vì ứng dụng khác), `RetrieverManager` chỉ ghi log rồi chạy
   tiếp mà không rerank. Giai đoạn 2 cần chuyển sang CPU hoặc báo lỗi rõ ràng.

### Baseline câu trả lời (commit `aca3c77`)

AdvancedRAG mặc định (hybrid + rerank, viết lại câu hỏi, chấm tài liệu bằng LLM)
qua `query_detailed`, chạy trên 95 câu. LLM là model đang gọi RAG (`claude-opus-5-5`).

| Chỉ số | Giá trị |
|--------|--------:|
| answer_recall (87 câu có đáp án) | 0.839 |
| faithfulness (model tự chấm) | 0.994 |
| abstention_accuracy (8 câu không có đáp án) | 1.000 |
| false_abstention_rate | 0.034 (3/87) |
| citation_rate | 1.000 |
| llm_calls_per_query | 7.0 |

| Nhóm | Số câu | answer_recall | false_abstention |
|------|------:|------:|------:|
| fact | 42 | 0.882 | 0.048 |
| sequence | 12 | 0.854 | 0 |
| creative | 9 | 0.787 | 0 |
| multi_source | 9 | 0.699 | 0 |
| character | 7 | 0.969 | 0 |
| motif | 7 | 0.726 | 0 |
| moral | 6 | 0.624 | 0 |
| no_diacritics | 5 | 0.657 | 0.200 |
| paraphrase | 4 | 0.879 | 0 |
| multi_hop | 2 | 0.633 | 0 |

**Đọc số liệu này thế nào:**

- **Mọi câu sai hay thiếu đều do ngữ cảnh**, không phải do sinh câu trả lời. Ba
  câu bị từ chối nhầm (q025, q037, q079) và các câu điểm thấp (q047, q036, q022)
  đều thiếu đoạn chứa đáp án trong ngữ cảnh. Khớp với nhận định 2: cần
  parent-child thật.
- **Có thể bị chệch.** Model trả lời và tự chấm faithfulness cũng là model đã
  viết golden set. Nó được dặn chỉ dùng ngữ cảnh, và đã từ chối cả những câu nó
  biết đáp án nhưng ngữ cảnh không có. Dù vậy, `faithfulness` nên xem là giới hạn
  trên. Khi so sánh về sau phải dùng cùng một model.
- **Độ trễ không gồm thời gian sinh câu trả lời**, vì câu trả lời được phát lại từ file.

## Giai đoạn 2: Sửa lõi retrieval ✅

Làm theo thứ tự số đo của Giai đoạn 1: parent-child trước, rồi câu gõ không dấu,
rồi chi phí reranker.

- [x] **Qdrant chạy được** qua `langchain-qdrant` (`QdrantVectorStore`) ở 3 chế độ:
      server (`QDRANT_URL`), nhúng trên đĩa (`PERSIST_DIRECTORY`), trong RAM
      (`url=":memory:"`). Id chuyển thành UUID ổn định; filter dạng dict dùng được
      cho cả tìm kiếm lẫn xoá.
- [x] **Qdrant là vector store mặc định** của API (`DEFAULT_VECTOR_STORE=qdrant`,
      `PERSIST_DIRECTORY=data/vector_store`); client Qdrant nằm trong phần lõi.
- [x] **FAISS** lưu đúng ids, ghi đè khi trùng id, xoá theo id/filter, tự nạp lại
      index đã lưu.
- [x] **Restart không mất index**: `AdvancedRAG` nạp lại chunk từ store khi khởi
      động (BM25, số tài liệu, refresh đều hoạt động). Có test cho FAISS và Qdrant.
- [x] **Cập nhật tăng dần thật**: refresh chỉ xoá/embed lại file thay đổi. Sửa lỗi
      manifest "không đổi" nhưng index trống thì không index lại.
- [x] **Parent-child thật**: `parent_id` dạng hash, chunk con mang theo đoạn parent;
      tìm trên chunk con, rerank và trả về parent. Bật mặc định
      (`ENABLE_PARENT_CONTEXT`).
- [x] **BM25**: áp filter metadata; thêm chỉ mục không dấu cho câu gõ không dấu.
- [x] **HyDE và multi-query** đi qua hybrid + filter, rerank theo câu hỏi gốc; RRF
      dùng khoá ổn định.
- [x] **Reranker** nạp một lần và dùng chung giữa các lần refresh, chấm theo batch.
      GPU hết bộ nhớ thì chuyển sang CPU; lỗi khác thì báo rõ và tắt rerank, không
      còn âm thầm dùng model tiếng Anh.
- [x] **docker-compose**: healthcheck của Qdrant gọi `curl`, nhưng image không có
      `curl`, nên `rag-api` không bao giờ khởi động được. Đã đổi sang kiểm tra cổng
      bằng bash. CI chạy test vector store với service `qdrant/qdrant`.
- [ ] Hybrid native của Qdrant (dense + sparse) để bỏ BM25 trong RAM: chưa cần.
      BM25 dựng lại từ store khi khởi động mất dưới 1 giây với corpus này. Xem lại
      ở Giai đoạn 5 nếu corpus lớn.

### Kết quả (commit `1992a93`, Qdrant, mặc định mới)

[`evals/fairy_tales/baselines/retrieval_stage2.json`](../evals/fairy_tales/baselines/retrieval_stage2.json),
so với baseline Giai đoạn 1 (`retrieval.json`, chunk con, FAISS):

| Cấu hình | evidence_recall | recall@5 | MRR | nDCG | context_chars | p50 |
|----------|------:|------:|------:|------:|------:|------:|
| hybrid + rerank (mặc định) | 0.829 → **0.941** | 0.965 → 0.987 | 0.952 → 0.974 | 0.944 → 0.963 | 926 → 1939 | 778 → 770 ms |
| hybrid | 0.810 → 0.909 | 0.980 → 0.979 | 0.972 → 0.987 | 0.959 → 0.973 | 890 → 1823 | 99 → 83 ms |
| vector + rerank | 0.817 → 0.918 | 0.965 → 0.987 | 0.946 → 0.974 | 0.940 → 0.963 | 929 → 1956 | 780 → 754 ms |
| vector | 0.816 → 0.907 | 0.976 → 0.976 | 0.966 → 0.966 | 0.954 → 0.954 | 894 → 1807 | 107 → 82 ms |

- **Ngữ cảnh gấp khoảng 2 lần** (~1900 ký tự, khoảng 600 token). Ở k=3 parent,
  ngữ cảnh chỉ tăng ~25% mà `evidence_recall` vẫn đạt 0.885, nên cải thiện không
  chỉ đến từ việc đưa thêm chữ.
- **Từng thay đổi được tách riêng:** Qdrant cho kết quả trùng khớp với FAISS. BM25
  không dấu chỉ đổi nhóm `no_diacritics` (hybrid 0.20 → 0.80), các nhóm khác giữ
  nguyên.
- **Giảm nhẹ:** `precision_at_k` giảm (0.77 → 0.72) vì các parent không trùng nhau
  trải trên nhiều truyện hơn; `hybrid` recall 0.980 → 0.979 (một câu thiếu một nguồn).
- **Còn yếu:** câu hỏi trải trên nhiều truyện (`motif` 0.63, `multi_source` 0.71
  với hybrid + rerank), để dành cho Giai đoạn 3/5.
- **Chưa kiểm chứng:** máy dev không có Docker, nên `docker compose up` chưa chạy
  thật. Chế độ server của Qdrant được test trong CI, nhưng CI chưa chạy vì nhánh
  chưa push được.
- **Bài học khi đo:** máy có 4 GB VRAM và paging file nhỏ (ổ C: gần đầy). Chạy hai
  tiến trình nặng cùng lúc làm câu hỏi lỗi vì thiếu bộ nhớ, và lỗi đó kéo điểm
  xuống. `scripts/eval.py` giờ cảnh báo khi có câu lỗi; không so sánh lần chạy
  có lỗi.

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

## Quyết định đã chốt
1. Mục đích dùng ưu tiên: **truyện cổ tích/sáng tác** (10/2026).
2. LLM đo đánh giá: **model đang gọi RAG** (AgentLLM), không dùng model khác (10/2026).
3. Vector store chính: **Qdrant** (10/2026).


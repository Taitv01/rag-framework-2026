# Bộ đánh giá truyện cổ tích

Bộ dữ liệu chuẩn dùng để đo mọi thay đổi của pipeline RAG trên trường hợp dùng
ưu tiên: truyện cổ tích và kịch bản sáng tác tiếng Việt.

| Thành phần | Nội dung |
|------------|----------|
| `corpus/` | 9 truyện cổ tích kể lại bằng lời mới và kịch bản *Sự tích Hồ Gươm* (10 tài liệu, khoảng 46 KB) |
| `golden.jsonl` | 95 câu hỏi: 87 câu có đáp án và 8 câu corpus không trả lời được |
| `baselines/` | Báo cáo baseline đã commit; mọi lần chạy sau đều so với các file này |
| `agent_answers.json` | Câu trả lời của model gọi RAG cho từng prompt (theo SHA-256), để chạy lại được |
| `runs/` | Các lần chạy thử (bị Git bỏ qua) |

Mỗi dòng của `golden.jsonl` có dạng:

```json
{"id": "q001", "question": "...", "answer": "...", "sources": ["thach_sanh"],
 "evidence": ["lưỡi búa của cha để lại"], "tags": ["fact"]}
```

- `sources`: tên file trong `corpus/` (không có đuôi). Danh sách rỗng nghĩa là
  câu hỏi không có đáp án trong corpus.
- `evidence`: các cụm từ trích nguyên văn từ nguồn. `tests/test_fairy_tale_golden_set.py`
  kiểm tra mọi cụm đều có thật trong corpus.
- `tags`: `fact`, `sequence`, `character`, `motif`, `moral`, `multi_source`,
  `multi_hop`, `paraphrase`, `no_diacritics`, `creative`, `unanswerable`.

## Chạy đánh giá

```powershell
# Retrieval, không cần LLM: vector, vector+rerank, hybrid, hybrid+rerank
py scripts/eval.py retrieval --cache-dir .cache/huggingface --device cpu `
    --baseline evals/fairy_tales/baselines/retrieval.json

# Câu trả lời end-to-end qua AdvancedRAG.query_detailed, do chính model AI đang
# chạy lệnh (Claude, Codex, ...) trả lời qua AgentLLM, không gọi API nào.
# Mỗi lần chạy ghi các prompt còn thiếu vào runs/agent_requests.json và thoát mã 3;
# model trả lời vào agent_answers.json rồi chạy lại, cho tới khi in báo cáo.
# Như pipeline mặc định, có thẻ tài liệu: mỗi tài liệu 1 prompt lúc index, cũng do
# model gọi RAG trả lời.
py scripts/eval.py answer --judge --device cpu --vector-store qdrant

# Không có thẻ tài liệu (cấu hình Giai đoạn 3)
py scripts/eval.py answer --judge --device cpu --vector-store qdrant --no-document-cards

# Hoặc gọi một API OpenAI-compatible (tốn credit)
py scripts/eval.py answer --llm api --base-url https://openrouter.ai/api/v1 `
    --llm-model openai/gpt-4o-mini --limit 20

# So sánh hai báo cáo bất kỳ
py scripts/eval.py compare evals/fairy_tales/baselines/retrieval.json evals/fairy_tales/runs/<file>.json

# Cổng chất lượng: thoát mã 1 nếu có chỉ số tụt quá ngưỡng (DEFAULT_GATE)
py scripts/eval.py compare <baseline>.json <báo cáo mới>.json --gate
```

### Cổng chất lượng trong CI

`.github/workflows/quality.yml` chạy trên mọi pull request đụng tới `src/`,
`evals/fairy_tales/`, `scripts/eval.py` hoặc `pyproject.toml`: đo retrieval cấu
hình `hybrid` (bge-m3 trên CPU, Qdrant trong RAM, không reranker, không LLM) rồi
so với [`baselines/retrieval_ci.json`](baselines/retrieval_ci.json) bằng `--gate`.
Recall, MRR, nDCG và `evidence_recall` được tụt tối đa 0,02; độ trễ và độ dài
ngữ cảnh không tính vì phụ thuộc máy và `k`.

Khi cố ý thay đổi chất lượng retrieval, tạo lại baseline bằng đúng lệnh của CI
và commit cùng thay đổi:

```powershell
py scripts/eval.py retrieval --configs hybrid --device cpu --vector-store qdrant `
    --cache-dir .cache/huggingface --out evals/fairy_tales/baselines/retrieval_ci.json
```

Lần chạy đầu sẽ tải `BAAI/bge-m3` và `AITeamVN/Vietnamese_Reranker` (khoảng 4,5 GB).
`--cache-dir` chuyển cache HuggingFace sang ổ còn trống. Reranker cần
`sentencepiece` (đã có trong extra `local-models`).

## Chỉ số

**Retrieval** (chỉ tính 87 câu có đáp án, cắt ở `k`):

- `recall_at_k`, `mrr`, `ndcg`, `precision_at_k`: tính theo truyện nguồn. Corpus
  chỉ có 10 truyện nên các chỉ số này dễ đạt mức cao.
- `evidence_recall`: tỉ lệ cụm bằng chứng nằm nguyên vẹn trong một chunk được
  truy xuất. Chỉ số này cho biết chunk lấy về có chứa đáp án hay không, nên
  phân biệt các cấu hình rõ hơn.
- `latency_p50_ms`, `latency_p95_ms`: phụ thuộc phần cứng, chỉ nên so trên cùng một máy.

**Câu trả lời**:

- `answer_recall`: tỉ lệ từ của đáp án chuẩn xuất hiện trong câu trả lời. Đây là
  chỉ số thay thế rẻ, không cần LLM.
- `faithfulness`: chấm bằng chính model của pipeline khi có `--judge`.
- `abstention_accuracy`: tỉ lệ câu không có đáp án mà pipeline trả lời "không có thông tin".
- `false_abstention_rate`: tỉ lệ câu có đáp án mà pipeline lại từ chối.
- `citation_rate`, `llm_calls_per_query`, token và độ trễ.

"""
Vietnamese-Optimized RAG
========================

Cấu hình tối ưu nhất cho tiếng Việt:
- Claude 200K context window
- Vietnamese embeddings (keepitreal/vietnamese-sbert)
- Vietnamese reranker (AITeamVN/Vietnamese_Reranker)
- Hybrid search (vector + BM25 Vietnamese)
- Contextual Retrieval (Anthropic pattern)
- Multi-query RRF
- Metadata Enhancement
- Hallucination Grader
- Context Window Validation

Usage:
    python examples/09_vietnamese_optimized.py
"""

import sys
import os
import io

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import AdvancedRAG


def create_vietnamese_rag():
    """
    Tạo AdvancedRAG với cấu hình tối ưu cho tiếng Việt.

    Returns:
        AdvancedRAG instance configured for Vietnamese
    """
    rag = AdvancedRAG(
        # ===== LLM: Claude 200K context =====
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",  # 200K tokens context
        # Hoặc dùng OpenAI nếu không có Anthropic key:
        # llm_provider="openai",
        # llm_model="gpt-4o",  # 128K tokens context

        # ===== Embeddings: Vietnamese SBERT =====
        embedding_provider="huggingface",
        embedding_model="keepitreal/vietnamese-sbert",  # 768d, tốt nhất cho tiếng Việt
        # Các lựa chọn khác:
        # "AITeamVN/Vietnamese_Embedding"       — 1024d, benchmarks tốt nhất
        # "bkai-foundation-models/vietnamese-bi-encoder" — 768d, PhoBERT-based
        # "BAAI/bge-m3"                         — 1024d, multilingual

        # ===== Vector Store =====
        vector_store_provider="faiss",  # hoặc "chroma" cho persistent

        # ===== Chunking: Contextual Retrieval =====
        chunk_size=1000,           # Chunk lớn hơn cho tiếng Việt (compound words)
        chunk_overlap=100,         # Overlap nhiều hơn để không mất ngữ cảnh
        use_contextual_chunking=True,  # Anthropic pattern — thêm context cho mỗi chunk

        # ===== Retrieval: Tối ưu =====
        retrieval_k=10,            # Retrieve nhiều chunk hơn
        use_hybrid=True,           # Vector + BM25 (Vietnamese tokenization)
        use_reranking=True,        # Vietnamese cross-encoder reranker
        use_multi_query_rrf=True,  # Multi-query + Reciprocal Rank Fusion
        num_query_variations=3,    # Số query variations

        # ===== Cache =====
        use_cache=True,            # Semantic cache
        cache_threshold=0.95,      # Similarity threshold
        cache_ttl=3600,            # 1 hour TTL

        # ===== Phase 3: Nâng cao =====
        use_metadata_enhancement=True,  # Tự gán metadata nhân vật/địa điểm/thời gian
        use_hallucination_check=True,   # Kiểm tra ảo giác
        # use_web_search=True,          # Bật nếu muốn web search fallback
        # web_search_provider="duckduckgo",
    )

    return rag


def demo_with_sample_data():
    """Demo với dữ liệu mẫu tiếng Việt."""

    print("=" * 60)
    print("🇻🇳 Vietnamese-Optimized RAG Demo")
    print("=" * 60)

    # Tạo RAG
    print("\n[1/5] Khởi tạo RAG với cấu hình tối ưu...")
    rag = create_vietnamese_rag()

    # Kiểm tra context window
    print("\n[2/5] Thông tin Context Window:")
    info = rag.context_info
    print(f"  Model:              {info.get('model', 'N/A')}")
    print(f"  Provider:           {info.get('provider', 'N/A')}")
    print(f"  Context Window:     {info.get('context_window', 'N/A'):,} tokens")
    print(f"  Available Tokens:   {info.get('available_tokens', 'N/A'):,} tokens")
    print(f"  Max Output Tokens:  {info.get('max_output_tokens', 'N/A'):,} tokens")
    print(f"  Retrieval K:        {info.get('retrieval_k', 'N/A')}")

    # Thêm dữ liệu mẫu
    print("\n[3/5] Thêm dữ liệu mẫu...")
    sample_texts = [
        """Thạch Sanh là nhân vật chính trong truyện cổ tích cùng tên của Việt Nam.
        Thạch Sanh mồ côi cha mẹ từ nhỏ, sống dưới gốc cây đa, làm nghề đốn củi.
        Thạch Sanh tính tình thật thà, chăm chỉ, hiếu thảo.""",

        """Lý Thông là bạn của Thạch Sanh, nhưng là người gian xảo, hay lừa đảo.
        Lý Thông đã lừa Thạch Sanh đi giết đại bàng để chiếm công.
        Sau này Lý Thông bị trừng phạt vì tội ác của mình.""",

        """Đại bàng là con quái vật hung dữ, sống trong hang sâu trên núi.
        Đại bàng bắt cóc công chúa và giam giữ trong hang.
        Thạch Sanh đã dùng cây đàn thần để đánh bại đại bàng và cứu công chúa.""",

        """Công chúa con vua bị đại bàng bắt cóc. Vua ra lệnh ai cứu được công chúa
        sẽ được gả công chúa và truyền ngôi. Thạch Sanh đã cứu công chúa và trở thành phò mã.""",

        """Cây đàn thần là báu vật mà Thạch Sanh được thần linh ban cho.
        Tiếng đàn có sức mạnh kỳ diệu, có thể xoa dịu nỗi đau và đánh bại kẻ thù.
        Thạch Sanh dùng cây đàn này để giết đại bàng và sau đó đánh tan quân xâm lược.""",
    ]

    sample_metadata = [
        {"source": "truyen_co_tich.txt", "characters": ["Thạch Sanh"], "topic": "Giới thiệu Thạch Sanh"},
        {"source": "truyen_co_tich.txt", "characters": ["Lý Thông"], "topic": "Giới thiệu Lý Thông"},
        {"source": "truyen_co_tich.txt", "characters": ["Đại bàng"], "topic": "Giới thiệu đại bàng"},
        {"source": "truyen_co_tich.txt", "characters": ["Công chúa", "Vua"], "topic": "Câu chuyện công chúa"},
        {"source": "truyen_co_tich.txt", "characters": ["Thạch Sanh"], "topic": "Cây đàn thần"},
    ]

    num_chunks = rag.add_texts(sample_texts, metadatas=sample_metadata)
    print(f"  Đã thêm {num_chunks} chunks")
    print(f"  Tổng documents: {rag.num_documents}")
    print(f"  Tổng chunks: {rag.num_chunks}")

    # Kiểm tra context usage sau khi thêm dữ liệu
    info = rag.context_info
    print(f"  Ước tính context tokens: {info.get('estimated_context_tokens', 'N/A'):,}")
    print(f"  Usage ratio: {info.get('usage_ratio', 'N/A'):.1%}")

    # Truy vấn
    print("\n[4/5] Truy vấn mẫu...")

    questions = [
        "Thạch Sanh là ai?",
        "Lý Thông đã làm gì với Thạch Sanh?",
        "Thạch Sanh đã đánh bại đại bàng như thế nào?",
        "Ai là công chúa trong câu chuyện?",
        "Cây đàn thần có sức mạnh gì?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n  Câu {i}: {question}")
        answer = rag.query(question)
        print(f"  Trả lời: {answer[:200]}...")

    # Kiểm tra cache
    print("\n[5/5] Kiểm tra Semantic Cache:")
    if rag.cache_stats:
        print(f"  Cache size: {rag.cache_stats.get('size', 0)}")
        print(f"  Threshold: {rag.cache_stats.get('threshold', 'N/A')}")
    else:
        print("  Cache chưa được sử dụng")

    print("\n" + "=" * 60)
    print("✅ Demo hoàn tất!")
    print("=" * 60)


def demo_without_api():
    """Demo không cần API key — chỉ kiểm tra cấu hình."""

    print("=" * 60)
    print("🇻🇳 Vietnamese-Optimized RAG — Configuration Check")
    print("=" * 60)

    from src.core.llm import LLMManager
    from src.core.embeddings import EmbeddingsManager
    from src.utils.context_validator import ContextValidator

    # Kiểm tra LLM models
    print("\n📊 LLM Models & Context Windows:")
    for provider, models in LLMManager.POPULAR_MODELS.items():
        for model_name, model_info in models.items():
            ctx = model_info.get("context_window", "N/A")
            print(f"  {provider}/{model_name}: {ctx:,} tokens")

    # Kiểm tra Embedding models
    print("\n🔢 Vietnamese Embedding Models:")
    for model_name, model_info in EmbeddingsManager.POPULAR_MODELS.get("huggingface", {}).items():
        dims = model_info.get("dimensions", "N/A")
        desc = model_info.get("description", "")
        print(f"  {model_name}: {dims}d — {desc}")

    # Kiểm tra Context Validator
    print("\n📏 Context Window Validation:")
    for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"]:
        if "gpt" in model:
            provider = "openai"
        else:
            provider = "anthropic"
        llm = LLMManager(provider=provider, model=model)
        validator = ContextValidator.from_llm_manager(llm)
        print(f"  {model}:")
        print(f"    Context Window: {validator.context_window:,} tokens")
        print(f"    Available:      {validator.available_tokens:,} tokens")
        print(f"    Output Reserve: {validator.reserve_tokens:,} tokens")

    # Mô phỏng context size
    print("\n📐 Ước tính Context Size:")
    configs = [
        ("Default (500 chars × 5)", 500, 5),
        ("Optimized (1000 chars × 10)", 1000, 10),
        ("Large (2000 chars × 20)", 2000, 20),
    ]
    for name, chunk_size, k in configs:
        total_chars = chunk_size * k
        # Vietnamese: ~3 chars per token
        est_tokens = total_chars // 3
        print(f"  {name}:")
        print(f"    Chars: {total_chars:,}")
        print(f"    ~Tokens (Vietnamese): {est_tokens:,}")
        print(f"    % of Claude 200K: {est_tokens/200000:.1%}")
        print(f"    % of GPT-4o 128K: {est_tokens/128000:.1%}")

    print("\n" + "=" * 60)
    print("✅ Configuration check hoàn tất!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vietnamese-Optimized RAG Demo")
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Run without API key (configuration check only)",
    )
    args = parser.parse_args()

    if args.no_api:
        demo_without_api()
    else:
        demo_with_sample_data()

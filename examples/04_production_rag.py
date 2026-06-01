"""
Example 04: Production RAG
==========================

Production-ready RAG with caching, error handling, and logging.

This example demonstrates:
1. Configuration management
2. Caching for performance
3. Error handling with retry
4. Logging and monitoring
5. Evaluation

Requirements:
    pip install -r requirements.txt

Usage:
    python examples/04_production_rag.py
"""

import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import AdvancedRAG
from src.utils.config import Config
from src.utils.cache import QueryCache
from src.utils.logger import setup_logger
from src.evaluation import RAGEvaluator


def main():
    """Run Production RAG example."""

    # =========================================================================
    # Step 1: Setup Configuration
    # =========================================================================
    print("=" * 60)
    print("Production RAG Example")
    print("=" * 60)

    # Load configuration
    config = Config()

    # Setup logger
    logger = setup_logger("production_rag", level="INFO")
    logger.info("Initializing Production RAG...")

    # =========================================================================
    # Step 2: Initialize RAG with Config
    # =========================================================================
    rag_config = config.get_rag_config()
    llm_config = config.get_llm_config()

    rag = AdvancedRAG(
        llm_provider=llm_config["provider"],
        llm_model=llm_config["model"],
        embedding_provider=config.get("DEFAULT_EMBEDDING_PROVIDER"),
        vector_store_provider=config.get("DEFAULT_VECTOR_STORE"),
        chunk_size=rag_config["chunk_size"],
        chunk_overlap=rag_config["chunk_overlap"],
        retrieval_k=rag_config["retrieval_k"],
        use_hybrid=rag_config["enable_hybrid_search"],
        use_reranking=rag_config["enable_reranking"],
    )

    logger.info(f"RAG initialized with {llm_config['provider']}/{llm_config['model']}")

    # =========================================================================
    # Step 3: Setup Cache
    # =========================================================================
    cache_ttl = config.get_int("CACHE_TTL", default=3600)
    query_cache = QueryCache(ttl=cache_ttl)

    logger.info(f"Cache enabled with TTL: {cache_ttl}s")

    # =========================================================================
    # Step 4: Add Documents
    # =========================================================================
    print("\n📚 Adding documents...")

    docs = [
        """Artificial Intelligence (AI) is the simulation of human intelligence
        by computer systems. AI can be categorized into narrow AI (designed for
        specific tasks) and general AI (capable of any intellectual task).""",

        """Machine Learning is a subset of AI that enables systems to learn from
        data. Common types include supervised learning, unsupervised learning,
        and reinforcement learning.""",

        """Deep Learning uses neural networks with multiple layers to learn
        representations of data. It has achieved breakthrough results in
        computer vision, NLP, and speech recognition.""",

        """Natural Language Processing (NLP) enables computers to understand and
        generate human language. Applications include chatbots, translation,
        sentiment analysis, and text summarization.""",
    ]

    num_chunks = rag.add_texts(docs)
    logger.info(f"Added {num_chunks} chunks")

    # =========================================================================
    # Step 5: Query with Caching
    # =========================================================================
    print("\n🔍 Querying with caching...")

    def cached_query(question: str) -> dict:
        """Query with caching."""
        # Check cache
        cached = query_cache.get(question)
        if cached:
            logger.info(f"Cache hit for: {question}")
            return cached

        # Query RAG
        logger.info(f"Cache miss for: {question}")
        result = rag.query_detailed(question)

        # Cache result
        query_cache.set(question, result)

        return result

    # First query (cache miss)
    result1 = cached_query("What is AI?")
    print(f"\n💡 Answer: {result1['answer']}")

    # Same query (cache hit)
    result2 = cached_query("What is AI?")
    print(f"\n💡 Answer (cached): {result2['answer']}")

    # =========================================================================
    # Step 6: Error Handling with Retry
    # =========================================================================
    print("\n\n🔄 Error handling with retry...")

    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def query_with_retry(question: str) -> str:
        """Query with retry logic."""
        try:
            result = rag.query(question)
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    # Example query
    answer = query_with_retry("What is machine learning?")
    print(f"💡 Answer: {answer}")

    # =========================================================================
    # Step 7: Evaluation
    # =========================================================================
    print("\n\n📊 Evaluation...")

    evaluator = RAGEvaluator(rag.llm)

    # Test data
    test_data = [
        {
            "question": "What is AI?",
            "expected": "AI is the simulation of human intelligence by computers",
        },
        {
            "question": "What is machine learning?",
            "expected": "Machine learning is a subset of AI that learns from data",
        },
    ]

    # Evaluate
    def query_func(question):
        result = rag.query_detailed(question)
        return {
            "answer": result["answer"],
            "contexts": [s["content"] for s in result["relevant_docs"]],
        }

    report = evaluator.evaluate(query_func, test_data)

    # Print report
    evaluator.print_report(report)

    # Export report
    report_path = Path("evaluation_report.json")
    evaluator.export_report(report, str(report_path))
    logger.info(f"Report exported to {report_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Production Features")
    print("=" * 60)
    print("✅ Configuration management")
    print("✅ Caching (TTL: {}s)".format(cache_ttl))
    print("✅ Error handling with retry")
    print("✅ Logging")
    print("✅ Evaluation framework")
    print("=" * 60)


if __name__ == "__main__":
    main()

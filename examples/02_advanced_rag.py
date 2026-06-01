"""
Example 02: Advanced RAG
========================

Advanced RAG with hybrid search and re-ranking.

This example demonstrates:
1. Hybrid search (vector + BM25)
2. Cross-encoder re-ranking
3. Query transformation
4. Document grading

Requirements:
    pip install -r requirements.txt

Usage:
    python examples/02_advanced_rag.py
"""

import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import AdvancedRAG


def main():
    """Run Advanced RAG example."""

    # =========================================================================
    # Step 1: Initialize Advanced RAG
    # =========================================================================
    print("=" * 60)
    print("Advanced RAG Example")
    print("=" * 60)

    rag = AdvancedRAG(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        embedding_provider="huggingface",
        vector_store_provider="faiss",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=5,
        use_hybrid=True,      # Enable hybrid search
        use_reranking=True,    # Enable re-ranking
    )

    # =========================================================================
    # Step 2: Add Documents
    # =========================================================================
    print("\n📚 Adding documents...")

    # Technical documentation samples
    docs = [
        """Machine Learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without being explicitly
        programmed. It focuses on developing computer programs that can access
        data and use it to learn for themselves.""",

        """Deep Learning is a subset of machine learning that uses neural networks
        with multiple layers (hence "deep") to analyze various factors of data.
        It is particularly effective for tasks like image recognition, natural
        language processing, and speech recognition.""",

        """Natural Language Processing (NLP) is a branch of artificial intelligence
        that helps computers understand, interpret and manipulate human language.
        NLP draws from many disciplines, including computer science and
        computational linguistics, in its pursuit to fill the gap between
        human communication and computer understanding.""",

        """Computer Vision is a field of artificial intelligence that trains
        computers to interpret and understand the visual world. Using digital
        images from cameras and videos and deep learning models, machines can
        accurately identify and classify objects.""",

        """Reinforcement Learning is an area of machine learning concerned with
        how software agents ought to take actions in an environment so as to
        maximize some notion of cumulative reward. It is employed in various
        applications including robotics, game playing, and resource management.""",

        """Transfer Learning is a machine learning method where a model developed
        for one task is reused as the starting point for a model on a second task.
        It is popular in deep learning because it allows building accurate models
        in a time-saving way.""",

        """Generative Adversarial Networks (GANs) are a class of machine learning
        frameworks designed by Ian Goodfellow in 2014. Two neural networks contest
        with each other in a game, leading to the generation of synthetic data
        that resembles the training data.""",

        """Transformer models have revolutionized natural language processing.
        Introduced in the paper 'Attention Is All You Need', transformers use
        self-attention mechanisms to process input data in parallel, making them
        more efficient than previous architectures like RNNs and LSTMs.""",
    ]

    num_chunks = rag.add_texts(docs)
    print(f"✅ Added {num_chunks} chunks")

    # =========================================================================
    # Step 3: Standard Query (with all advanced features)
    # =========================================================================
    print("\n🔍 Querying with advanced features...")

    questions = [
        "What is the difference between machine learning and deep learning?",
        "How do transformers work in NLP?",
        "What are GANs used for?",
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)

        # Query with detailed results
        result = rag.query_detailed(question)

        print(f"🔄 Transformed query: {result['transformed_query']}")
        print(f"📄 Retrieved {result['total_docs_retrieved']} docs, "
              f"{result['relevant_docs_count']} relevant")
        print(f"\n💡 Answer: {result['answer']}")

    # =========================================================================
    # Step 4: Compare Retrieval Strategies
    # =========================================================================
    print("\n\n📊 Comparing retrieval strategies...")

    query = "What is deep learning?"

    # Standard similarity search
    print(f"\n🔍 Query: {query}")
    print("\n1️⃣ Similarity Search:")
    docs_similarity = rag.retrieve(query, use_hybrid=False, use_reranking=False)
    for i, doc in enumerate(docs_similarity[:2], 1):
        print(f"   [{i}] {doc.page_content[:100]}...")

    # Hybrid search
    print("\n2️⃣ Hybrid Search (vector + BM25):")
    docs_hybrid = rag.retrieve(query, use_hybrid=True, use_reranking=False)
    for i, doc in enumerate(docs_hybrid[:2], 1):
        print(f"   [{i}] {doc.page_content[:100]}...")

    # With re-ranking
    print("\n3️⃣ With Re-ranking:")
    docs_reranked = rag.retrieve(query, use_hybrid=False, use_reranking=True)
    for i, doc in enumerate(docs_reranked[:2], 1):
        print(f"   [{i}] {doc.page_content[:100]}...")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Documents loaded: {rag.num_documents}")
    print(f"Total chunks: {rag.num_chunks}")
    print(f"Hybrid search: ✅ Enabled")
    print(f"Re-ranking: ✅ Enabled")
    print("=" * 60)


if __name__ == "__main__":
    main()

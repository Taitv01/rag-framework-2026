"""
Example 01: Naive RAG
====================

Basic RAG implementation using vector search + LLM generation.

This example demonstrates:
1. Loading documents
2. Creating embeddings
3. Storing in vector database
4. Querying with LLM

Requirements:
    pip install -r requirements.txt

Usage:
    python examples/01_naive_rag.py
"""

import os
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import NaiveRAG


def main():
    """Run Naive RAG example."""

    # =========================================================================
    # Step 1: Initialize RAG
    # =========================================================================
    print("=" * 60)
    print("Naive RAG Example")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  No OPENAI_API_KEY found.")
        print("Using local HuggingFace embeddings (no API key needed).")
        print("For LLM, set OPENAI_API_KEY in .env file.\n")

        # Use local embeddings only
        rag = NaiveRAG(
            llm_provider="openai",  # Will use env variable
            embedding_provider="huggingface",
            vector_store_provider="faiss",
            chunk_size=500,
            chunk_overlap=50,
        )
    else:
        rag = NaiveRAG(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            embedding_provider="huggingface",
            vector_store_provider="faiss",
            chunk_size=500,
            chunk_overlap=50,
        )

    # =========================================================================
    # Step 2: Add Sample Documents
    # =========================================================================
    print("\n📚 Adding sample documents...")

    # Sample documents about Python
    sample_texts = [
        """Python is a high-level, general-purpose programming language. Its design
        philosophy emphasizes code readability with the use of significant indentation.
        Python is dynamically typed and garbage-collected. It supports multiple
        programming paradigms, including structured (particularly procedural),
        object-oriented and functional programming.""",

        """Python was created by Guido van Rossum and was first released in 1991.
        The language has evolved significantly over the years, with Python 3.0 being
        released in 2008. Python 3 introduced many backward-incompatible changes,
        including a new print function syntax.""",

        """Python has a comprehensive standard library and a large ecosystem of
        third-party packages. The Python Package Index (PyPI) hosts over 400,000
        packages for various tasks including web development, data science, machine
        learning, and automation.""",

        """Python is widely used in data science and machine learning due to libraries
        like NumPy, Pandas, Scikit-learn, TensorFlow, and PyTorch. Its simple syntax
        makes it accessible to beginners while being powerful enough for experts.""",

        """Virtual environments in Python allow you to create isolated environments
        for different projects. Tools like venv, virtualenv, and conda help manage
        dependencies and avoid conflicts between projects.""",
    ]

    # Add texts
    num_chunks = rag.add_texts(sample_texts)
    print(f"✅ Added {num_chunks} chunks from {len(sample_texts)} documents")

    # =========================================================================
    # Step 3: Query the Knowledge Base
    # =========================================================================
    print("\n🔍 Querying knowledge base...")

    # Example queries
    questions = [
        "What is Python?",
        "Who created Python?",
        "What is Python used for?",
        "How to manage Python dependencies?",
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)

        # Query with sources
        result = rag.query_with_sources(question)

        print(f"💡 Answer: {result['answer']}")
        print(f"\n📄 Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source['content'][:100]}...")

    # =========================================================================
    # Step 4: Retrieve Documents Only
    # =========================================================================
    print("\n\n📚 Retrieving documents without LLM...")

    query = "machine learning libraries"
    docs = rag.retrieve(query, k=3)

    print(f"\n🔍 Query: {query}")
    print(f"📄 Retrieved {len(docs)} documents:")

    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] {doc.page_content[:150]}...")
        print(f"      Metadata: {doc.metadata}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Documents loaded: {rag.num_documents}")
    print(f"Total chunks: {rag.num_chunks}")
    print(f"Vector store: FAISS (in-memory)")
    print(f"Embeddings: HuggingFace (local)")
    print("=" * 60)


if __name__ == "__main__":
    main()

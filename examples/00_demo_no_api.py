"""
Example 00: Demo (No API Key Required)
======================================

This demo shows the RAG framework capabilities without requiring any API keys.

It demonstrates:
1. Document loading
2. Text splitting
3. Embeddings generation (local)
4. Vector store operations
5. Similarity search

Usage:
    python examples/00_demo_no_api.py
"""

import sys
import io
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run demo without API keys."""

    print("=" * 60)
    print("🚀 Ultimate RAG Framework - Demo (No API Key Required)")
    print("=" * 60)

    # =========================================================================
    # 1. Document Loading
    # =========================================================================
    print("\n📚 1. Document Loading")
    print("-" * 40)

    from src.core.document_loader import DocumentLoader

    loader = DocumentLoader()

    # Create sample documents
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content="""Python is a high-level, general-purpose programming language.
            Its design philosophy emphasizes code readability with the use of significant
            indentation. Python is dynamically typed and garbage-collected.""",
            metadata={"source": "python_intro.txt", "topic": "programming"}
        ),
        Document(
            page_content="""Machine learning is a subset of artificial intelligence that
            enables systems to learn and improve from experience without being explicitly
            programmed. It focuses on developing computer programs that can access data
            and use it to learn for themselves.""",
            metadata={"source": "ml_basics.txt", "topic": "ai"}
        ),
        Document(
            page_content="""Deep learning is a subset of machine learning that uses neural
            networks with multiple layers to analyze various factors of data. It is
            particularly effective for tasks like image recognition and NLP.""",
            metadata={"source": "deep_learning.txt", "topic": "ai"}
        ),
        Document(
            page_content="""Natural Language Processing (NLP) is a branch of artificial
            intelligence that helps computers understand, interpret and manipulate human
            language. NLP draws from many disciplines including computer science.""",
            metadata={"source": "nlp_intro.txt", "topic": "ai"}
        ),
        Document(
            page_content="""Python has a comprehensive standard library and a large
            ecosystem of third-party packages. The Python Package Index (PyPI) hosts
            over 400,000 packages for various tasks including web development and
            data science.""",
            metadata={"source": "python_ecosystem.txt", "topic": "programming"}
        ),
    ]

    print(f"✅ Loaded {len(documents)} documents")
    for doc in documents:
        print(f"   - {doc.metadata['source']}: {doc.page_content[:50]}...")

    # =========================================================================
    # 2. Text Splitting
    # =========================================================================
    print("\n\n✂️ 2. Text Splitting")
    print("-" * 40)

    from src.core.text_splitter import TextSplitter

    splitter = TextSplitter(chunk_size=200, chunk_overlap=50)

    # Split documents
    chunks = splitter.split_documents(documents)

    print(f"✅ Split into {len(chunks)} chunks")
    print(f"   Original documents: {len(documents)}")
    print(f"   Chunk size: 200 characters")
    print(f"   Chunk overlap: 50 characters")

    # Show sample chunks
    print("\n📄 Sample chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n   [{i}] {chunk.page_content[:100]}...")
        print(f"       Source: {chunk.metadata.get('source', 'unknown')}")

    # =========================================================================
    # 3. Embeddings (Local)
    # =========================================================================
    print("\n\n🔢 3. Embeddings (Local HuggingFace)")
    print("-" * 40)

    from src.core.embeddings import EmbeddingsManager

    embeddings = EmbeddingsManager(
        provider="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("✅ Embedding model loaded")
    print(f"   Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   Dimensions: 384")

    # Generate embeddings for sample text
    sample_texts = ["What is Python?", "Machine learning basics"]
    vectors = embeddings.embed_documents(sample_texts)

    print(f"\n📊 Generated embeddings:")
    print(f"   Texts: {len(sample_texts)}")
    print(f"   Vector dimensions: {len(vectors[0])}")
    print(f"   Sample vector (first 5 values): {vectors[0][:5]}")

    # =========================================================================
    # 4. Vector Store
    # =========================================================================
    print("\n\n💾 4. Vector Store (FAISS)")
    print("-" * 40)

    from src.core.vector_store import VectorStoreManager

    vector_store = VectorStoreManager(
        provider="faiss",
        embeddings=embeddings
    )

    # Add documents
    vector_store.add_documents(chunks)

    print("✅ Vector store created")
    print(f"   Provider: FAISS (in-memory)")
    print(f"   Documents indexed: {len(chunks)}")

    # =========================================================================
    # 5. Similarity Search
    # =========================================================================
    print("\n\n🔍 5. Similarity Search")
    print("-" * 40)

    # Search queries
    queries = [
        "What is Python?",
        "machine learning",
        "deep learning neural networks",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = vector_store.similarity_search(query, k=2)

        for i, doc in enumerate(results, 1):
            print(f"   [{i}] {doc.page_content[:80]}...")
            print(f"       Score: {doc.metadata.get('score', 'N/A')}")

    # =========================================================================
    # 6. Retrieval Manager
    # =========================================================================
    print("\n\n🎯 6. Retrieval Manager")
    print("-" * 40)

    from src.core.retriever import RetrieverManager

    retriever = RetrieverManager(
        vector_store=vector_store,
        embeddings=embeddings,
        documents=chunks,
        k=3,
        use_hybrid=True,
    )

    print("✅ Retriever initialized")
    print(f"   Search modes: Similarity, Hybrid, MMR")

    # Test different search modes
    query = "artificial intelligence"

    # Similarity search
    print(f"\n🔍 Query: '{query}'")
    print("\n   Similarity Search:")
    docs = retriever.search(query, k=2)
    for i, doc in enumerate(docs, 1):
        print(f"      [{i}] {doc.page_content[:60]}...")

    # =========================================================================
    # 7. Caching
    # =========================================================================
    print("\n\n⚡ 7. Caching")
    print("-" * 40)

    from src.utils.cache import Cache, QueryCache

    # Basic cache
    cache = Cache(max_size=100, ttl=3600)

    cache.set("key1", "value1")
    cache.set("key2", {"data": "value2"})

    print("✅ Cache created")
    print(f"   key1: {cache.get('key1')}")
    print(f"   key2: {cache.get('key2')}")
    print(f"   Size: {cache.size()}")

    # Query cache
    query_cache = QueryCache(ttl=3600)
    query_cache.set("What is Python?", {"answer": "Python is a programming language"})

    print(f"\n   Query cache:")
    print(f"   'What is Python?': {query_cache.get('What is Python?')}")

    # =========================================================================
    # 8. Configuration
    # =========================================================================
    print("\n\n⚙️ 8. Configuration")
    print("-" * 40)

    from src.utils.config import Config

    config = Config()

    print("✅ Configuration loaded")
    print(f"   Default LLM: {config.get('DEFAULT_LLM_MODEL', 'gpt-4o-mini')}")
    print(f"   Chunk size: {config.get_int('CHUNK_SIZE', default=500)}")
    print(f"   Hybrid search: {config.get_bool('ENABLE_HYBRID_SEARCH', default=True)}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("=" * 60)
    print("\n✅ What we demonstrated:")
    print("   1. Document loading")
    print("   2. Text splitting")
    print("   3. Local embeddings (no API key)")
    print("   4. Vector store (FAISS)")
    print("   5. Similarity search")
    print("   6. Retrieval manager")
    print("   7. Caching")
    print("   8. Configuration")
    print("\n📚 Next steps:")
    print("   - Set OPENAI_API_KEY in .env for full RAG capabilities")
    print("   - Try examples/01_naive_rag.py for complete RAG pipeline")
    print("   - Check docs/ for detailed documentation")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Example 05: Graph RAG
=====================

Knowledge Graph-based RAG for structured reasoning.

This example demonstrates:
1. Knowledge graph construction
2. Entity and relationship extraction
3. Graph-based retrieval
4. Hybrid retrieval (graph + vector)

Requirements:
    pip install -r requirements.txt

Usage:
    python examples/05_graph_rag.py
"""

import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run Graph RAG example."""

    print("=" * 60)
    print("Graph RAG Example")
    print("=" * 60)

    from src.rag import GraphRAG

    # Initialize Graph RAG
    rag = GraphRAG(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        embedding_provider="huggingface",
        chunk_size=1000,
        chunk_overlap=100,
    )

    # Add documents
    print("\n📚 Adding documents and extracting knowledge graph...")

    docs = [
        """Python is a high-level programming language created by Guido van Rossum.
        Python is widely used in artificial intelligence and machine learning.
        Major libraries include NumPy, Pandas, and TensorFlow.""",

        """TensorFlow is an open-source machine learning framework developed by Google.
        It is used for building and training neural networks.
        TensorFlow works well with Python and supports GPU acceleration.""",

        """Machine learning is a subset of artificial intelligence.
        It enables systems to learn from data without explicit programming.
        Common types include supervised, unsupervised, and reinforcement learning.""",

        """Deep learning uses neural networks with multiple layers.
        It is particularly effective for image recognition and natural language processing.
        Popular frameworks include TensorFlow, PyTorch, and Keras.""",
    ]

    num_chunks = rag.add_texts(docs)
    print(f"✅ Added {num_chunks} chunks")

    # Show knowledge graph statistics
    print(f"\n📊 Knowledge Graph Statistics:")
    print(f"   Entities: {rag.num_entities}")
    print(f"   Relationships: {rag.num_relationships}")

    # Get knowledge graph
    kg = rag.get_knowledge_graph()

    # Show entities
    print(f"\n📌 Entities:")
    for entity in kg.get_all_entities()[:10]:
        print(f"   - {entity.name} ({entity.entity_type}): {entity.description[:50]}...")

    # Show relationships
    print(f"\n🔗 Relationships:")
    for rel in kg.get_all_relationships()[:10]:
        print(f"   - {rel.source} -> {rel.target}: {rel.relationship_type}")

    # Query with graph reasoning
    print(f"\n🔍 Querying with graph reasoning...")

    questions = [
        "What is the relationship between Python and TensorFlow?",
        "How does machine learning relate to deep learning?",
        "What libraries are used for AI in Python?",
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)

        answer = rag.query(question)
        print(f"💡 Answer: {answer}")

    # Save knowledge graph
    graph_path = "knowledge_graph.json"
    rag.save_graph(graph_path)
    print(f"\n💾 Knowledge graph saved to {graph_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Documents: {rag.num_documents}")
    print(f"Chunks: {rag.num_chunks}")
    print(f"Entities: {rag.num_entities}")
    print(f"Relationships: {rag.num_relationships}")
    print("=" * 60)


if __name__ == "__main__":
    main()

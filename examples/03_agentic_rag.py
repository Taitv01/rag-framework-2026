"""
Example 03: Agentic RAG
=======================

Agent-based RAG using LangGraph.

This example demonstrates:
1. LLM deciding whether to retrieve
2. Document relevance grading
3. Query rewriting loop
4. Multi-step reasoning

Requirements:
    pip install -r requirements.txt

Usage:
    python examples/03_agentic_rag.py
"""

import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import AgenticRAG


def main():
    """Run Agentic RAG example."""

    # =========================================================================
    # Step 1: Initialize Agentic RAG
    # =========================================================================
    print("=" * 60)
    print("Agentic RAG Example")
    print("=" * 60)

    rag = AgenticRAG(
        llm_provider="openai",
        llm_model="gpt-4o",
        embedding_provider="huggingface",
        vector_store_provider="faiss",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=4,
        max_retries=3,
    )

    # =========================================================================
    # Step 2: Add Documents
    # =========================================================================
    print("\n📚 Adding documents...")

    # Knowledge base about programming
    docs = [
        """Python is a high-level, interpreted programming language known for
        its simplicity and readability. Created by Guido van Rossum in 1991,
        Python has become one of the most popular programming languages in the
        world, used in web development, data science, AI, and automation.""",

        """JavaScript is the programming language of the web. Originally created
        for client-side scripting, it has evolved into a versatile language used
        for both frontend and backend development (Node.js). JavaScript supports
        multiple programming paradigms including event-driven and functional.""",

        """Rust is a systems programming language focused on safety, speed, and
        concurrency. Developed by Mozilla, Rust achieves memory safety without
        garbage collection through its ownership system. It's ideal for systems
        programming, embedded systems, and performance-critical applications.""",

        """Go (Golang) is a statically typed, compiled programming language
        designed at Google. Known for its simplicity, efficiency, and built-in
        concurrency support, Go is widely used for cloud services, microservices,
        and distributed systems.""",

        """TypeScript is a superset of JavaScript that adds static typing.
        Developed by Microsoft, TypeScript helps catch errors at compile time
        and improves code maintainability for large-scale applications. It has
        become the standard for enterprise JavaScript development.""",

        """Kotlin is a modern programming language that runs on the JVM. It is
        fully interoperable with Java and has become the preferred language for
        Android development. Kotlin offers null safety, coroutines, and
        functional programming features.""",
    ]

    num_chunks = rag.add_texts(docs)
    print(f"✅ Added {num_chunks} chunks")

    # =========================================================================
    # Step 3: Query with Agent
    # =========================================================================
    print("\n🤖 Querying with agent...")

    # The agent will decide whether to retrieve based on the question
    questions = [
        "What is Python?",  # Factual - should retrieve
        "Tell me a joke",   # Non-factual - may not retrieve
        "Compare Python and JavaScript",  # Comparison - should retrieve
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)

        try:
            # Query with trace
            result = rag.query_with_trace(question)

            print(f"💡 Answer: {result['answer']}")

            # Show execution trace
            print(f"\n📍 Execution trace:")
            for step in result['trace']:
                print(f"   → {step['node']}")

        except Exception as e:
            print(f"⚠️ Error: {e}")

    # =========================================================================
    # Step 4: Multi-turn Conversation
    # =========================================================================
    print("\n\n💬 Multi-turn conversation...")

    conversation = [
        {"role": "user", "content": "What is Python?"},
    ]

    # First turn
    print("\n👤 User: What is Python?")
    answer1 = rag.query(
        "What is Python?",
        conversation_history=conversation
    )
    print(f"🤖 Assistant: {answer1}")

    # Add to history
    conversation.append({"role": "assistant", "content": answer1})

    # Second turn (follow-up)
    print("\n👤 User: What are its main uses?")
    answer2 = rag.query(
        "What are its main uses?",
        conversation_history=conversation
    )
    print(f"🤖 Assistant: {answer2}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Documents loaded: {rag.num_documents}")
    print(f"Total chunks: {rag.num_chunks}")
    print(f"Agent features:")
    print(f"  - Smart retrieval decisions")
    print(f"  - Document relevance grading")
    print(f"  - Query rewriting")
    print(f"  - Multi-turn conversation")
    print("=" * 60)


if __name__ == "__main__":
    main()

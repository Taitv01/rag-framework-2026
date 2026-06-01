"""
Example 07: Web UI
==================

Start a Gradio Web UI for the RAG system.

This example demonstrates:
1. Starting a Gradio interface
2. Chat interface
3. Document upload
4. Search interface
5. Statistics dashboard

Requirements:
    pip install -r requirements.txt

Usage:
    python examples/07_web_ui.py

    Then visit: http://localhost:7860
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
    """Start Web UI."""

    print("=" * 60)
    print("RAG Web UI")
    print("=" * 60)

    try:
        import gradio as gr
    except ImportError:
        print("Error: gradio is required. Install with: pip install gradio")
        return

    from src.ui import create_ui
    from src.utils.config import Config

    config = Config()

    print("\n🚀 Starting Web UI...")
    print("\n📌 Features:")
    print("   - 💬 Chat with your knowledge base")
    print("   - 📄 Upload and index documents")
    print("   - 🔍 Search documents")
    print("   - 📊 View statistics")
    print("\n🌐 URL: http://localhost:7860")
    print("\n" + "=" * 60)

    # Create and launch UI
    ui = create_ui(
        rag_type="advanced",
        llm_provider=config.get("DEFAULT_LLM_PROVIDER", "openai"),
        llm_model=config.get("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
        embedding_provider=config.get("DEFAULT_EMBEDDING_PROVIDER", "huggingface"),
    )

    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()

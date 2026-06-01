"""
Web UI
======

Gradio-based Web UI for RAG system.

Features:
- Chat interface
- Document upload
- Knowledge graph visualization
- Search interface
- Settings panel

Usage:
    python -m src.ui.app
"""

import io
import sys
from typing import List, Optional, Tuple
from pathlib import Path

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import gradio as gr

from src.rag import NaiveRAG, AdvancedRAG
from src.utils.config import Config


def create_ui(
    rag_type: str = "advanced",
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    embedding_provider: str = "huggingface",
    **kwargs
) -> gr.Blocks:
    """
    Create Gradio UI.

    Args:
        rag_type: Type of RAG ('naive' or 'advanced')
        llm_provider: LLM provider
        llm_model: LLM model name
        embedding_provider: Embedding provider
        **kwargs: Additional arguments

    Returns:
        Gradio Blocks interface
    """
    # Initialize RAG
    if rag_type == "advanced":
        rag = AdvancedRAG(
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_provider=embedding_provider,
            **kwargs
        )
    else:
        rag = NaiveRAG(
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_provider=embedding_provider,
            **kwargs
        )

    # ========================================================================
    # Helper Functions
    # ========================================================================

    def chat(message: str, history: List[Tuple[str, str]]) -> str:
        """Chat with RAG system."""
        try:
            # Build conversation history
            conversation = []
            for user_msg, assistant_msg in history:
                conversation.append({"role": "user", "content": user_msg})
                conversation.append({"role": "assistant", "content": assistant_msg})

            # Query
            if rag_type == "advanced":
                result = rag.query_detailed(message)
                answer = result["answer"]
            else:
                answer = rag.query(message)

            return answer
        except Exception as e:
            return f"Error: {str(e)}"

    def upload_files(files) -> str:
        """Upload and index files."""
        try:
            file_paths = [f.name for f in files]
            num_chunks = rag.add_documents(file_paths)
            return f"Successfully indexed {len(files)} files ({num_chunks} chunks)"
        except Exception as e:
            return f"Error: {str(e)}"

    def add_texts(texts: str) -> str:
        """Add text documents."""
        try:
            text_list = [t.strip() for t in texts.split("\n---\n") if t.strip()]
            num_chunks = rag.add_texts(text_list)
            return f"Added {len(text_list)} documents ({num_chunks} chunks)"
        except Exception as e:
            return f"Error: {str(e)}"

    def search(query: str, k: int) -> str:
        """Search documents."""
        try:
            docs = rag.retrieve(query, k=int(k))
            results = []
            for i, doc in enumerate(docs, 1):
                results.append(f"**[{i}]** {doc.page_content[:300]}...")
                results.append(f"Source: {doc.metadata.get('source', 'unknown')}")
                results.append("")
            return "\n".join(results)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_stats() -> str:
        """Get knowledge base statistics."""
        return f"""
### Knowledge Base Statistics

- **Documents:** {rag.num_documents}
- **Chunks:** {rag.num_chunks}
- **RAG Type:** {rag_type}
- **LLM:** {llm_provider}/{llm_model}
- **Embeddings:** {embedding_provider}
"""

    # ========================================================================
    # Build UI
    # ========================================================================

    with gr.Blocks(
        title="Ultimate RAG",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 2rem; }
        """
    ) as ui:

        # Header
        gr.Markdown("""
        <div class="header">
            <h1>🚀 Ultimate RAG Framework</h1>
            <p>Retrieval-Augmented Generation for AI Models</p>
        </div>
        """)

        with gr.Tabs():
            # ==================================================================
            # Chat Tab
            # ==================================================================
            with gr.Tab("💬 Chat"):
                gr.Markdown("### Chat with your Knowledge Base")
                gr.ChatInterface(
                    fn=chat,
                    title="RAG Chat",
                    description="Ask questions about your documents",
                    examples=[
                        "What is Python?",
                        "Explain machine learning",
                        "What are the main features of this framework?",
                    ],
                )

            # ==================================================================
            # Documents Tab
            # ==================================================================
            with gr.Tab("📄 Documents"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Upload Files")
                        file_upload = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md", ".csv", ".json"],
                            label="Upload Documents"
                        )
                        upload_btn = gr.Button("Upload & Index", variant="primary")
                        upload_output = gr.Textbox(label="Upload Status")

                    with gr.Column():
                        gr.Markdown("### Add Text")
                        text_input = gr.Textbox(
                            lines=10,
                            placeholder="Enter text here...\nUse --- to separate multiple documents",
                            label="Text Documents"
                        )
                        add_text_btn = gr.Button("Add Text", variant="primary")
                        text_output = gr.Textbox(label="Status")

                upload_btn.click(upload_files, inputs=file_upload, outputs=upload_output)
                add_text_btn.click(add_texts, inputs=text_input, outputs=text_output)

            # ==================================================================
            # Search Tab
            # ==================================================================
            with gr.Tab("🔍 Search"):
                gr.Markdown("### Search Documents")
                with gr.Row():
                    search_input = gr.Textbox(
                        placeholder="Enter search query...",
                        label="Search Query"
                    )
                    search_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
                search_btn = gr.Button("Search", variant="primary")
                search_output = gr.Markdown(label="Results")

                search_btn.click(
                    search,
                    inputs=[search_input, search_k],
                    outputs=search_output
                )

            # ==================================================================
            # Statistics Tab
            # ==================================================================
            with gr.Tab("📊 Statistics"):
                gr.Markdown("### Knowledge Base Statistics")
                stats_btn = gr.Button("Refresh Statistics", variant="primary")
                stats_output = gr.Markdown(value=get_stats())

                stats_btn.click(get_stats, outputs=stats_output)

    return ui


def main():
    """Run the Web UI."""
    config = Config()

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

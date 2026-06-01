"""
Naive RAG
=========

Basic RAG implementation: Vector Search + LLM Generation.

This is the simplest RAG pattern, suitable for:
- Basic Q&A systems
- Document search
- Simple chatbots

Pipeline:
1. Load documents
2. Split into chunks
3. Create embeddings
4. Store in vector database
5. Retrieve relevant chunks
6. Generate answer with LLM

Usage:
    rag = NaiveRAG()
    rag.add_documents(["doc1.pdf", "doc2.pdf"])
    answer = rag.query("What is Python?")
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from langchain_core.documents import Document

from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingsManager
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager


class NaiveRAG:
    """
    Basic RAG implementation.

    Simple vector search + LLM generation pipeline.

    Example:
        # Initialize
        rag = NaiveRAG(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            embedding_provider="huggingface"
        )

        # Add documents
        rag.add_documents(["document.pdf", "article.txt"])

        # Query
        answer = rag.query("What is the main topic?")

        # Query with sources
        result = rag.query_with_sources("What is the main topic?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        embedding_provider: str = "huggingface",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_provider: str = "faiss",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 4,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Naive RAG.

        Args:
            llm_provider: LLM provider ('openai', 'anthropic', 'ollama')
            llm_model: LLM model name
            llm_api_key: LLM API key
            embedding_provider: Embedding provider ('huggingface', 'openai')
            embedding_model: Embedding model name
            vector_store_provider: Vector store ('faiss', 'chroma')
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            retrieval_k: Number of documents to retrieve
            system_prompt: Custom system prompt
        """
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = EmbeddingsManager(
            provider=embedding_provider,
            model_name=embedding_model,
        )
        self.vector_store = VectorStoreManager(
            provider=vector_store_provider,
            embeddings=self.embeddings,
        )
        self.llm = LLMManager(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
        )

        self.retrieval_k = retrieval_k
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Track loaded documents
        self._documents = []
        self._chunks = []

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a helpful AI assistant. Use the provided context to answer the user's question.

Rules:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
3. Be concise and accurate
4. Cite relevant parts of the context when possible

Context:
{context}

Question: {question}"""

    def add_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add documents to the knowledge base.

        Args:
            sources: File path(s) or directory path(s)
            metadata: Additional metadata to attach

        Returns:
            Number of chunks added
        """
        # Normalize to list
        if isinstance(sources, (str, Path)):
            sources = [sources]

        # Load documents
        all_docs = []
        for source in sources:
            source = Path(source)
            if source.is_dir():
                docs = self.document_loader.load_directory(source, metadata=metadata)
            else:
                docs = self.document_loader.load(source, metadata=metadata)
            all_docs.extend(docs)

        self._documents.extend(all_docs)

        # Split into chunks
        chunks = self.text_splitter.split_documents(all_docs)
        self._chunks.extend(chunks)

        # Add to vector store
        self.vector_store.add_documents(chunks)

        return len(chunks)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add raw texts to the knowledge base.

        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text

        Returns:
            Number of chunks added
        """
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            docs.append(Document(page_content=text, metadata=metadata))

        self._documents.extend(docs)

        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        self._chunks.extend(chunks)

        # Add to vector store
        self.vector_store.add_documents(chunks)

        return len(chunks)

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Query the knowledge base.

        Args:
            question: Question to ask
            k: Number of documents to retrieve
            filter: Metadata filter

        Returns:
            Answer string
        """
        k = k or self.retrieval_k

        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(
            question, k=k, filter=filter
        )

        # Build context
        context = self._build_context(docs)

        # Generate answer
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )

        answer = self.llm.generate(prompt, **kwargs)

        return answer

    def query_with_sources(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with source documents returned.

        Args:
            question: Question to ask
            k: Number of documents to retrieve
            filter: Metadata filter

        Returns:
            Dict with 'answer' and 'sources' keys
        """
        k = k or self.retrieval_k

        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(
            question, k=k, filter=filter
        )

        # Build context
        context = self._build_context(docs)

        # Generate answer
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )

        answer = self.llm.generate(prompt, **kwargs)

        # Format sources
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
            })

        return {
            "answer": answer,
            "sources": sources,
        }

    def _build_context(self, docs: List[Document]) -> str:
        """Build context string from documents."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents without generating answer.

        Args:
            query: Search query
            k: Number of documents
            filter: Metadata filter

        Returns:
            List of relevant Document objects
        """
        k = k or self.retrieval_k
        return self.vector_store.similarity_search(query, k=k, filter=filter)

    def get_retriever(self, **kwargs):
        """Get retriever interface for LangChain integration."""
        return self.vector_store.get_retriever(
            k=self.retrieval_k,
            **kwargs
        )

    @property
    def num_documents(self) -> int:
        """Number of loaded documents."""
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        """Number of chunks."""
        return len(self._chunks)

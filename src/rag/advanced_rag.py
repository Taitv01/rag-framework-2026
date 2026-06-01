"""
Advanced RAG
============

Enhanced RAG with hybrid search, re-ranking, and query transformation.

Features:
- Hybrid search (vector + BM25)
- Cross-encoder re-ranking
- Query transformation
- Document grading

Pipeline:
1. Load and chunk documents
2. Create vector + BM25 indices
3. Transform query for better retrieval
4. Hybrid retrieval (vector + keyword)
5. Re-rank candidates
6. Grade document relevance
7. Generate answer with LLM

Usage:
    rag = AdvancedRAG()
    rag.add_documents(["docs/"])
    answer = rag.query("What is Python?")
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from langchain_core.documents import Document

from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingsManager
from src.core.vector_store import VectorStoreManager
from src.core.retriever import RetrieverManager
from src.core.llm import LLMManager


class AdvancedRAG:
    """
    Advanced RAG with hybrid search and re-ranking.

    Implements a sophisticated retrieval pipeline with:
    - Query transformation for better recall
    - Hybrid search (vector + BM25)
    - Cross-encoder re-ranking for precision
    - Document relevance grading

    Example:
        rag = AdvancedRAG(
            llm_provider="openai",
            use_hybrid=True,
            use_reranking=True
        )

        rag.add_documents(["documents/"])

        # Standard query
        answer = rag.query("What is machine learning?")

        # Query with detailed results
        result = rag.query_detailed("What is machine learning?")
        print(result["answer"])
        print(result["transformed_query"])
        print(result["relevant_docs"])
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
        retrieval_k: int = 5,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Advanced RAG.

        Args:
            llm_provider: LLM provider
            llm_model: LLM model name
            llm_api_key: LLM API key
            embedding_provider: Embedding provider
            embedding_model: Embedding model name
            vector_store_provider: Vector store provider
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            retrieval_k: Number of documents to retrieve
            use_hybrid: Enable hybrid search
            use_reranking: Enable re-ranking
            reranker_model: Re-ranker model name
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
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Track documents
        self._documents = []
        self._chunks = []

        # Initialize retriever (will be created after documents are added)
        self._retriever = None

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a helpful AI assistant. Use the provided context to answer the user's question accurately.

Rules:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
3. Be concise and accurate
4. Cite specific parts of the context when possible

Context:
{context}

Question: {question}"""

    def _get_query_transform_prompt(self) -> str:
        """Get query transformation prompt."""
        return """You are a search query optimizer. Your task is to transform the user's question into a better search query.

Original question: {question}

Transform this into a clearer, more specific search query that will retrieve relevant documents.
Return ONLY the transformed query, nothing else."""

    def _get_grading_prompt(self) -> str:
        """Get document grading prompt."""
        return """You are a document relevance grader. Determine if the document is relevant to the question.

Question: {question}

Document:
{document}

Is this document relevant to the question? Answer only 'yes' or 'no'."""

    def add_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add documents to the knowledge base.

        Args:
            sources: File path(s) or directory path(s)
            metadata: Additional metadata

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

        # Initialize retriever
        self._retriever = RetrieverManager(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            documents=self._chunks,
            k=self.retrieval_k,
            use_hybrid=self.use_hybrid,
            use_reranking=self.use_reranking,
        )

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

        # Initialize retriever
        self._retriever = RetrieverManager(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            documents=self._chunks,
            k=self.retrieval_k,
            use_hybrid=self.use_hybrid,
            use_reranking=self.use_reranking,
        )

        return len(chunks)

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        transform_query: bool = True,
        grade_documents: bool = True,
        **kwargs
    ) -> str:
        """
        Query with advanced retrieval.

        Args:
            question: Question to ask
            k: Number of documents
            transform_query: Whether to transform query
            grade_documents: Whether to grade document relevance

        Returns:
            Answer string
        """
        k = k or self.retrieval_k

        # Step 1: Transform query (optional)
        search_query = question
        if transform_query:
            search_query = self._transform_query(question)

        # Step 2: Retrieve documents
        if self._retriever:
            docs = self._retriever.search(search_query, k=k)
        else:
            docs = self.vector_store.similarity_search(search_query, k=k)

        # Step 3: Grade documents (optional)
        if grade_documents:
            docs = self._grade_documents(question, docs)

        # Step 4: Generate answer
        context = self._build_context(docs)
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )

        answer = self.llm.generate(prompt, **kwargs)

        return answer

    def query_detailed(
        self,
        question: str,
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with detailed results.

        Args:
            question: Question to ask
            k: Number of documents

        Returns:
            Dict with answer, transformed query, and sources
        """
        k = k or self.retrieval_k

        # Transform query
        transformed_query = self._transform_query(question)

        # Retrieve documents
        if self._retriever:
            docs = self._retriever.search(transformed_query, k=k)
        else:
            docs = self.vector_store.similarity_search(transformed_query, k=k)

        # Grade documents
        relevant_docs = self._grade_documents(question, docs)

        # Generate answer
        context = self._build_context(relevant_docs)
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )

        answer = self.llm.generate(prompt, **kwargs)

        # Format sources
        sources = []
        for doc in relevant_docs:
            sources.append({
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata,
            })

        return {
            "answer": answer,
            "original_query": question,
            "transformed_query": transformed_query,
            "relevant_docs": sources,
            "total_docs_retrieved": len(docs),
            "relevant_docs_count": len(relevant_docs),
        }

    def _transform_query(self, question: str) -> str:
        """Transform query for better retrieval."""
        try:
            prompt = self._get_query_transform_prompt().format(question=question)
            transformed = self.llm.generate(prompt)
            return transformed.strip()
        except Exception:
            # Fallback to original query
            return question

    def _grade_documents(
        self,
        question: str,
        docs: List[Document]
    ) -> List[Document]:
        """Grade documents for relevance."""
        relevant_docs = []

        for doc in docs:
            try:
                prompt = self._get_grading_prompt().format(
                    question=question,
                    document=doc.page_content[:500]  # Limit for grading
                )
                grade = self.llm.generate(prompt).strip().lower()

                if grade == "yes":
                    relevant_docs.append(doc)
            except Exception:
                # If grading fails, include the document
                relevant_docs.append(doc)

        # Return at least one document
        return relevant_docs if relevant_docs else docs[:1]

    def _build_context(self, docs: List[Document]) -> str:
        """Build context from documents."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        use_reranking: Optional[bool] = None
    ) -> List[Document]:
        """
        Retrieve documents with specific strategy.

        Args:
            query: Search query
            k: Number of results
            use_hybrid: Override hybrid setting
            use_reranking: Override reranking setting

        Returns:
            List of relevant documents
        """
        k = k or self.retrieval_k

        if self._retriever is None:
            return self.vector_store.similarity_search(query, k=k)

        # Temporarily override settings if needed
        original_hybrid = self._retriever.config.use_hybrid
        original_reranking = self._retriever.config.use_reranking

        if use_hybrid is not None:
            self._retriever.config.use_hybrid = use_hybrid
        if use_reranking is not None:
            self._retriever.config.use_reranking = use_reranking

        try:
            results = self._retriever.search(query, k=k)
        finally:
            # Restore original settings
            self._retriever.config.use_hybrid = original_hybrid
            self._retriever.config.use_reranking = original_reranking

        return results

    @property
    def num_documents(self) -> int:
        """Number of loaded documents."""
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        """Number of chunks."""
        return len(self._chunks)

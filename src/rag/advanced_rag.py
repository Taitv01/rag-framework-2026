"""
Advanced RAG
============

Enhanced RAG with hybrid search, re-ranking, and query transformation.

Features:
- Hybrid search (vector + BM25) with Vietnamese tokenization
- Cross-encoder re-ranking (Vietnamese-aware)
- Query transformation (bilingual)
- Document grading
- Semantic caching (embedding-based similarity)
- Contextual retrieval (Anthropic pattern)
- HyDE (Hypothetical Document Embeddings)
- Multi-query with RRF (Reciprocal Rank Fusion)
- Streaming responses

Pipeline:
1. Load and chunk documents (optionally with contextual headers)
2. Create vector + BM25 indices
3. Check semantic cache
4. Transform query for better retrieval (optionally HyDE)
5. Hybrid retrieval (vector + keyword, optionally multi-query RRF)
6. Re-rank candidates
7. Grade document relevance
8. Generate answer with LLM
9. Cache the result

Usage:
    rag = AdvancedRAG(use_cache=True, use_contextual_chunking=True)
    rag.add_documents(["docs/"])
    answer = rag.query("Thạch Sanh là ai?")
"""

import logging
from typing import List, Optional, Dict, Any, Union, Generator
from pathlib import Path

from langchain_core.documents import Document

from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingsManager
from src.core.vector_store import VectorStoreManager
from src.core.retriever import RetrieverManager
from src.core.llm import LLMManager

logger = logging.getLogger(__name__)


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
        answer = rag.query("Thạch Sanh là ai?")

        # Query with detailed results
        result = rag.query_detailed("Thạch Sanh đánh đại bàng như thế nào?")
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
        embedding_model: str = "keepitreal/vietnamese-sbert",
        vector_store_provider: str = "faiss",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 5,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        system_prompt: Optional[str] = None,
        # Phase 2 options
        use_cache: bool = False,
        cache_ttl: int = 3600,
        cache_threshold: float = 0.95,
        use_contextual_chunking: bool = False,
        use_hyde: bool = False,
        use_multi_query_rrf: bool = False,
        num_query_variations: int = 3,
    ):
        """
        Initialize Advanced RAG.

        Args:
            llm_provider: LLM provider
            llm_model: LLM model name
            llm_api_key: LLM API key
            embedding_provider: Embedding provider
            embedding_model: Embedding model name (default: Vietnamese SBERT)
            vector_store_provider: Vector store provider
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            retrieval_k: Number of documents to retrieve
            use_hybrid: Enable hybrid search
            use_reranking: Enable re-ranking
            system_prompt: Custom system prompt
            use_cache: Enable semantic caching
            cache_ttl: Cache time-to-live in seconds
            cache_threshold: Similarity threshold for cache hit (0-1)
            use_contextual_chunking: Use Anthropic-style contextual retrieval chunking
            use_hyde: Use HyDE (Hypothetical Document Embeddings) for retrieval
            use_multi_query_rrf: Use multi-query with RRF fusion
            num_query_variations: Number of query variations for multi-query
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

        # Phase 2 options
        self.use_hyde = use_hyde
        self.use_multi_query_rrf = use_multi_query_rrf
        self.num_query_variations = num_query_variations
        self.use_contextual_chunking = use_contextual_chunking

        # Semantic cache
        self._cache = None
        if use_cache:
            from src.utils.cache import SemanticCache
            self._cache = SemanticCache(
                embeddings=self.embeddings,
                threshold=cache_threshold,
                ttl=cache_ttl,
            )
            logger.info(f"Semantic cache enabled (threshold={cache_threshold}, ttl={cache_ttl}s)")

        # Contextual chunker (lazy init)
        self._contextual_chunker = None
        if use_contextual_chunking:
            from src.core.advanced_chunking import ContextualRetrievalChunker
            self._contextual_chunker = ContextualRetrievalChunker(
                llm=self.llm,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            logger.info("Contextual retrieval chunking enabled")

        # Track documents
        self._documents = []
        self._chunks = []
        self._parent_chunks = []  # For parent-child retrieval

        # Initialize retriever (will be created after documents are added)
        self._retriever = None

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt (bilingual Vietnamese/English)."""
        return """You are a helpful AI assistant / Bạn là trợ lý AI hữu ích.
Use the provided context to answer the user's question accurately.
Sử dụng ngữ cảnh được cung cấp để trả lời câu hỏi một cách chính xác.

Rules / Quy tắc:
1. Answer based ONLY on the provided context / Chỉ trả lời dựa trên ngữ cảnh
2. If the context doesn't contain the answer, say "I don't have enough information" / Nếu không đủ thông tin, hãy nói rõ
3. Be concise and accurate / Ngắn gọn và chính xác
4. Answer in the same language as the question / Trả lời bằng ngôn ngữ của câu hỏi

Context / Ngữ cảnh:
{context}

Question / Câu hỏi: {question}"""

    def _get_query_transform_prompt(self) -> str:
        """Get query transformation prompt (bilingual)."""
        return """You are a search query optimizer / Bạn là người tối ưu hóa truy vấn.
Transform the user's question into a better search query.
Chuyển đổi câu hỏi của người dùng thành truy vấn tìm kiếm tốt hơn.

Original question / Câu hỏi gốc: {question}

Transform this into a clearer, more specific search query that will retrieve relevant documents.
Chuyển đổi thành truy vấn rõ ràng hơn, cụ thể hơn.
Return ONLY the transformed query, nothing else."""

    def _get_grading_prompt(self) -> str:
        """Get document grading prompt (bilingual)."""
        return """You are a document relevance grader / Bạn là người đánh giá tài liệu.
Determine if the document is relevant to the question.
Xác định tài liệu có liên quan đến câu hỏi không.

Question / Câu hỏi: {question}

Document / Tài liệu:
{document}

Is this document relevant? Answer only 'yes' or 'no'.
Tài liệu này có liên quan không? Chỉ trả lời 'yes' hoặc 'no'."""

    def add_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add documents to the knowledge base.

        Uses contextual retrieval chunking if enabled (Anthropic pattern),
        otherwise falls back to standard text splitting.

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

        # Split into chunks (contextual or standard)
        if self._contextual_chunker:
            logger.info("Using contextual retrieval chunking")
            chunks = self._contextual_chunker.split(all_docs)
        else:
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

        Pipeline:
        1. Check semantic cache (if enabled)
        2. Transform query (optionally HyDE)
        3. Retrieve (optionally multi-query RRF)
        4. Grade documents
        5. Generate answer
        6. Cache result

        Args:
            question: Question to ask
            k: Number of documents
            transform_query: Whether to transform query
            grade_documents: Whether to grade document relevance

        Returns:
            Answer string
        """
        k = k or self.retrieval_k

        # Step 1: Check semantic cache
        if self._cache:
            try:
                query_embedding = self.embeddings.embed_query(question)
                cached = self._cache.get(query_embedding)
                if cached is not None:
                    logger.debug("Semantic cache hit")
                    return cached
            except Exception as e:
                logger.debug(f"Cache lookup failed: {e}")

        # Step 2: Transform query
        search_query = question
        if transform_query:
            search_query = self._transform_query(question)

        # Step 3: Retrieve documents
        docs = self._retrieve(search_query, k=k)

        # Step 4: Grade documents (optional)
        if grade_documents:
            docs = self._grade_documents(question, docs)

        # Step 5: Generate answer
        context = self._build_context(docs)
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )

        answer = self.llm.generate(prompt, **kwargs)

        # Step 6: Cache the result
        if self._cache:
            try:
                query_embedding = self.embeddings.embed_query(question)
                self._cache.put(query_embedding, question, answer)
            except Exception as e:
                logger.debug(f"Cache store failed: {e}")

        return answer

    def _retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Internal retrieval method supporting multiple strategies.

        Checks in order:
        1. HyDE (if enabled)
        2. Multi-query RRF (if enabled)
        3. Standard search
        """
        if self._retriever is None:
            return self.vector_store.similarity_search(query, k=k)

        # HyDE: generate hypothetical answer, search with that
        if self.use_hyde:
            docs = self._retriever.hyde_search(query, k=k, llm=self.llm)
            if docs:
                return docs

        # Multi-query RRF: generate variations, fuse with RRF
        if self.use_multi_query_rrf:
            docs = self._retriever.multi_query_rrf_search(
                query, k=k, llm=self.llm,
                num_queries=self.num_query_variations,
            )
            if docs:
                return docs

        # Standard search (hybrid + reranking)
        return self._retriever.search(query, k=k)

    def query_detailed(
        self,
        question: str,
        k: Optional[int] = None,
        transform_query: bool = True,
        grade_documents: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with detailed results.

        Args:
            question: Question to ask
            k: Number of documents
            transform_query: Whether to transform query
            grade_documents: Whether to grade document relevance

        Returns:
            Dict with answer, transformed query, and sources
        """
        k = k or self.retrieval_k

        # Check cache
        cache_hit = False
        if self._cache:
            try:
                query_embedding = self.embeddings.embed_query(question)
                cached = self._cache.get(query_embedding)
                if cached is not None:
                    cache_hit = True
                    return {
                        "answer": cached,
                        "original_query": question,
                        "transformed_query": None,
                        "relevant_docs": [],
                        "total_docs_retrieved": 0,
                        "relevant_docs_count": 0,
                        "cache_hit": True,
                    }
            except Exception:
                pass

        # Transform query
        transformed_query = question
        if transform_query:
            transformed_query = self._transform_query(question)

        # Retrieve documents
        docs = self._retrieve(transformed_query, k=k)

        # Grade documents
        relevant_docs = docs
        if grade_documents:
            relevant_docs = self._grade_documents(question, docs)

        # Generate answer
        context = self._build_context(relevant_docs)
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )

        answer = self.llm.generate(prompt, **kwargs)

        # Cache the result
        if self._cache:
            try:
                query_embedding = self.embeddings.embed_query(question)
                self._cache.put(query_embedding, question, answer)
            except Exception:
                pass

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
            "cache_hit": cache_hit,
        }

    def _transform_query(self, question: str) -> str:
        """Transform query for better retrieval."""
        try:
            prompt = self._get_query_transform_prompt().format(question=question)
            transformed = self.llm.generate(prompt)
            return transformed.strip()
        except Exception as e:
            logger.warning(f"Query transformation failed, using original: {e}")
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
                    document=doc.page_content[:500]
                )
                grade = self.llm.generate(prompt).strip().lower()

                if grade in ("yes", "có", "đúng"):
                    relevant_docs.append(doc)
            except Exception as e:
                logger.warning(f"Document grading failed, including doc: {e}")
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

    def stream(
        self,
        question: str,
        k: Optional[int] = None,
        transform_query: bool = True,
        grade_documents: bool = True,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream response tokens.

        Same pipeline as query() but yields tokens as they arrive.
        Used by ConversationalRAG.stream().

        Args:
            question: Question to ask
            k: Number of documents
            transform_query: Whether to transform query
            grade_documents: Whether to grade document relevance

        Yields:
            Response tokens
        """
        k = k or self.retrieval_k

        # Check cache (return full answer if cached)
        if self._cache:
            try:
                query_embedding = self.embeddings.embed_query(question)
                cached = self._cache.get(query_embedding)
                if cached is not None:
                    yield cached
                    return
            except Exception:
                pass

        # Transform query
        search_query = question
        if transform_query:
            search_query = self._transform_query(question)

        # Retrieve
        docs = self._retrieve(search_query, k=k)

        # Grade
        if grade_documents:
            docs = self._grade_documents(question, docs)

        # Build context
        context = self._build_context(docs)
        prompt = self.system_prompt.format(
            context=context,
            question=question
        )

        # Stream response
        full_response = []
        for token in self.llm.stream(prompt, **kwargs):
            full_response.append(token)
            yield token

        # Cache the full response
        if self._cache:
            try:
                query_embedding = self.embeddings.embed_query(question)
                self._cache.put(query_embedding, question, "".join(full_response))
            except Exception:
                pass

    @property
    def cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if cache is enabled."""
        if self._cache:
            return self._cache.stats()
        return None

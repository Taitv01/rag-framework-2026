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
        # Phase 3 options
        use_web_search: bool = False,
        web_search_provider: str = "duckduckgo",
        web_search_api_key: Optional[str] = None,
        use_hallucination_check: bool = False,
        hallucination_threshold: float = 0.8,
        use_metadata_enhancement: bool = False,
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
            use_web_search: Enable web search fallback when retrieval quality is poor
            web_search_provider: Web search provider ("duckduckgo" or "tavily")
            web_search_api_key: API key for web search provider (if needed)
            use_hallucination_check: Enable hallucination verification after generation
            hallucination_threshold: Minimum grounded score to accept answer (0-1)
            use_metadata_enhancement: Enable LLM-based metadata extraction for chunks
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

        # Context window validation
        from src.utils.context_validator import ContextValidator
        self._context_validator = ContextValidator.from_llm_manager(self.llm)

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

        # Phase 3: Web search fallback
        self._web_searcher = None
        if use_web_search:
            from src.core.web_search import create_web_searcher
            self._web_searcher = create_web_searcher(
                provider=web_search_provider,
                llm=self.llm,
                api_key=web_search_api_key,
            )
            logger.info(f"Web search fallback enabled (provider={web_search_provider})")

        # Phase 3: Hallucination grader
        self._hallucination_grader = None
        self.use_hallucination_check = use_hallucination_check
        if use_hallucination_check:
            from src.agents.hallucination_grader import HallucinationGrader
            self._hallucination_grader = HallucinationGrader(
                llm=self.llm,
                grounded_threshold=hallucination_threshold,
            )
            logger.info(f"Hallucination check enabled (threshold={hallucination_threshold})")

        # Phase 3: Metadata enhancer
        self._metadata_enhancer = None
        if use_metadata_enhancement:
            from src.core.metadata_enhancer import MetadataEnhancer
            self._metadata_enhancer = MetadataEnhancer(llm=self.llm)
            logger.info("Metadata enhancement enabled")

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

        # Enhance metadata if enabled
        if self._metadata_enhancer:
            logger.info(f"Enhancing metadata for {len(chunks)} chunks")
            chunks = self._metadata_enhancer.enhance(chunks)

        # Store parent chunks for parent-child retrieval
        parent_start_idx = len(self._parent_chunks)
        self._parent_chunks.extend(chunks)

        # Create smaller child chunks for retrieval, linking to parents
        child_splitter = TextSplitter(
            chunk_size=self.text_splitter.chunk_size // 2,
            chunk_overlap=self.text_splitter.chunk_overlap,
        )
        child_chunks = []
        for parent_idx, parent_chunk in enumerate(chunks, start=parent_start_idx):
            sub_chunks = child_splitter.split_documents([parent_chunk])
            for child in sub_chunks:
                child.metadata = {**child.metadata, "_parent_idx": parent_idx}
            child_chunks.extend(sub_chunks)

        # Use child chunks for retrieval if any were created,
        # otherwise fall back to the parent chunks themselves
        retrieval_chunks = child_chunks if child_chunks else chunks
        self._chunks.extend(retrieval_chunks)

        # Add to vector store
        self.vector_store.add_documents(retrieval_chunks)

        # Initialize retriever
        self._retriever = RetrieverManager(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            documents=self._chunks,
            k=self.retrieval_k,
            use_hybrid=self.use_hybrid,
            use_reranking=self.use_reranking,
        )

        return len(retrieval_chunks)

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

        # Enhance metadata if enabled
        if getattr(self, "_metadata_enhancer", None):
            logger.info(f"Enhancing metadata for {len(chunks)} chunks")
            chunks = self._metadata_enhancer.enhance(chunks)

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
        5. Web search fallback (if retrieval quality poor and enabled)
        6. Generate answer
        7. Hallucination check (if enabled)
        8. Cache result

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

        # Step 5: Web search fallback if retrieval quality is poor
        web_docs = []
        if self._web_searcher and self._is_retrieval_quality_poor(docs, question):
            logger.info("Retrieval quality poor, falling back to web search")
            try:
                web_results = self._web_searcher.search(search_query, num_results=3)
                web_docs = self._web_searcher.to_documents(web_results)
                logger.info(f"Web search returned {len(web_docs)} results")
            except Exception as e:
                logger.warning(f"Web search fallback failed: {e}")

        # Step 6: Generate answer
        if web_docs and self._web_searcher:
            # Use special prompt that labels web sources
            prompt = self._web_searcher.create_web_answer_prompt(
                question=question,
                web_docs=web_docs,
                local_docs=docs,
            )
        else:
            context = self._build_context(docs)
            prompt = self.system_prompt.format(
                context=context,
                question=question,
            )

        answer = self.llm.generate(prompt, **kwargs)

        # Step 7: Hallucination check (optional)
        if self._hallucination_grader and docs:
            all_docs = docs + web_docs
            grade = self._hallucination_grader.grade(
                answer=answer,
                context=self._build_context(all_docs),
            )
            if not grade.is_grounded and grade.grounded_score < self._hallucination_grader.grounded_threshold:
                logger.warning(
                    f"Hallucination detected (score={grade.grounded_score:.2f}), "
                    f"unsupported: {grade.unsupported_claims}"
                )
                # Regenerate with stricter prompt
                answer, _ = self._hallucination_grader.safe_generate(
                    question=question,
                    context=self._build_context(all_docs),
                    max_retries=1,
                )

        # Step 8: Cache the result (only if no web search used — web results change frequently)
        if self._cache and not web_docs:
            try:
                query_embedding = self.embeddings.embed_query(question)
                self._cache.put(query_embedding, question, answer)
            except Exception as e:
                logger.debug(f"Cache store failed: {e}")

        return answer

    def _is_retrieval_quality_poor(self, docs: List[Document], question: str) -> bool:
        """
        Check if retrieval quality is poor enough to warrant web search fallback.

        Returns True if:
        - No documents retrieved, OR
        - All documents were graded as irrelevant (empty after grading)
        """
        if not docs:
            return True

        # Check if docs have relevance metadata from grading
        # The _grade_documents method already filters, so if we get here
        # with docs, at least some were deemed relevant
        return False

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
                        "citations": [],
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

        # Format sources and citations
        sources = self._format_sources(relevant_docs)

        return {
            "answer": answer,
            "original_query": question,
            "transformed_query": transformed_query,
            "relevant_docs": sources,
            "citations": sources,
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
        """Build context from documents with context window validation."""
        context_parts = [
            "Source IDs are shown as [S1], [S2], etc. Cite them when using facts."
        ]
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("url")
                or f"Document {i}"
            )
            context_parts.append(f"[S{i}] Source: {source}\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # Validate context fits within model's window
        if self._context_validator:
            result = self._context_validator.validate(
                prompt=context,
                system_prompt=self.system_prompt,
            )

            if result.warning:
                logger.warning(result.warning)

            if result.is_too_large and result.truncated_prompt:
                logger.warning(
                    f"Context truncated: {result.prompt_tokens:,} → "
                    f"{self._context_validator.count_tokens(result.truncated_prompt):,} tokens"
                )
                return result.truncated_prompt

        return context

    def _format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Format retrieved documents with stable source IDs for citations."""
        sources = []

        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            content = doc.page_content
            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("url")
                or f"Document {i}"
            )

            sources.append({
                "source_id": f"S{i}",
                "source": source,
                "content": content[:300] + "..." if len(content) > 300 else content,
                "metadata": metadata,
            })

        return sources

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

    @property
    def context_info(self) -> Dict[str, Any]:
        """
        Get context window information.

        Returns:
            Dict with context window size, available tokens, and current usage
        """
        if not self._context_validator:
            return {"error": "Context validator not initialized"}

        # Estimate current context size
        total_chunk_chars = sum(len(c.page_content) for c in self._chunks)
        estimated_context_tokens = self._context_validator.estimate_tokens(
            "\n\n".join(c.page_content for c in self._chunks[:self.retrieval_k])
        )

        return {
            "context_window": self._context_validator.context_window,
            "available_tokens": self._context_validator.available_tokens,
            "max_output_tokens": self._context_validator.reserve_tokens,
            "retrieval_k": self.retrieval_k,
            "chunk_size_avg": total_chunk_chars // max(len(self._chunks), 1),
            "estimated_context_tokens": estimated_context_tokens,
            "usage_ratio": round(
                estimated_context_tokens / self._context_validator.available_tokens, 3
            ) if self._context_validator.available_tokens > 0 else 1.0,
            "model": self.llm.config.model,
            "provider": self.llm.config.provider,
        }

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

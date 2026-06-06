"""
Adaptive RAG
============

Intelligent RAG router that selects the optimal pipeline based on query complexity.

Features:
- LLM-based query complexity classification (simple/medium/complex)
- Routes to appropriate RAG pipeline:
  - Simple queries → NaiveRAG (lightweight vector search)
  - Medium queries → AdvancedRAG (hybrid search + reranking)
  - Complex multi-hop queries → AgenticRAG (full agentic pipeline)
- Bilingual (Vietnamese/English) classification prompts
- Logging of routing decisions
- Shared knowledge base across all pipelines

Pipeline:
1. Receive query
2. Classify complexity via LLM router
3. Dispatch to appropriate RAG pipeline
4. Return answer (optionally with sources and routing metadata)

Usage:
    rag = AdaptiveRAG()
    rag.add_documents(["docs/"])
    answer = rag.query("Thạch Sanh là ai?")

    # With routing metadata
    result = rag.query_with_sources("So sánh Thạch Sanh và Lý Thông")
    print(result["route"])  # "complex"
"""

import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document

from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingsManager
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager

logger = logging.getLogger(__name__)


class ComplexityLevel(str, Enum):
    """Query complexity levels for routing."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class QueryComplexity:
    """
    Result of query complexity classification.

    Attributes:
        level: Classified complexity level
        reasoning: LLM's reasoning for the classification
        original_query: The original query text
        confidence: Optional confidence score from the LLM
    """
    level: ComplexityLevel
    reasoning: str = ""
    original_query: str = ""
    confidence: float = 0.0


@dataclass
class RoutingConfig:
    """
    Configuration for the adaptive router.

    Attributes:
        default_route: Default route when classification fails
        enable_fallback: Whether to fall back to a simpler pipeline on error
        cache_routes: Whether to cache routing decisions
        max_classification_retries: Max retries for classification failures
    """
    default_route: ComplexityLevel = ComplexityLevel.MEDIUM
    enable_fallback: bool = True
    cache_routes: bool = False
    max_classification_retries: int = 2


class AdaptiveRAG:
    """
    Adaptive RAG that routes queries to the optimal pipeline.

    Uses an LLM-based router to classify query complexity and dispatches
    to the appropriate RAG implementation:
    - Simple → NaiveRAG (fast, lightweight)
    - Medium → AdvancedRAG (hybrid search + reranking)
    - Complex → AgenticRAG (multi-step agentic reasoning)

    Example:
        rag = AdaptiveRAG(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        )

        # Add documents (shared across all pipelines)
        rag.add_documents(["documents/"])

        # Simple query → routed to NaiveRAG
        answer = rag.query("Thạch Sanh là ai?")

        # Complex query → routed to AgenticRAG
        result = rag.query_with_sources(
            "So sánh nhân vật Thạch Sanh và Lý Thông về phẩm chất đạo đức"
        )
        print(result["answer"])
        print(result["route"])  # "complex"
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
        routing_config: Optional[RoutingConfig] = None,
        # Pipeline-specific options
        advanced_use_hybrid: bool = True,
        advanced_use_reranking: bool = True,
        agentic_max_retries: int = 3,
    ):
        """
        Initialize Adaptive RAG.

        Args:
            llm_provider: LLM provider ('openai', 'anthropic', 'ollama')
            llm_model: LLM model name
            llm_api_key: LLM API key
            embedding_provider: Embedding provider ('huggingface', 'openai')
            embedding_model: Embedding model name
            vector_store_provider: Vector store provider ('faiss', 'chroma')
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            retrieval_k: Number of documents to retrieve
            routing_config: Router configuration
            advanced_use_hybrid: Enable hybrid search in AdvancedRAG
            advanced_use_reranking: Enable reranking in AdvancedRAG
            agentic_max_retries: Max retries for AgenticRAG
        """
        self._routing_config = routing_config or RoutingConfig()

        # Store shared parameters for lazy pipeline initialization
        self._shared_params = {
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_api_key": llm_api_key,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "vector_store_provider": vector_store_provider,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_k": retrieval_k,
        }
        self._advanced_params = {
            "use_hybrid": advanced_use_hybrid,
            "use_reranking": advanced_use_reranking,
        }
        self._agentic_params = {
            "max_retries": agentic_max_retries,
        }

        # Initialize the router LLM (lightweight, used only for classification)
        self._router_llm = LLMManager(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
        )

        # Lazy-initialized pipelines (created on first add_documents)
        self._naive_rag = None
        self._advanced_rag = None
        self._agentic_rag = None
        self._pipelines_initialized = False

        # Routing cache
        self._route_cache: Dict[str, QueryComplexity] = {}

        # Statistics
        self._route_stats: Dict[str, int] = {
            ComplexityLevel.SIMPLE: 0,
            ComplexityLevel.MEDIUM: 0,
            ComplexityLevel.COMPLEX: 0,
        }

        # Track documents
        self._documents: List[Document] = []

        logger.info(
            f"AdaptiveRAG initialized with router LLM={llm_model}, "
            f"default_route={self._routing_config.default_route.value}"
        )

    def _initialize_pipelines(self):
        """Lazily initialize all RAG pipelines."""
        if self._pipelines_initialized:
            return

        from src.rag.naive_rag import NaiveRAG
        from src.rag.advanced_rag import AdvancedRAG
        from src.rag.agentic_rag import AgenticRAG

        logger.info("Initializing RAG pipelines for AdaptiveRAG...")

        # NaiveRAG for simple queries
        self._naive_rag = NaiveRAG(
            llm_provider=self._shared_params["llm_provider"],
            llm_model=self._shared_params["llm_model"],
            llm_api_key=self._shared_params["llm_api_key"],
            embedding_provider=self._shared_params["embedding_provider"],
            embedding_model=self._shared_params["embedding_model"],
            vector_store_provider=self._shared_params["vector_store_provider"],
            chunk_size=self._shared_params["chunk_size"],
            chunk_overlap=self._shared_params["chunk_overlap"],
            retrieval_k=self._shared_params["retrieval_k"],
        )

        # AdvancedRAG for medium queries
        self._advanced_rag = AdvancedRAG(
            llm_provider=self._shared_params["llm_provider"],
            llm_model=self._shared_params["llm_model"],
            llm_api_key=self._shared_params["llm_api_key"],
            embedding_provider=self._shared_params["embedding_provider"],
            embedding_model=self._shared_params["embedding_model"],
            vector_store_provider=self._shared_params["vector_store_provider"],
            chunk_size=self._shared_params["chunk_size"],
            chunk_overlap=self._shared_params["chunk_overlap"],
            retrieval_k=self._shared_params["retrieval_k"],
            use_hybrid=self._advanced_params["use_hybrid"],
            use_reranking=self._advanced_params["use_reranking"],
        )

        # AgenticRAG for complex queries
        self._agentic_rag = AgenticRAG(
            llm_provider=self._shared_params["llm_provider"],
            llm_model=self._shared_params["llm_model"],
            llm_api_key=self._shared_params["llm_api_key"],
            embedding_provider=self._shared_params["embedding_provider"],
            embedding_model=self._shared_params["embedding_model"],
            vector_store_provider=self._shared_params["vector_store_provider"],
            chunk_size=self._shared_params["chunk_size"],
            chunk_overlap=self._shared_params["chunk_overlap"],
            retrieval_k=self._shared_params["retrieval_k"],
            max_retries=self._agentic_params["max_retries"],
        )

        self._pipelines_initialized = True
        logger.info("All RAG pipelines initialized successfully")

    def _get_classification_prompt(self) -> str:
        """
        Get the LLM prompt for query complexity classification.

        Returns a bilingual (Vietnamese/English) prompt that instructs the LLM
        to classify queries into simple, medium, or complex categories.
        """
        return """You are a query complexity classifier / Bạn là bộ phân loại độ phức tạp truy vấn.

Classify the following query into one of three complexity levels:
Phân loại truy vấn sau vào một trong ba mức độ phức tạp:

SIMPLE - Câu hỏi đơn giản:
- Direct factual questions / Câu hỏi thực tế trực tiếp
- Single-entity lookups / Tra cứu một thực thể
- Definition or description requests / Yêu cầu định nghĩa hoặc mô tả
- Examples: "X là ai?", "What is X?", "Khi nào X xảy ra?"

MEDIUM - Câu hỏi trung bình:
- Questions requiring synthesis from multiple sources / Câu hỏi cần tổng hợp từ nhiều nguồn
- Questions with specific filters or conditions / Câu hỏi có điều kiện cụ thể
- Analytical questions about a single topic / Câu hỏi phân tích về một chủ đề
- Examples: "Tại sao X lại Y?", "How does X work?", "Giải thích quá trình X"

COMPLEX - Câu hỏi phức tạp:
- Multi-hop reasoning across multiple documents / Suy luận đa bước qua nhiều tài liệu
- Comparative analysis between entities / Phân tích so sánh giữa các thực thể
- Questions requiring inference and reasoning / Câu hỏi cần suy luận
- Multi-part questions / Câu hỏi nhiều phần
- Examples: "So sánh X và Y", "What are the implications of X on Y?", "Phân tích mối quan hệ giữa X, Y và Z"

Query / Truy vấn: {query}

Respond with EXACTLY one of: SIMPLE, MEDIUM, or COMPLEX
Then on a new line, briefly explain why (1 sentence).
Trả lời CHÍNH XÁC một trong: SIMPLE, MEDIUM, hoặc COMPLEX
Sau đó giải thích ngắn gọn lý do (1 câu).

Format:
LEVEL
Reason"""

    def classify_query(self, query: str) -> QueryComplexity:
        """
        Classify query complexity using the LLM router.

        Args:
            query: The query to classify

        Returns:
            QueryComplexity with the classification result
        """
        # Check cache
        if self._routing_config.cache_routes and query in self._route_cache:
            cached = self._route_cache[query]
            logger.debug(f"Route cache hit: '{query[:50]}...' → {cached.level.value}")
            return cached

        # Classify via LLM
        for attempt in range(self._routing_config.max_classification_retries):
            try:
                prompt = self._get_classification_prompt().format(query=query)
                response = self._router_llm.generate(prompt).strip()

                # Parse response
                complexity = self._parse_classification(response, query)

                # Cache if enabled
                if self._routing_config.cache_routes:
                    self._route_cache[query] = complexity

                logger.info(
                    f"Query classified: '{query[:80]}...' → {complexity.level.value} "
                    f"(reason: {complexity.reasoning[:100]})"
                )
                return complexity

            except Exception as e:
                logger.warning(
                    f"Classification attempt {attempt + 1} failed: {e}"
                )
                if attempt == self._routing_config.max_classification_retries - 1:
                    # Fall back to default
                    logger.warning(
                        f"All classification attempts failed, "
                        f"using default route: {self._routing_config.default_route.value}"
                    )
                    return QueryComplexity(
                        level=self._routing_config.default_route,
                        reasoning="Classification failed, using default route",
                        original_query=query,
                        confidence=0.0,
                    )

        # Should not reach here, but just in case
        return QueryComplexity(
            level=self._routing_config.default_route,
            reasoning="Fallback",
            original_query=query,
        )

    def _parse_classification(self, response: str, query: str) -> QueryComplexity:
        """
        Parse the LLM classification response.

        Args:
            response: Raw LLM response
            query: Original query

        Returns:
            QueryComplexity parsed from the response

        Raises:
            ValueError: If the response cannot be parsed
        """
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

        if not lines:
            raise ValueError("Empty classification response")

        # Extract level from first line
        level_str = lines[0].upper().strip()

        # Handle cases where the LLM includes extra text
        for candidate in [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]:
            if candidate.value.upper() in level_str:
                level = candidate
                break
        else:
            raise ValueError(f"Could not parse complexity level from: '{level_str}'")

        # Extract reasoning from remaining lines
        reasoning = " ".join(lines[1:]) if len(lines) > 1 else ""

        return QueryComplexity(
            level=level,
            reasoning=reasoning,
            original_query=query,
            confidence=1.0,
        )

    def _get_pipeline(self, level: ComplexityLevel):
        """
        Get the appropriate RAG pipeline for a complexity level.

        Args:
            level: Query complexity level

        Returns:
            The corresponding RAG pipeline instance
        """
        pipeline_map = {
            ComplexityLevel.SIMPLE: self._naive_rag,
            ComplexityLevel.MEDIUM: self._advanced_rag,
            ComplexityLevel.COMPLEX: self._agentic_rag,
        }
        return pipeline_map[level]

    def add_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add documents to all RAG pipelines.

        Documents are loaded once and distributed to all pipelines
        to keep their knowledge bases synchronized.

        Args:
            sources: File path(s) or directory path(s)
            metadata: Additional metadata to attach

        Returns:
            Number of chunks added (from the NaiveRAG pipeline)
        """
        # Initialize pipelines on first use
        self._initialize_pipelines()

        # Add to all pipelines
        logger.info(f"Adding documents to all AdaptiveRAG pipelines...")

        naive_chunks = self._naive_rag.add_documents(sources, metadata=metadata)
        logger.debug(f"NaiveRAG: {naive_chunks} chunks added")

        advanced_chunks = self._advanced_rag.add_documents(sources, metadata=metadata)
        logger.debug(f"AdvancedRAG: {advanced_chunks} chunks added")

        agentic_chunks = self._agentic_rag.add_documents(sources, metadata=metadata)
        logger.debug(f"AgenticRAG: {agentic_chunks} chunks added")

        logger.info(
            f"Documents added to all pipelines "
            f"(naive={naive_chunks}, advanced={advanced_chunks}, agentic={agentic_chunks})"
        )

        return naive_chunks

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add raw texts to all RAG pipelines.

        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text

        Returns:
            Number of chunks added (from the NaiveRAG pipeline)
        """
        # Initialize pipelines on first use
        self._initialize_pipelines()

        naive_chunks = self._naive_rag.add_texts(texts, metadatas=metadatas)
        self._advanced_rag.add_texts(texts, metadatas=metadatas)

        # AgenticRAG doesn't have add_texts, so convert to documents and add
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            docs.append(Document(page_content=text, metadata=meta))

        # Use a temporary file approach or directly add chunks
        # For simplicity, we add to its vector store directly
        chunks = self._agentic_rag.text_splitter.split_documents(docs)
        self._agentic_rag._documents.extend(docs)
        self._agentic_rag._chunks.extend(chunks)
        self._agentic_rag.vector_store.add_documents(chunks)
        self._agentic_rag._create_retriever_tool()
        self._agentic_rag._build_graph()

        return naive_chunks

    def query(
        self,
        question: str,
        force_route: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Query using adaptive routing.

        The query is first classified for complexity, then routed
        to the appropriate RAG pipeline.

        Args:
            question: Question to ask
            force_route: Force a specific route ('simple', 'medium', 'complex')
            **kwargs: Additional arguments passed to the pipeline

        Returns:
            Answer string
        """
        if not self._pipelines_initialized:
            raise RuntimeError(
                "No documents loaded. Call add_documents() first."
            )

        # Classify or force route
        if force_route:
            try:
                level = ComplexityLevel(force_route.lower())
            except ValueError:
                logger.warning(
                    f"Invalid force_route '{force_route}', using default"
                )
                level = self._routing_config.default_route
            complexity = QueryComplexity(
                level=level,
                reasoning=f"Forced route: {force_route}",
                original_query=question,
            )
        else:
            complexity = self.classify_query(question)

        # Update stats
        self._route_stats[complexity.level] += 1

        # Get pipeline and execute
        pipeline = self._get_pipeline(complexity.level)

        logger.info(
            f"Routing query to {complexity.level.value} pipeline "
            f"({type(pipeline).__name__})"
        )

        try:
            answer = pipeline.query(question, **kwargs)
            return answer
        except Exception as e:
            if self._routing_config.enable_fallback:
                return self._fallback_query(question, complexity.level, e, **kwargs)
            raise

    def query_with_sources(
        self,
        question: str,
        force_route: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with source documents and routing metadata.

        Args:
            question: Question to ask
            force_route: Force a specific route ('simple', 'medium', 'complex')
            **kwargs: Additional arguments passed to the pipeline

        Returns:
            Dict with 'answer', 'sources', 'route', and 'complexity' keys
        """
        if not self._pipelines_initialized:
            raise RuntimeError(
                "No documents loaded. Call add_documents() first."
            )

        # Classify or force route
        if force_route:
            try:
                level = ComplexityLevel(force_route.lower())
            except ValueError:
                logger.warning(
                    f"Invalid force_route '{force_route}', using default"
                )
                level = self._routing_config.default_route
            complexity = QueryComplexity(
                level=level,
                reasoning=f"Forced route: {force_route}",
                original_query=question,
            )
        else:
            complexity = self.classify_query(question)

        # Update stats
        self._route_stats[complexity.level] += 1

        pipeline = self._get_pipeline(complexity.level)

        logger.info(
            f"Routing query_with_sources to {complexity.level.value} pipeline "
            f"({type(pipeline).__name__})"
        )

        try:
            # Each pipeline has different detailed query methods
            if complexity.level == ComplexityLevel.SIMPLE:
                result = self._naive_rag.query_with_sources(question, **kwargs)
            elif complexity.level == ComplexityLevel.MEDIUM:
                result = self._advanced_rag.query_detailed(question, **kwargs)
            elif complexity.level == ComplexityLevel.COMPLEX:
                result = self._agentic_rag.query_with_trace(question, **kwargs)
            else:
                result = {"answer": pipeline.query(question, **kwargs), "sources": []}

            # Normalize result format
            return self._normalize_result(result, complexity)

        except Exception as e:
            if self._routing_config.enable_fallback:
                answer = self._fallback_query(
                    question, complexity.level, e, **kwargs
                )
                return {
                    "answer": answer,
                    "sources": [],
                    "route": complexity.level.value,
                    "complexity": {
                        "level": complexity.level.value,
                        "reasoning": complexity.reasoning,
                        "confidence": complexity.confidence,
                    },
                    "fallback_used": True,
                }
            raise

    def _normalize_result(
        self, result: Dict[str, Any], complexity: QueryComplexity
    ) -> Dict[str, Any]:
        """
        Normalize results from different pipelines into a common format.

        Args:
            result: Raw result from a pipeline
            complexity: Query complexity classification

        Returns:
            Normalized result dict
        """
        normalized = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", result.get("relevant_docs", [])),
            "route": complexity.level.value,
            "complexity": {
                "level": complexity.level.value,
                "reasoning": complexity.reasoning,
                "confidence": complexity.confidence,
            },
            "fallback_used": False,
        }

        # Include pipeline-specific metadata
        if "transformed_query" in result:
            normalized["transformed_query"] = result["transformed_query"]
        if "trace" in result:
            normalized["trace"] = result["trace"]
        if "cache_hit" in result:
            normalized["cache_hit"] = result["cache_hit"]

        return normalized

    def _fallback_query(
        self,
        question: str,
        failed_level: ComplexityLevel,
        error: Exception,
        **kwargs
    ) -> str:
        """
        Fall back to a simpler pipeline when the target pipeline fails.

        Tries progressively simpler pipelines:
        Complex → Medium → Simple → Direct LLM

        Args:
            question: Original question
            failed_level: The level that failed
            error: The exception that caused the failure
            **kwargs: Additional arguments

        Returns:
            Answer from the fallback pipeline
        """
        fallback_order = {
            ComplexityLevel.COMPLEX: [ComplexityLevel.MEDIUM, ComplexityLevel.SIMPLE],
            ComplexityLevel.MEDIUM: [ComplexityLevel.SIMPLE],
            ComplexityLevel.SIMPLE: [],
        }

        fallbacks = fallback_order.get(failed_level, [])

        for fallback_level in fallbacks:
            try:
                pipeline = self._get_pipeline(fallback_level)
                logger.warning(
                    f"Falling back from {failed_level.value} to "
                    f"{fallback_level.value} due to: {error}"
                )
                return pipeline.query(question, **kwargs)
            except Exception as fallback_error:
                logger.warning(
                    f"Fallback to {fallback_level.value} also failed: {fallback_error}"
                )
                continue

        # Last resort: direct LLM answer without retrieval
        logger.warning(
            "All pipelines failed, falling back to direct LLM answer"
        )
        try:
            return self._router_llm.generate(
                f"Answer this question based on your knowledge: {question}"
            )
        except Exception as final_error:
            logger.error(f"Direct LLM fallback also failed: {final_error}")
            return (
                "I'm sorry, I was unable to process your query. "
                "Please try again later. / "
                "Xin lỗi, tôi không thể xử lý truy vấn của bạn. "
                "Vui lòng thử lại sau."
            )

    @property
    def route_stats(self) -> Dict[str, int]:
        """
        Get routing statistics.

        Returns:
            Dict mapping route names to number of queries routed there
        """
        return {k.value if isinstance(k, ComplexityLevel) else k: v for k, v in self._route_stats.items()}

    @property
    def num_documents(self) -> int:
        """Number of loaded documents (from NaiveRAG pipeline)."""
        if self._naive_rag:
            return self._naive_rag.num_documents
        return 0

    @property
    def num_chunks(self) -> int:
        """Number of chunks (from NaiveRAG pipeline)."""
        if self._naive_rag:
            return self._naive_rag.num_chunks
        return 0

    def clear_route_cache(self):
        """Clear the routing decision cache."""
        self._route_cache.clear()
        logger.debug("Route cache cleared")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AdaptiveRAG("
            f"pipelines_initialized={self._pipelines_initialized}, "
            f"default_route={self._routing_config.default_route.value}, "
            f"stats={self.route_stats})"
        )

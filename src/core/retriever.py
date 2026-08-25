"""
Retriever Manager
=================

Advanced retrieval strategies for RAG pipelines.

Supported strategies:
- Similarity search (basic)
- Hybrid search (vector + BM25) with Vietnamese tokenization
- Re-ranking with cross-encoders (Vietnamese-aware)
- Metadata filtering

Usage:
    retriever = RetrieverManager(vector_store, embeddings)

    # Basic search
    results = retriever.search("Thạch Sanh là ai?", k=5)

    # Hybrid search
    results = retriever.hybrid_search("Thạch Sanh đánh đại bàng", k=5)

    # With re-ranking
    results = retriever.search_with_reranking("câu chuyện Thạch Sanh", k=5)
"""

import hashlib
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Configuration for retriever."""
    search_type: str = "similarity"
    k: int = 4
    score_threshold: Optional[float] = None
    use_hybrid: bool = False
    use_reranking: bool = False
    # Available reranker models:
    #   - AITeamVN/Vietnamese_Reranker (default, best for Vietnamese)
    #   - Qwen/Qwen3-Reranker-0.6B (top open-source, multilingual, 32k context)
    #   - BAAI/bge-reranker-v2-m3 (reliable lightweight multilingual baseline)
    #   - cross-encoder/ms-marco-MiniLM-L-6-v2 (English-only fallback)
    reranker_model: str = "AITeamVN/Vietnamese_Reranker"
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    vector_score_mode: str = "auto"  # "auto", "distance", or "similarity"
    extra: Dict[str, Any] = field(default_factory=dict)


class RetrieverManager:
    """
    Advanced retriever with multiple search strategies.

    Supports:
    - Similarity search (vector-based)
    - Hybrid search (vector + BM25)
    - Re-ranking with cross-encoders
    - MMR (Maximum Marginal Relevance)

    Example:
        from src.core import VectorStoreManager, EmbeddingsManager, RetrieverManager

        # Setup (Vietnamese embeddings by default)
        embeddings = EmbeddingsManager(provider="huggingface")
        store = VectorStoreManager(provider="chroma", embeddings=embeddings)
        retriever = RetrieverManager(vector_store=store, embeddings=embeddings)

        # Basic search
        results = retriever.search("Thạch Sanh là ai?")

        # Hybrid search
        results = retriever.hybrid_search("Thạch Sanh đánh đại bàng")

        # With re-ranking
        results = retriever.search_with_reranking("câu chuyện Thạch Sanh")
    """

    # Supported reranker models
    RERANKER_MODELS = {
        "AITeamVN/Vietnamese_Reranker": {
            "description": "Vietnamese-specific reranker, best for Vietnamese content (default)",
            "context_length": 512,
        },
        "Qwen/Qwen3-Reranker-0.6B": {
            "description": "Top open-source multilingual reranker, 32k context, excellent quality",
            "context_length": 32768,
        },
        "BAAI/bge-reranker-v2-m3": {
            "description": "Reliable lightweight multilingual baseline, good balance of speed and quality",
            "context_length": 8192,
        },
        "cross-encoder/ms-marco-MiniLM-L-6-v2": {
            "description": "English-only cross-encoder, fast fallback",
            "context_length": 512,
        },
    }

    def __init__(
        self,
        vector_store,
        embeddings=None,
        documents: Optional[List[Document]] = None,
        k: int = 4,
        use_hybrid: bool = False,
        use_reranking: bool = False,
    ):
        """
        Initialize retriever manager.

        Args:
            vector_store: VectorStoreManager instance
            embeddings: EmbeddingsManager instance
            documents: Documents for BM25 (required for hybrid search)
            k: Number of results to return
            use_hybrid: Enable hybrid search by default
            use_reranking: Enable re-ranking by default
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.documents = documents or []

        self.config = RetrieverConfig(
            k=k,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
        )

        # Initialize BM25 for hybrid search
        self._bm25 = None
        if use_hybrid and documents:
            self._init_bm25(documents)

        # Initialize re-ranker
        self._reranker = None
        if use_reranking:
            self._init_reranker()

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 with Vietnamese word segmentation support.

        Vietnamese is monosyllabic but compound words span multiple syllables.
        Word segmentation significantly improves BM25 quality for Vietnamese.
        """
        import re

        text = text.lower().strip()

        # Try Vietnamese word segmentation first
        try:
            from underthesea import word_tokenize
            tokens = word_tokenize(text, format="text")
            return [t.strip() for t in tokens.split() if t.strip()]
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"underthesea tokenization failed, falling back: {e}")

        # Fallback: basic regex
        return re.findall(r'\w+', text)

    def _init_bm25(self, documents: List[Document]):
        """Initialize BM25 retriever with Vietnamese-aware tokenization."""
        try:
            from rank_bm25 import BM25Okapi

            # Tokenize documents with Vietnamese support
            tokenized_docs = [
                self._tokenize_for_bm25(doc.page_content)
                for doc in documents
            ]

            self._bm25 = BM25Okapi(tokenized_docs)
            self._bm25_docs = documents
            logger.info(f"BM25 initialized with {len(documents)} documents")
        except ImportError:
            logger.warning("rank-bm25 not installed. Hybrid search disabled.")

    def _init_reranker(self):
        """Initialize cross-encoder re-ranker with Vietnamese model support."""
        try:
            from sentence_transformers import CrossEncoder

            model_name = self.config.reranker_model
            try:
                self._reranker = CrossEncoder(model_name)
                logger.info(f"Reranker initialized: {model_name}")
            except Exception as e:
                # Fallback to a multilingual model if Vietnamese reranker unavailable
                fallback = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                logger.warning(
                    f"Failed to load reranker '{model_name}': {e}. "
                    f"Falling back to '{fallback}'"
                )
                try:
                    self._reranker = CrossEncoder(fallback)
                except Exception:
                    logger.error("Failed to load fallback reranker")
        except ImportError:
            logger.warning("sentence-transformers not installed. Re-ranking disabled.")

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Basic similarity search.

        Args:
            query: Search query
            k: Number of results (overrides config)
            filter: Metadata filter

        Returns:
            List of relevant Document objects
        """
        k = k or self.config.k

        if self.config.use_hybrid and self.config.use_reranking:
            initial_k = kwargs.pop("initial_k", k * 4)
            candidates = self.hybrid_search(query, k=initial_k, filter=filter)
            return self._rerank_documents(query, candidates, k=k, filter=filter)
        elif self.config.use_hybrid:
            return self.hybrid_search(query, k=k, filter=filter)
        elif self.config.use_reranking:
            return self.search_with_reranking(query, k=k, filter=filter)
        else:
            return self.vector_store.similarity_search(
                query, k=k, filter=filter, **kwargs
            )

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search with relevance scores.

        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter

        Returns:
            List of (Document, score) tuples
        """
        k = k or self.config.k

        return self.vector_store.similarity_search_with_score(
            query, k=k, filter=filter
        )

    def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None
    ) -> List[Document]:
        """
        Hybrid search combining vector and BM25.

        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores

        Returns:
            List of relevant Document objects
        """
        k = k or self.config.k
        bm25_weight = bm25_weight or self.config.bm25_weight
        vector_weight = vector_weight or self.config.vector_weight

        if not self._bm25:
            print("Warning: BM25 not initialized. Falling back to vector search.")
            return self.vector_store.similarity_search(query, k=k, filter=filter)

        import numpy as np

        # BM25 search with Vietnamese-aware tokenization
        tokenized_query = self._tokenize_for_bm25(query)
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Get top BM25 results
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k * 2]
        bm25_results = [
            (self._bm25_docs[i], bm25_scores[i])
            for i in top_bm25_indices
        ]

        # Vector search
        vector_results = self.vector_store.similarity_search_with_score(
            query, k=k * 2, filter=filter
        )

        # Normalize scores
        bm25_scores_normalized = self._normalize_scores(
            [score for _, score in bm25_results]
        )
        vector_scores_normalized = self._normalize_vector_scores(vector_results)

        # Combine results using stable document keys for deduplication.
        combined_scores = {}

        for i, (doc, _) in enumerate(bm25_results):
            doc_key = self._document_key(doc)
            combined_scores[doc_key] = {
                "doc": doc,
                "score": bm25_weight * bm25_scores_normalized[i],
            }

        for i, (doc, _) in enumerate(vector_results):
            doc_key = self._document_key(doc)
            if doc_key in combined_scores:
                combined_scores[doc_key]["score"] += (
                    vector_weight * vector_scores_normalized[i]
                )
            else:
                combined_scores[doc_key] = {
                    "doc": doc,
                    "score": vector_weight * vector_scores_normalized[i],
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [item["doc"] for item in sorted_results[:k]]

    def search_with_reranking(
        self,
        query: str,
        k: Optional[int] = None,
        initial_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search with cross-encoder re-ranking.

        First retrieves more candidates, then re-ranks for better precision.

        Args:
            query: Search query
            k: Number of final results
            initial_k: Number of initial candidates (default: k * 4)
            filter: Metadata filter

        Returns:
            List of re-ranked Document objects
        """
        k = k or self.config.k
        initial_k = initial_k or (k * 4)

        if not self._reranker:
            print("Warning: Re-ranker not initialized. Falling back to basic search.")
            return self.vector_store.similarity_search(query, k=k, filter=filter)

        # Get more candidates
        candidates = self.vector_store.similarity_search(
            query, k=initial_k, filter=filter
        )

        return self._rerank_documents(query, candidates, k=k, filter=filter)

    def _rerank_documents(
        self,
        query: str,
        candidates: List[Document],
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Re-rank an existing candidate set with the configured cross-encoder.

        This is used by both vector-only reranking and hybrid+reranking so the
        two retrieval improvements can stack instead of being mutually exclusive.
        """
        k = k or self.config.k

        if not candidates:
            return []

        if not self._reranker:
            logger.warning("Re-ranker not initialized. Returning original candidates.")
            return candidates[:k]

        if filter:
            candidates = self._filter_documents(candidates, filter)

        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self._reranker.predict(pairs)

        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_candidates[:k]]

    def mmr_search(
        self,
        query: str,
        k: Optional[int] = None,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Maximum Marginal Relevance search.

        Balances relevance and diversity in results.

        Args:
            query: Search query
            k: Number of results
            lambda_mult: Lambda multiplier (0 = max diversity, 1 = max relevance)
            filter: Metadata filter

        Returns:
            List of diverse yet relevant Document objects
        """
        k = k or self.config.k

        retriever = self.vector_store.get_retriever(
            search_type="mmr",
            k=k,
            lambda_mult=lambda_mult,
        )

        return retriever.invoke(query)

    def multi_query_search(
        self,
        query: str,
        num_queries: int = 3,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search with multiple query variations for better recall.

        Args:
            query: Original query
            num_queries: Number of query variations
            k: Number of results per query

        Returns:
            List of unique Document objects
        """
        k = k or self.config.k

        # Generate query variations
        queries = self._generate_query_variations(query, num_queries)

        # Search with each query
        all_results = []
        seen_contents = set()

        for q in queries:
            results = self.vector_store.similarity_search(q, k=k)

            for doc in results:
                if doc.page_content not in seen_contents:
                    all_results.append(doc)
                    seen_contents.add(doc.page_content)

        return all_results[:k * 2]  # Return more results for diversity

    def _generate_query_variations(
        self,
        query: str,
        num_variations: int
    ) -> List[str]:
        """Generate query variations for multi-query search.

        Supports both Vietnamese and English queries.
        """
        variations = [query]

        # Detect if query is Vietnamese
        is_vietnamese = any(
            c in "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
            for c in query.lower()
        )

        if is_vietnamese:
            # Vietnamese variations
            if not query.endswith("?"):
                variations.append(f"{query}?")
            if not query.lower().startswith(("là gì", "thế nào", "tại sao", "ở đâu", "khi nào")):
                variations.append(f"{query} là gì?")
            variations.append(f"Giải thích về {query}")
        else:
            # English variations
            if not query.endswith("?"):
                variations.append(f"{query}?")
            if not query.lower().startswith(("what", "how", "why", "when", "where")):
                variations.append(f"What is {query}?")
            variations.append(f"Explain {query}")

        return variations[:num_variations + 1]

    def generate_query_variations_llm(
        self,
        query: str,
        num_variations: int = 3,
        llm=None,
    ) -> List[str]:
        """
        Use LLM to generate query variations for better recall.

        Unlike the template-based _generate_query_variations, this uses
        an LLM to create semantically diverse phrasings.

        Args:
            query: Original query
            num_variations: Number of variations to generate
            llm: LLMManager instance

        Returns:
            List of query variations including the original
        """
        if llm is None:
            # Fallback to template-based variations
            return self._generate_query_variations(query, num_variations)

        try:
            prompt = f"""Generate {num_variations} alternative search queries for the following question.
Each query should express the same intent but use different words or phrasing.
Tạo {num_variations} truy vấn tìm kiếm thay thế cho câu hỏi sau.
Mỗi truy vấn phải thể hiện cùng ý nghĩa nhưng dùng từ khác nhau.

Original / Gốc: {query}

Return one query per line, nothing else.
Mỗi dòng một truy vấn, không thêm gì khác."""

            response = llm.generate(prompt)
            variations = [
                line.strip() for line in response.strip().split("\n")
                if line.strip() and line.strip() != query
            ]

            # Include original query
            return [query] + variations[:num_variations]
        except Exception as e:
            logger.warning(f"LLM query generation failed, using templates: {e}")
            return self._generate_query_variations(query, num_variations)

    def multi_query_rrf_search(
        self,
        query: str,
        num_queries: int = 3,
        k: Optional[int] = None,
        llm=None,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Multi-query search with Reciprocal Rank Fusion (RRF).

        Generates multiple query variations (LLM or template-based),
        retrieves for each, and fuses results using RRF for superior
        recall compared to simple deduplication.

        RRF score = sum(1 / (rrf_k + rank_i)) for each result list.

        Args:
            query: Original query
            num_queries: Number of query variations
            k: Number of results per query
            llm: LLMManager for generating variations (optional)
            rrf_k: RRF constant (default 60, standard value)

        Returns:
            List of Document objects ranked by RRF score
        """
        k = k or self.config.k

        # Generate query variations
        if llm:
            queries = self.generate_query_variations_llm(query, num_queries, llm)
        else:
            queries = self._generate_query_variations(query, num_queries)

        # Retrieve for each query
        result_lists = []
        for q in queries:
            results = self.vector_store.similarity_search_with_score(q, k=k)
            result_lists.append(results)

        # RRF fusion
        return self._rrf_fusion(result_lists, k=k, rrf_k=rrf_k)

    def _rrf_fusion(
        self,
        result_lists: List[List[Tuple[Document, float]]],
        k: int = 5,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion across multiple result lists.

        Combines results from multiple retrieval runs using RRF scoring:
        score(d) = sum(1 / (k + rank_i)) for each list where d appears.

        Args:
            result_lists: List of (Document, score) lists from different queries
            k: Number of final results
            rrf_k: RRF constant (higher = less weight on rank position)

        Returns:
            List of Document objects ranked by RRF score
        """
        # Track RRF scores per document (by content hash)
        rrf_scores: Dict[int, float] = {}
        doc_map: Dict[int, Document] = {}

        for result_list in result_lists:
            for rank, (doc, _score) in enumerate(result_list):
                doc_key = hash(doc.page_content[:200])
                rrf_score = 1.0 / (rrf_k + rank + 1)

                if doc_key in rrf_scores:
                    rrf_scores[doc_key] += rrf_score
                else:
                    rrf_scores[doc_key] = rrf_score
                    doc_map[doc_key] = doc

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        return [doc_map[key] for key in sorted_keys[:k]]

    def hyde_search(
        self,
        query: str,
        k: Optional[int] = None,
        llm=None,
    ) -> List[Document]:
        """
        HyDE (Hypothetical Document Embeddings) search.

        Instead of embedding the query directly, generates a hypothetical
        answer using the LLM, then embeds that answer for retrieval.
        This bridges the embedding gap between short queries and
        longer document passages.

        Args:
            query: Search query
            k: Number of results
            llm: LLMManager for generating hypothetical answer

        Returns:
            List of relevant Document objects
        """
        k = k or self.config.k

        if llm is None:
            logger.warning("HyDE requires an LLM. Falling back to standard search.")
            return self.search(query, k=k)

        try:
            # Generate hypothetical answer
            prompt = f"""Write a short, factual passage that answers the following question.
Viết một đoạn văn ngắn, chính xác trả lời câu hỏi sau.

Question / Câu hỏi: {query}

Passage / Đoạn văn:"""
            hypothetical = llm.generate(prompt).strip()

            # Search using the hypothetical answer as the query
            return self.vector_store.similarity_search(hypothetical, k=k)
        except Exception as e:
            logger.warning(f"HyDE generation failed, falling back to standard search: {e}")
            return self.search(query, k=k)

    def parent_child_search(
        self,
        query: str,
        k: Optional[int] = None,
        parent_chunks: Optional[List[Document]] = None,
    ) -> List[Document]:
        """
        Parent-Child retrieval pattern.

        Searches using small child chunks (precise matching) but returns
        the corresponding parent chunks (rich context for generation).

        Requires that child chunks have "parent_id" in their metadata
        and that parent_chunks are provided or stored.

        Args:
            query: Search query
            k: Number of parent chunks to return
            parent_chunks: List of parent Document objects to retrieve from

        Returns:
            List of parent Document objects
        """
        k = k or self.config.k

        # Search with child chunks (retrieve more to ensure parent coverage)
        child_results = self.vector_store.similarity_search(query, k=k * 3)

        if not child_results:
            return []

        # If parent_chunks provided, look up parents
        if parent_chunks:
            parent_map = {
                p.metadata.get("parent_id", ""): p for p in parent_chunks
            }

            seen_parents = set()
            parent_results = []

            for child in child_results:
                parent_id = child.metadata.get("parent_id", "")
                if parent_id and parent_id not in seen_parents:
                    parent = parent_map.get(parent_id)
                    if parent:
                        parent_results.append(parent)
                        seen_parents.add(parent_id)

            return parent_results[:k]

        # If no parent_chunks provided, return child results as-is
        # (caller can look up parents by parent_id in metadata)
        return child_results[:k]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    def _normalize_vector_scores(
        self,
        vector_results: List[Tuple[Document, float]]
    ) -> List[float]:
        """
        Normalize vector-store scores into relevance scores where higher is better.

        LangChain vector stores are not consistent here: FAISS/Chroma commonly
        return distances where lower is better, while some remote stores return
        similarity scores where higher is better. Hybrid search needs relevance
        scores before combining with BM25.
        """
        scores = [score for _, score in vector_results]
        if not scores:
            return []

        mode = self._infer_vector_score_mode()
        if mode == "distance":
            return self._normalize_distance_scores(scores)

        return self._normalize_scores(scores)

    def _normalize_distance_scores(self, scores: List[float]) -> List[float]:
        """Normalize distance scores to relevance scores where lower distance wins."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [
            (max_score - score) / (max_score - min_score)
            for score in scores
        ]

    def _infer_vector_score_mode(self) -> str:
        """Infer whether vector-store scores are distances or similarities."""
        mode = self.config.vector_score_mode.lower()
        if mode in ("distance", "similarity"):
            return mode

        provider = getattr(getattr(self.vector_store, "config", None), "provider", "")
        provider = (provider or "").lower()

        if provider in ("faiss", "chroma"):
            return "distance"
        if provider in ("qdrant",):
            return "similarity"

        return "similarity"

    def _document_key(self, doc: Document) -> str:
        """Build a stable key for deduplicating retrieved documents."""
        metadata = doc.metadata or {}
        key_parts = [
            str(metadata.get("source", "")),
            str(metadata.get("page_number", "")),
            str(metadata.get("start_index", "")),
            doc.page_content[:500],
        ]
        raw_key = "\n".join(key_parts)
        return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()

    def _filter_documents(
        self,
        docs: List[Document],
        filter: Dict[str, Any],
    ) -> List[Document]:
        """Apply simple equality metadata filters to in-memory candidates."""
        if not filter:
            return docs

        filtered = []
        for doc in docs:
            metadata = doc.metadata or {}
            if all(metadata.get(key) == value for key, value in filter.items()):
                filtered.append(doc)
        return filtered

    def get_retriever(self, search_type: str = "similarity", **kwargs):
        """
        Get a retriever interface for use with LangChain.

        Args:
            search_type: Type of search
            **kwargs: Additional parameters

        Returns:
            Retriever instance
        """
        return self.vector_store.get_retriever(
            search_type=search_type,
            k=self.config.k,
            **kwargs
        )

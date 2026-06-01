"""
Retriever Manager
=================

Advanced retrieval strategies for RAG pipelines.

Supported strategies:
- Similarity search (basic)
- Hybrid search (vector + BM25)
- Re-ranking with cross-encoders
- Metadata filtering

Usage:
    retriever = RetrieverManager(vector_store, embeddings)

    # Basic search
    results = retriever.search("query", k=5)

    # Hybrid search
    results = retriever.hybrid_search("query", k=5)

    # With re-ranking
    results = retriever.search_with_reranking("query", k=5)
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class RetrieverConfig:
    """Configuration for retriever."""
    search_type: str = "similarity"
    k: int = 4
    score_threshold: Optional[float] = None
    use_hybrid: bool = False
    use_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
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

        # Setup
        embeddings = EmbeddingsManager(provider="huggingface")
        store = VectorStoreManager(provider="chroma", embeddings=embeddings)
        retriever = RetrieverManager(vector_store=store, embeddings=embeddings)

        # Basic search
        results = retriever.search("What is Python?")

        # Hybrid search
        results = retriever.hybrid_search("What is Python?")

        # With re-ranking
        results = retriever.search_with_reranking("What is Python?")
    """

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

    def _init_bm25(self, documents: List[Document]):
        """Initialize BM25 retriever."""
        try:
            from rank_bm25 import BM25Okapi
            import re

            # Tokenize documents
            tokenized_docs = [
                re.findall(r'\w+', doc.page_content.lower())
                for doc in documents
            ]

            self._bm25 = BM25Okapi(tokenized_docs)
            self._bm25_docs = documents
        except ImportError:
            print("Warning: rank-bm25 not installed. Hybrid search disabled.")

    def _init_reranker(self):
        """Initialize cross-encoder re-ranker."""
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.config.reranker_model)
        except ImportError:
            print("Warning: sentence-transformers not installed. Re-ranking disabled.")

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

        if self.config.use_hybrid:
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

        import re
        import numpy as np

        # BM25 search
        tokenized_query = re.findall(r'\w+', query.lower())
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
        vector_scores_normalized = self._normalize_scores(
            [score for _, score in vector_results]
        )

        # Combine results
        combined_scores = {}

        for i, (doc, _) in enumerate(bm25_results):
            doc_id = id(doc)
            combined_scores[doc_id] = {
                "doc": doc,
                "score": bm25_weight * bm25_scores_normalized[i],
            }

        for i, (doc, _) in enumerate(vector_results):
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]["score"] += (
                    vector_weight * vector_scores_normalized[i]
                )
            else:
                combined_scores[doc_id] = {
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

        if not candidates:
            return []

        # Create query-document pairs for re-ranking
        pairs = [(query, doc.page_content) for doc in candidates]

        # Get re-ranking scores
        scores = self._reranker.predict(pairs)

        # Sort by re-ranking score
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
        """Generate query variations for multi-query search."""
        # Simple variations (could use LLM for better variations)
        variations = [query]

        # Add question variations
        if not query.endswith("?"):
            variations.append(f"{query}?")

        # Add "What is" prefix
        if not query.lower().startswith(("what", "how", "why", "when", "where")):
            variations.append(f"What is {query}?")

        # Add "Explain" prefix
        variations.append(f"Explain {query}")

        return variations[:num_variations + 1]

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

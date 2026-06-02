"""
Cache
=====

Caching utilities for RAG framework.

Features:
- In-memory caching with TTL and LRU eviction
- Query result caching
- Semantic caching (embedding-based similarity matching)

Usage:
    cache = Cache(ttl=3600)
    cache.set("key", "value")
    value = cache.get("key")

    # Semantic cache
    sem_cache = SemanticCache(embeddings, threshold=0.95)
    sem_cache.put(query_embedding, query, answer)
    answer = sem_cache.get(new_query_embedding)
"""

import logging
import time
from typing import Any, Optional, Dict, List, Tuple
from collections import OrderedDict
import hashlib
import json

logger = logging.getLogger(__name__)


class Cache:
    """
    In-memory cache with TTL support.

    Features:
    - TTL (Time To Live) for automatic expiration
    - LRU (Least Recently Used) eviction
    - Maximum size limit
    - JSON serialization for complex objects

    Example:
        cache = Cache(max_size=1000, ttl=3600)

        # Set value
        cache.set("query_result", {"answer": "Python is..."})

        # Get value
        result = cache.get("query_result")

        # Check if exists
        if cache.has("query_result"):
            result = cache.get("query_result")

        # Clear cache
        cache.clear()
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600
    ):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of items
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        if key not in self._cache:
            return default

        # Check TTL
        if self._is_expired(key):
            self.delete(key)
            return default

        # Move to end (most recently used)
        self._cache.move_to_end(key)

        return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove if exists
        if key in self._cache:
            del self._cache[key]

        # Check size limit
        while len(self._cache) >= self.max_size:
            self._evict_oldest()

        # Add to cache
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def delete(self, key: str) -> None:
        """
        Delete value from cache.

        Args:
            key: Cache key
        """
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]

    def has(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and not expired
        """
        if key not in self._cache:
            return False

        if self._is_expired(key):
            self.delete(key)
            return False

        return True

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._timestamps.clear()

    def size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of items in cache
        """
        return len(self._cache)

    def _is_expired(self, key: str) -> bool:
        """Check if item is expired."""
        if key not in self._timestamps:
            return True

        elapsed = time.time() - self._timestamps[key]
        return elapsed > self.ttl

    def _evict_oldest(self) -> None:
        """Evict oldest item (LRU)."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self.delete(oldest_key)

    def get_or_set(
        self,
        key: str,
        factory_func,
        *args,
        **kwargs
    ) -> Any:
        """
        Get value from cache or compute and cache it.

        Args:
            key: Cache key
            factory_func: Function to compute value if not cached
            *args: Arguments for factory function
            **kwargs: Keyword arguments for factory function

        Returns:
            Cached or computed value
        """
        value = self.get(key)

        if value is None:
            value = factory_func(*args, **kwargs)
            self.set(key, value)

        return value

    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """
        Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = "|".join(key_parts)

        return hashlib.md5(key_string.encode()).hexdigest()


class QueryCache:
    """
    Specialized cache for RAG queries.

    Caches query results based on question and context.

    Example:
        cache = QueryCache()

        # Cache query result
        cache.set(
            question="What is Python?",
            result={"answer": "Python is...", "sources": [...]},
            context_hash="abc123"
        )

        # Get cached result
        result = cache.get(
            question="What is Python?",
            context_hash="abc123"
        )
    """

    def __init__(self, ttl: int = 3600):
        """
        Initialize query cache.

        Args:
            ttl: Time to live in seconds
        """
        self.cache = Cache(max_size=500, ttl=ttl)

    def get(
        self,
        question: str,
        context_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached query result.

        Args:
            question: Question string
            context_hash: Optional hash of context

        Returns:
            Cached result or None
        """
        key = self._make_key(question, context_hash)
        return self.cache.get(key)

    def set(
        self,
        question: str,
        result: Dict[str, Any],
        context_hash: Optional[str] = None
    ) -> None:
        """
        Cache query result.

        Args:
            question: Question string
            result: Query result
            context_hash: Optional hash of context
        """
        key = self._make_key(question, context_hash)
        self.cache.set(key, result)

    def has(
        self,
        question: str,
        context_hash: Optional[str] = None
    ) -> bool:
        """
        Check if result is cached.

        Args:
            question: Question string
            context_hash: Optional hash of context

        Returns:
            True if cached
        """
        key = self._make_key(question, context_hash)
        return self.cache.has(key)

    def _make_key(
        self,
        question: str,
        context_hash: Optional[str] = None
    ) -> str:
        """Create cache key."""
        if context_hash:
            return Cache.generate_key(question, context_hash)
        return Cache.generate_key(question)


class SemanticCache:
    """
    Semantic cache using embedding similarity.

    Caches query-answer pairs and retrieves by semantic similarity
    rather than exact match. If a new query is semantically similar
    to a cached query (similarity > threshold), returns the cached answer.

    This dramatically reduces redundant LLM calls for paraphrased questions.

    Example:
        from src.core.embeddings import EmbeddingsManager
        from src.utils.cache import SemanticCache

        embeddings = EmbeddingsManager()
        cache = SemanticCache(embeddings, threshold=0.95)

        # Cache a result
        query_emb = embeddings.embed_query("Thạch Sanh là ai?")
        cache.put(query_emb, "Thạch Sanh là ai?", "Thạch Sanh là...")

        # Later, a similar query hits the cache
        new_emb = embeddings.embed_query("Thạch Sanh là nhân vật nào?")
        result = cache.get(new_emb)  # Returns "Thạch Sanh là..."
    """

    def __init__(
        self,
        embeddings=None,
        threshold: float = 0.95,
        max_size: int = 1000,
        ttl: int = 3600,
    ):
        """
        Initialize semantic cache.

        Args:
            embeddings: EmbeddingsManager instance for computing query embeddings
            threshold: Similarity threshold for cache hit (0.0-1.0)
            max_size: Maximum number of cached entries
            ttl: Time to live in seconds
        """
        self.embeddings = embeddings
        self.threshold = threshold
        self.max_size = max_size
        self.ttl = ttl

        # Store: {key: (query_text, answer, embedding, timestamp)}
        self._cache: Dict[str, Tuple[str, Any, List[float], float]] = {}
        self._access_order: List[str] = []  # For LRU eviction

    def get(self, query_embedding: List[float]) -> Optional[Any]:
        """
        Find cached answer for semantically similar query.

        Args:
            query_embedding: Embedding vector of the new query

        Returns:
            Cached answer if similar query found, None otherwise
        """
        if not self._cache:
            return None

        best_score = 0.0
        best_key = None

        for key, (query_text, answer, cached_embedding, timestamp) in self._cache.items():
            # Check TTL
            if time.time() - timestamp > self.ttl:
                self._remove(key)
                continue

            # Compute cosine similarity
            score = self._cosine_similarity(query_embedding, cached_embedding)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key and best_score >= self.threshold:
            # Move to end (most recently used)
            if best_key in self._access_order:
                self._access_order.remove(best_key)
            self._access_order.append(best_key)

            logger.debug(
                f"Semantic cache hit: score={best_score:.3f}, "
                f"threshold={self.threshold}"
            )
            return self._cache[best_key][1]  # Return answer

        return None

    def get_with_score(
        self, query_embedding: List[float]
    ) -> Optional[Tuple[Any, float]]:
        """
        Find cached answer and return with similarity score.

        Args:
            query_embedding: Embedding vector of the new query

        Returns:
            Tuple of (answer, similarity_score) if found, None otherwise
        """
        if not self._cache:
            return None

        best_score = 0.0
        best_key = None

        for key, (query_text, answer, cached_embedding, timestamp) in self._cache.items():
            if time.time() - timestamp > self.ttl:
                self._remove(key)
                continue

            score = self._cosine_similarity(query_embedding, cached_embedding)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key and best_score >= self.threshold:
            if best_key in self._access_order:
                self._access_order.remove(best_key)
            self._access_order.append(best_key)
            return (self._cache[best_key][1], best_score)

        return None

    def put(
        self,
        query_embedding: List[float],
        query_text: str,
        answer: Any,
    ) -> None:
        """
        Cache a query-answer pair.

        Args:
            query_embedding: Embedding vector of the query
            query_text: Original query text (for debugging)
            answer: The answer to cache
        """
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            self._evict_lru()

        key = hashlib.md5(query_text.encode()).hexdigest()
        self._cache[key] = (query_text, answer, query_embedding, time.time())

        if key not in self._access_order:
            self._access_order.append(key)

    def _remove(self, key: str) -> None:
        """Remove a cache entry."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if self._access_order:
            oldest_key = self._access_order[0]
            self._remove(oldest_key)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "threshold": self.threshold,
            "ttl": self.ttl,
        }

"""
Cache
=====

Caching utilities for RAG framework.

Features:
- In-memory caching
- TTL support
- LRU eviction

Usage:
    cache = Cache(ttl=3600)
    cache.set("key", "value")
    value = cache.get("key")
"""

import time
from typing import Any, Optional, Dict
from collections import OrderedDict
import hashlib
import json


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

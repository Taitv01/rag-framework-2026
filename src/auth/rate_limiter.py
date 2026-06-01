"""
Rate Limiter
============

Rate limiting for API endpoints.

Features:
- Per-user rate limiting
- Per-endpoint rate limiting
- Sliding window algorithm

Usage:
    from src.auth import RateLimiter

    limiter = RateLimiter(max_requests=100, window_seconds=60)

    if limiter.is_allowed("user_123"):
        # Process request
        pass
    else:
        # Rate limit exceeded
        pass
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time


class RateLimiter:
    """
    Rate limiter using sliding window algorithm.

    Example:
        limiter = RateLimiter(max_requests=100, window_seconds=60)

        # Check if request is allowed
        if limiter.is_allowed("user_123"):
            # Process request
            pass
        else:
            # Rate limit exceeded
            return {"error": "Rate limit exceeded"}

        # Get remaining requests
        remaining = limiter.get_remaining("user_123")
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed.

        Args:
            key: Rate limit key (e.g., user_id)

        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[key] = [
            ts for ts in self.requests[key]
            if ts > window_start
        ]

        # Check if under limit
        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return True

        return False

    def get_remaining(self, key: str) -> int:
        """
        Get remaining requests in window.

        Args:
            key: Rate limit key

        Returns:
            Number of remaining requests
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[key] = [
            ts for ts in self.requests[key]
            if ts > window_start
        ]

        return max(0, self.max_requests - len(self.requests[key]))

    def get_reset_time(self, key: str) -> Optional[float]:
        """
        Get time until rate limit resets.

        Args:
            key: Rate limit key

        Returns:
            Seconds until reset, or None if not limited
        """
        if self.get_remaining(key) > 0:
            return None

        # Find oldest request in window
        if self.requests[key]:
            oldest = min(self.requests[key])
            return self.window_seconds - (time.time() - oldest)

        return 0

    def reset(self, key: str) -> None:
        """
        Reset rate limit for key.

        Args:
            key: Rate limit key
        """
        if key in self.requests:
            del self.requests[key]

    def reset_all(self) -> None:
        """Reset all rate limits."""
        self.requests.clear()

    def get_stats(self, key: str) -> Dict:
        """
        Get rate limit statistics.

        Args:
            key: Rate limit key

        Returns:
            Statistics dictionary
        """
        remaining = self.get_remaining(key)
        reset_time = self.get_reset_time(key)

        return {
            "key": key,
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "remaining": remaining,
            "used": self.max_requests - remaining,
            "reset_in_seconds": reset_time,
        }


class MultiKeyRateLimiter:
    """
    Rate limiter with different limits for different keys.

    Example:
        limiter = MultiKeyRateLimiter()

        # Set different limits
        limiter.set_limit("user_123", max_requests=100, window_seconds=60)
        limiter.set_limit("user_456", max_requests=50, window_seconds=60)

        # Check
        if limiter.is_allowed("user_123"):
            pass
    """

    def __init__(self):
        """Initialize multi-key rate limiter."""
        self.limiters: Dict[str, RateLimiter] = {}
        self.default_limiter = RateLimiter(max_requests=100, window_seconds=60)

    def set_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60
    ) -> None:
        """
        Set rate limit for key.

        Args:
            key: Rate limit key
            max_requests: Maximum requests
            window_seconds: Window duration
        """
        self.limiters[key] = RateLimiter(max_requests, window_seconds)

    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed.

        Args:
            key: Rate limit key

        Returns:
            True if allowed
        """
        limiter = self.limiters.get(key, self.default_limiter)
        return limiter.is_allowed(key)

    def get_remaining(self, key: str) -> int:
        """
        Get remaining requests.

        Args:
            key: Rate limit key

        Returns:
            Remaining requests
        """
        limiter = self.limiters.get(key, self.default_limiter)
        return limiter.get_remaining(key)

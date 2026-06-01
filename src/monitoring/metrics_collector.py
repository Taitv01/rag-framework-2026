"""
Metrics Collector
=================

Collect and analyze usage metrics.

Features:
- Query logging
- Response time tracking
- Token usage tracking
- Error tracking
- Analytics generation

Usage:
    from src.monitoring import MetricsCollector

    metrics = MetricsCollector()

    # Track query
    metrics.track_query(
        question="What is Python?",
        response_time=1.2,
        tokens_used=150,
        user_id="user_123"
    )

    # Get analytics
    stats = metrics.get_analytics(period="7d")
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json


@dataclass
class QueryMetric:
    """Metric for a single query."""
    question: str
    response_time: float
    tokens_used: int
    sources_count: int
    user_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Analytics:
    """Analytics summary."""
    total_queries: int
    avg_response_time: float
    total_tokens: int
    avg_tokens_per_query: float
    unique_users: int
    top_topics: List[Dict[str, Any]]
    error_count: int
    period: str


class MetricsCollector:
    """
    Metrics collector for tracking usage.

    Example:
        metrics = MetricsCollector()

        # Track queries
        metrics.track_query(
            question="What is Python?",
            response_time=1.2,
            tokens_used=150
        )

        # Get analytics
        stats = metrics.get_analytics(period="7d")
        print(f"Total queries: {stats.total_queries}")
        print(f"Avg response time: {stats.avg_response_time}s")
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize metrics collector.

        Args:
            storage_path: Path for storing metrics
        """
        self.storage_path = storage_path
        self.metrics: List[QueryMetric] = []
        self.errors: List[Dict[str, Any]] = []

        # Load existing metrics
        if storage_path:
            self._load_metrics()

    def track_query(
        self,
        question: str,
        response_time: float,
        tokens_used: int = 0,
        sources_count: int = 0,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track a query.

        Args:
            question: User question
            response_time: Response time in seconds
            tokens_used: Tokens used
            sources_count: Number of sources retrieved
            user_id: User ID
            metadata: Additional metadata
        """
        metric = QueryMetric(
            question=question,
            response_time=response_time,
            tokens_used=tokens_used,
            sources_count=sources_count,
            user_id=user_id,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        self.metrics.append(metric)
        self._save_metrics()

    def track_error(
        self,
        error: str,
        question: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track an error.

        Args:
            error: Error message
            question: User question
            user_id: User ID
            metadata: Additional metadata
        """
        self.errors.append({
            "error": error,
            "question": question,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })

        self._save_metrics()

    def get_analytics(
        self,
        period: str = "7d",
        user_id: Optional[str] = None
    ) -> Analytics:
        """
        Get analytics for period.

        Args:
            period: Time period ('1d', '7d', '30d', 'all')
            user_id: Filter by user ID

        Returns:
            Analytics summary
        """
        # Filter by time period
        if period == "all":
            filtered = self.metrics
        else:
            days = int(period.replace("d", ""))
            cutoff = datetime.now() - timedelta(days=days)
            filtered = [
                m for m in self.metrics
                if m.timestamp > cutoff
            ]

        # Filter by user
        if user_id:
            filtered = [m for m in filtered if m.user_id == user_id]

        # Calculate statistics
        total_queries = len(filtered)

        if total_queries == 0:
            return Analytics(
                total_queries=0,
                avg_response_time=0,
                total_tokens=0,
                avg_tokens_per_query=0,
                unique_users=0,
                top_topics=[],
                error_count=len(self.errors),
                period=period,
            )

        avg_response_time = sum(m.response_time for m in filtered) / total_queries
        total_tokens = sum(m.tokens_used for m in filtered)
        avg_tokens = total_tokens / total_queries

        # Unique users
        unique_users = len(set(m.user_id for m in filtered if m.user_id))

        # Top topics (simple word frequency)
        top_topics = self._get_top_topics(filtered)

        # Error count for period
        if period == "all":
            error_count = len(self.errors)
        else:
            days = int(period.replace("d", ""))
            cutoff = datetime.now() - timedelta(days=days)
            error_count = len([
                e for e in self.errors
                if datetime.fromisoformat(e["timestamp"]) > cutoff
            ])

        return Analytics(
            total_queries=total_queries,
            avg_response_time=round(avg_response_time, 2),
            total_tokens=total_tokens,
            avg_tokens_per_query=round(avg_tokens, 2),
            unique_users=unique_users,
            top_topics=top_topics,
            error_count=error_count,
            period=period,
        )

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for specific user.

        Args:
            user_id: User ID

        Returns:
            User statistics
        """
        user_metrics = [m for m in self.metrics if m.user_id == user_id]

        if not user_metrics:
            return {"user_id": user_id, "total_queries": 0}

        total_queries = len(user_metrics)
        avg_response_time = sum(m.response_time for m in user_metrics) / total_queries
        total_tokens = sum(m.tokens_used for m in user_metrics)

        return {
            "user_id": user_id,
            "total_queries": total_queries,
            "avg_response_time": round(avg_response_time, 2),
            "total_tokens": total_tokens,
            "first_query": min(m.timestamp for m in user_metrics).isoformat(),
            "last_query": max(m.timestamp for m in user_metrics).isoformat(),
        }

    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """
        Get recent queries.

        Args:
            limit: Number of queries to return

        Returns:
            List of recent query records
        """
        recent = sorted(self.metrics, key=lambda m: m.timestamp, reverse=True)[:limit]

        return [
            {
                "question": m.question,
                "response_time": m.response_time,
                "tokens_used": m.tokens_used,
                "user_id": m.user_id,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in recent
        ]

    def _get_top_topics(self, metrics: List[QueryMetric], top_n: int = 5) -> List[Dict]:
        """Get top topics from queries."""
        # Simple word frequency
        word_freq: Dict[str, int] = {}

        for metric in metrics:
            words = metric.question.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return [
            {"topic": word, "count": count}
            for word, count in sorted_words[:top_n]
        ]

    def _save_metrics(self) -> None:
        """Save metrics to storage."""
        if not self.storage_path:
            return

        data = {
            "metrics": [
                {
                    "question": m.question,
                    "response_time": m.response_time,
                    "tokens_used": m.tokens_used,
                    "sources_count": m.sources_count,
                    "user_id": m.user_id,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata,
                }
                for m in self.metrics
            ],
            "errors": self.errors,
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_metrics(self) -> None:
        """Load metrics from storage."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for m in data.get("metrics", []):
                self.metrics.append(QueryMetric(
                    question=m["question"],
                    response_time=m["response_time"],
                    tokens_used=m.get("tokens_used", 0),
                    sources_count=m.get("sources_count", 0),
                    user_id=m.get("user_id"),
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                    metadata=m.get("metadata", {}),
                ))

            self.errors = data.get("errors", [])
        except FileNotFoundError:
            pass

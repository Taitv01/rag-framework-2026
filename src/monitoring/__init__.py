"""
Monitoring & Analytics
=====================

Usage tracking and performance monitoring.

Features:
- Query logging
- Response time tracking
- Token usage tracking
- Error rate monitoring
- Usage dashboard

Usage:
    from src.monitoring import MetricsCollector

    metrics = MetricsCollector()
    metrics.track_query(question, response_time, tokens_used)
    stats = metrics.get_analytics(period="7d")
"""

from src.monitoring.metrics_collector import MetricsCollector

__all__ = ["MetricsCollector"]

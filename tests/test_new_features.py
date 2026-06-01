"""
Tests for New Features
======================

Comprehensive tests for all new modules.
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestStreamingModule:
    """Test Streaming module."""

    def test_streaming_buffer(self):
        """Test StreamingBuffer."""
        from src.core.streaming import StreamingBuffer

        buffer = StreamingBuffer(max_size=5)

        # Test add
        buffer.add("Hello")
        buffer.add(" ")
        buffer.add("World")
        assert len(buffer) == 3
        assert not buffer.is_full()

        # Test flush
        content = buffer.flush()
        assert content == "Hello World"
        assert len(buffer) == 0

    def test_streaming_buffer_full(self):
        """Test StreamingBuffer full detection."""
        from src.core.streaming import StreamingBuffer

        buffer = StreamingBuffer(max_size=3)
        buffer.add("a")
        buffer.add("b")
        buffer.add("c")
        assert buffer.is_full()

    def test_streaming_buffer_full_text(self):
        """Test StreamingBuffer full text."""
        from src.core.streaming import StreamingBuffer

        buffer = StreamingBuffer()
        buffer.add("Hello")
        buffer.add(" ")
        buffer.add("World")
        assert buffer.get_full_text() == "Hello World"

    def test_streaming_buffer_clear(self):
        """Test StreamingBuffer clear."""
        from src.core.streaming import StreamingBuffer

        buffer = StreamingBuffer()
        buffer.add("test")
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.get_full_text() == ""


class TestMemoryModule:
    """Test Memory module."""

    def test_conversation_memory(self):
        """Test ConversationMemory."""
        from src.core.memory import ConversationMemory

        memory = ConversationMemory(type="buffer", max_history=10)

        memory.add_user_message("Hello")
        memory.add_ai_message("Hi there!")

        assert memory.message_count == 2

    def test_memory_history(self):
        """Test memory history retrieval."""
        from src.core.memory import ConversationMemory

        memory = ConversationMemory()
        memory.add_user_message("Q1")
        memory.add_ai_message("A1")
        memory.add_user_message("Q2")
        memory.add_ai_message("A2")

        history = memory.get_history()
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Q1"

    def test_memory_context_string(self):
        """Test memory context string."""
        from src.core.memory import ConversationMemory

        memory = ConversationMemory()
        memory.add_user_message("What is Python?")
        memory.add_ai_message("Python is a language.")

        context = memory.get_context_string()
        assert "User: What is Python?" in context
        assert "Assistant: Python is a language." in context

    def test_window_memory(self):
        """Test window memory."""
        from src.core.memory import ConversationMemory

        memory = ConversationMemory(type="window", max_history=2)

        for i in range(5):
            memory.add_user_message(f"Q{i}")
            memory.add_ai_message(f"A{i}")

        history = memory.get_history()
        assert len(history) <= 4  # 2 pairs = 4 messages

    def test_memory_save_load(self):
        """Test memory save/load."""
        from src.core.memory import ConversationMemory

        memory = ConversationMemory()
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi!")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            memory.save(temp_path)

            new_memory = ConversationMemory()
            new_memory.load(temp_path)

            assert new_memory.message_count == 2
        finally:
            os.unlink(temp_path)

    def test_memory_clear(self):
        """Test memory clear."""
        from src.core.memory import ConversationMemory

        memory = ConversationMemory()
        memory.add_user_message("Hello")
        memory.clear()

        assert memory.message_count == 0


class TestDocumentManagement:
    """Test Document Management module."""

    def test_document_manager(self):
        """Test DocumentManager."""
        from src.documents import DocumentManager

        manager = DocumentManager()

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            doc_id = manager.add(temp_path, metadata={"category": "test"})
            assert doc_id is not None
            assert manager.count == 1
        finally:
            os.unlink(temp_path)

    def test_document_list(self):
        """Test document listing."""
        from src.documents import DocumentManager

        manager = DocumentManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test")
            temp_path = f.name

        try:
            manager.add(temp_path, metadata={"category": "test"})
            docs = manager.list(category="test")
            assert len(docs) == 1
        finally:
            os.unlink(temp_path)

    def test_document_delete(self):
        """Test document deletion."""
        from src.documents import DocumentManager

        manager = DocumentManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test")
            temp_path = f.name

        try:
            doc_id = manager.add(temp_path)
            manager.delete(doc_id)
            docs = manager.list()
            assert len(docs) == 0
        finally:
            os.unlink(temp_path)


class TestAuthModule:
    """Test Auth module."""

    def test_api_key_manager(self):
        """Test APIKeyManager."""
        from src.auth import APIKeyManager

        manager = APIKeyManager()

        api_key = manager.create(user_id="user_123", name="Test Key")
        assert api_key.startswith("rag_")

    def test_api_key_verify(self):
        """Test API key verification."""
        from src.auth import APIKeyManager

        manager = APIKeyManager()

        api_key = manager.create(user_id="user_123")
        user_id = manager.verify(api_key)

        assert user_id == "user_123"

    def test_api_key_revoke(self):
        """Test API key revocation."""
        from src.auth import APIKeyManager

        manager = APIKeyManager()

        api_key = manager.create(user_id="user_123")
        manager.revoke(api_key)

        user_id = manager.verify(api_key)
        assert user_id is None

    def test_rate_limiter(self):
        """Test RateLimiter."""
        from src.auth import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=60)

        assert limiter.is_allowed("user_1")
        assert limiter.is_allowed("user_1")
        assert limiter.is_allowed("user_1")
        assert not limiter.is_allowed("user_1")

    def test_rate_limiter_remaining(self):
        """Test rate limiter remaining."""
        from src.auth import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for _ in range(3):
            limiter.is_allowed("user_1")

        remaining = limiter.get_remaining("user_1")
        assert remaining == 2

    def test_rate_limiter_stats(self):
        """Test rate limiter stats."""
        from src.auth import RateLimiter

        limiter = RateLimiter(max_requests=10, window_seconds=60)
        limiter.is_allowed("user_1")

        stats = limiter.get_stats("user_1")
        assert stats["max_requests"] == 10
        assert stats["remaining"] == 9


class TestMonitoringModule:
    """Test Monitoring module."""

    def test_metrics_collector(self):
        """Test MetricsCollector."""
        from src.monitoring import MetricsCollector

        metrics = MetricsCollector()

        metrics.track_query(
            question="What is Python?",
            response_time=1.2,
            tokens_used=150,
            user_id="user_123"
        )

        stats = metrics.get_analytics(period="7d")
        assert stats.total_queries == 1

    def test_metrics_analytics(self):
        """Test metrics analytics."""
        from src.monitoring import MetricsCollector

        metrics = MetricsCollector()

        metrics.track_query("Q1", 1.0, 100, 5, "user_1")
        metrics.track_query("Q2", 2.0, 200, 3, "user_2")

        stats = metrics.get_analytics(period="7d")
        assert stats.total_queries == 2
        assert stats.avg_response_time == 1.5
        assert stats.unique_users == 2

    def test_metrics_user_stats(self):
        """Test user statistics."""
        from src.monitoring import MetricsCollector

        metrics = MetricsCollector()

        metrics.track_query("Q1", 1.0, 100, 5, "user_1")
        metrics.track_query("Q2", 2.0, 200, 3, "user_1")

        user_stats = metrics.get_user_stats("user_1")
        assert user_stats["total_queries"] == 2

    def test_metrics_recent_queries(self):
        """Test recent queries."""
        from src.monitoring import MetricsCollector

        metrics = MetricsCollector()

        metrics.track_query("Q1", 1.0, 100)
        metrics.track_query("Q2", 2.0, 200)

        recent = metrics.get_recent_queries(limit=1)
        assert len(recent) == 1

    def test_metrics_error_tracking(self):
        """Test error tracking."""
        from src.monitoring import MetricsCollector

        metrics = MetricsCollector()

        metrics.track_error("Error occurred", question="Q1", user_id="user_1")

        stats = metrics.get_analytics(period="7d")
        assert stats.error_count == 1

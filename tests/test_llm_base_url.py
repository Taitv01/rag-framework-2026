"""Tests for LLM provider endpoint configuration."""

from types import SimpleNamespace

import pytest


class FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_openai_base_url_is_forwarded(monkeypatch):
    import sys

    from src.core.llm import LLMManager

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatModel))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")

    manager = LLMManager(provider="openai", model="qwen-plus", temperature=0.1)
    llm = manager._create_openai_llm()

    assert llm.kwargs["model"] == "qwen-plus"
    assert llm.kwargs["api_key"] == "test-key"
    assert llm.kwargs["base_url"] == "https://proxy.example/v1"
    assert llm.kwargs["temperature"] == 0.1


def test_anthropic_base_url_is_forwarded(monkeypatch):
    import sys

    from src.core.llm import LLMManager

    monkeypatch.setitem(sys.modules, "langchain_anthropic", SimpleNamespace(ChatAnthropic=FakeChatModel))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic-proxy.example")

    manager = LLMManager(provider="anthropic", model="claude-sonnet-4-20250514")
    llm = manager._create_anthropic_llm()

    assert llm.kwargs["model"] == "claude-sonnet-4-20250514"
    assert llm.kwargs["api_key"] == "test-key"
    assert llm.kwargs["anthropic_api_url"] == "https://anthropic-proxy.example"


def test_ox_uses_openrouter_defaults(monkeypatch):
    import sys

    from src.core.llm import LLMManager

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatModel))
    monkeypatch.setenv("OX_API_KEY", "test-ox-key")
    monkeypatch.delenv("OX_BASE_URL", raising=False)

    manager = LLMManager(provider="ox")
    llm = manager._create_ox_llm()

    assert manager.config.model == "stealth/ox-alpha"
    assert llm.kwargs["model"] == "stealth/ox-alpha"
    assert llm.kwargs["api_key"] == "test-ox-key"
    assert llm.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert llm.kwargs["max_retries"] == 5
    assert llm.kwargs["timeout"] == 180
    assert manager.get_context_window() == 1_048_576
    assert manager.get_max_output_tokens() == 131_072


def test_ox_accepts_alias_and_openrouter_key(monkeypatch):
    import sys

    from src.core.llm import LLMManager

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatModel))
    monkeypatch.setattr("src.core.llm.load_environment", lambda: None)
    monkeypatch.delenv("OX_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OX_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OX_APP_URL", "https://rag.example")
    monkeypatch.setenv("OX_APP_NAME", "RAG Test")
    monkeypatch.setenv("OX_MAX_RETRIES", "7")
    monkeypatch.setenv("OX_TIMEOUT_SECONDS", "90")

    manager = LLMManager(provider="ox-ai", temperature=0.2, max_tokens=2048)
    llm = manager._create_llm()

    assert manager.config.provider == "ox"
    assert llm.kwargs["api_key"] == "test-openrouter-key"
    assert llm.kwargs["base_url"] == "https://gateway.example/v1"
    assert llm.kwargs["default_headers"] == {
        "HTTP-Referer": "https://rag.example",
        "X-OpenRouter-Title": "RAG Test",
    }
    assert llm.kwargs["temperature"] == 0.2
    assert llm.kwargs["max_tokens"] == 2048
    assert llm.kwargs["max_retries"] == 7
    assert llm.kwargs["timeout"] == 90


def test_ox_does_not_fall_back_to_openai_key(monkeypatch):
    import sys

    from src.core.llm import LLMManager

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatModel))
    monkeypatch.setattr("src.core.llm.load_environment", lambda: None)
    monkeypatch.delenv("OX_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-key")

    with pytest.raises(ValueError, match="OX_API_KEY"):
        LLMManager(provider="ox")._create_ox_llm()


def test_ox_retry_settings_are_bounded(monkeypatch):
    from src.core.llm import LLMManager

    monkeypatch.setenv("OX_MAX_RETRIES", "11")
    with pytest.raises(ValueError, match="between 0 and 10"):
        LLMManager._get_ox_max_retries()

    monkeypatch.setenv("OX_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="between 1 and 600"):
        LLMManager._get_ox_timeout()

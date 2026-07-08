"""Tests for LLM provider endpoint configuration."""

from types import SimpleNamespace


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

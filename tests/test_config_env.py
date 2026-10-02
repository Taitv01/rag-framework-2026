"""Tests for environment file precedence."""

from src.utils.config import Config


def test_env_local_overrides_env_file(tmp_path, monkeypatch):
    """Local env files should override committed/default .env values."""
    (tmp_path / ".env").write_text("RAG_TEST_VALUE=from_env\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("RAG_TEST_VALUE=from_local\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_TEST_VALUE", raising=False)
    monkeypatch.delenv("RAG_DISABLE_DOTENV", raising=False)

    config = Config()

    assert config.get("RAG_TEST_VALUE") == "from_local"


def test_dotenv_discovery_can_be_disabled(tmp_path, monkeypatch):
    """RAG_DISABLE_DOTENV keeps local secret files out of the environment."""
    (tmp_path / ".env.local").write_text("RAG_TEST_VALUE=from_local\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_TEST_VALUE", raising=False)
    monkeypatch.setenv("RAG_DISABLE_DOTENV", "1")

    assert Config().get("RAG_TEST_VALUE") is None


def test_model_defaults_follow_provider(monkeypatch):
    """Unset model settings resolve to the configured provider's default."""
    from src.core.embeddings import EmbeddingsManager
    from src.core.llm import LLMManager

    monkeypatch.delenv("DEFAULT_LLM_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "anthropic")

    config = Config()

    assert config.get_llm_config()["model"] == "claude-sonnet-4-20250514"
    assert config.get_embedding_config()["model"] == "BAAI/bge-m3"
    assert LLMManager(provider="anthropic").config.model == "claude-sonnet-4-20250514"
    assert EmbeddingsManager(provider="openai").config.model_name == "text-embedding-3-small"


def test_openrouter_llm_config_prefers_dedicated_key(monkeypatch):
    """OpenRouter must not reuse an unrelated OpenAI key when its own key exists."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "stealth/ox-alpha")

    llm_config = Config().get_llm_config()

    assert llm_config["api_key"] == "openrouter-key"
    assert llm_config["base_url"] == "https://openrouter.ai/api/v1"
    assert llm_config["model"] == "stealth/ox-alpha"

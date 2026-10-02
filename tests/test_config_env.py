"""Tests for environment file precedence."""

from src.utils.config import Config


def test_env_local_overrides_env_file(tmp_path, monkeypatch):
    """Local env files should override committed/default .env values."""
    (tmp_path / ".env").write_text("RAG_TEST_VALUE=from_env\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("RAG_TEST_VALUE=from_local\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_TEST_VALUE", raising=False)

    config = Config()

    assert config.get("RAG_TEST_VALUE") == "from_local"


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

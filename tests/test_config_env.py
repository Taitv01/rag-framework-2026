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

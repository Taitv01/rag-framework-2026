"""Regression tests for safe environment-variable precedence."""

from src.utils.config import Config


def test_process_environment_overrides_local_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("RAG_PRECEDENCE=shared\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("RAG_PRECEDENCE=local\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_PRECEDENCE", "process")

    config = Config()

    assert config.get("RAG_PRECEDENCE") == "process"

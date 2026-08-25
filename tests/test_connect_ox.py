"""Tests for the safe local Ox Alpha connection helper."""

import base64
import hashlib

from scripts.connect_ox import create_pkce_pair, read_ox_key, update_env_file


def test_pkce_pair_uses_s256():
    verifier, challenge = create_pkce_pair()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")

    assert challenge == expected
    assert "=" not in challenge


def test_update_env_file_preserves_other_secrets(tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "OPENAI_API_KEY=existing-openai-key\nOX_API_KEY=old-ox-key\nSETTING=value\n",
        encoding="utf-8",
    )

    update_env_file(env_file, "new-ox-key")

    assert env_file.read_text(encoding="utf-8") == (
        "OPENAI_API_KEY=existing-openai-key\nOX_API_KEY=new-ox-key\nSETTING=value\n"
    )


def test_update_env_file_adds_missing_key_once(tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("SETTING=value\n", encoding="utf-8")

    update_env_file(env_file, "new-ox-key")
    update_env_file(env_file, "newer-ox-key")

    contents = env_file.read_text(encoding="utf-8")
    assert contents.count("OX_API_KEY=") == 1
    assert "OX_API_KEY=newer-ox-key" in contents
    assert read_ox_key(env_file) == "newer-ox-key"


def test_read_ox_key_does_not_fall_back_to_another_provider(tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("OPENAI_API_KEY=paid-key\n", encoding="utf-8")

    assert read_ox_key(env_file) is None

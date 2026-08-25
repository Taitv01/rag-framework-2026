"""Minimal Ox Alpha connectivity and RAG example.

Configure ``OX_API_KEY`` in ``.env.local`` before running this file. The key is
created at https://openrouter.ai/settings/keys and is never stored in source.
"""

import argparse
import io
import sys
from pathlib import Path

# Preserve Vietnamese output on Windows consoles that default to a legacy codepage.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Allow direct execution from the repository root, matching the other examples.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.llm import LLMManager
from src.rag import NaiveRAG
from src.utils.config import Config


def get_ox_key(config: Config) -> str:
    """Return the configured Ox/OpenRouter key or exit with a safe message."""
    api_key = config.get("OX_API_KEY") or config.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OX_API_KEY. Create a free key at "
            "https://openrouter.ai/settings/keys and add it to .env.local."
        )
    return api_key


def check_connection(api_key: str) -> None:
    """Send one small request without loading an embedding model."""
    llm = LLMManager(
        provider="ox",
        model="stealth/ox-alpha",
        api_key=api_key,
        temperature=0,
        # Ox is a reasoning model; leave enough room for hidden reasoning plus
        # the short visible confirmation.
        max_tokens=256,
    )
    answer = llm.generate("Reply with exactly: Ox Alpha connected")
    print(answer)


def run_rag(api_key: str, config: Config) -> None:
    """Run a tiny retrieval-augmented generation flow."""
    rag = NaiveRAG(
        llm_provider="ox",
        llm_model="stealth/ox-alpha",
        llm_api_key=api_key,
        embedding_provider=config.get("DEFAULT_EMBEDDING_PROVIDER", "huggingface"),
        embedding_model=config.get(
            "DEFAULT_EMBEDDING_MODEL", "keepitreal/vietnamese-sbert"
        ),
        vector_store_provider="faiss",
    )
    rag.add_texts(
        [
            "Hà Nội là thủ đô của Việt Nam.",
            "Thành phố Hồ Chí Minh là trung tâm kinh tế lớn ở phía Nam.",
        ]
    )
    print(rag.query("Thủ đô của Việt Nam là thành phố nào?"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Ox Alpha with this RAG project.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only test the Ox API; do not load embeddings or build a vector index.",
    )
    args = parser.parse_args()

    config = Config()
    api_key = get_ox_key(config)
    if args.check:
        check_connection(api_key)
    else:
        run_rag(api_key, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Configuration Manager
====================

Centralized configuration management for RAG framework.

Features:
- Environment variable loading
- Configuration file support
- Default values
- Validation

Usage:
    config = Config()
    api_key = config.get("OPENAI_API_KEY")

    # Or with defaults
    chunk_size = config.get("CHUNK_SIZE", default=500)
"""

import os
from typing import Any, Optional, Dict
from pathlib import Path

from dotenv import load_dotenv


# Single source of truth for per-provider default models. LLMManager and
# EmbeddingsManager fall back to these when no model is given, and Config
# uses them when DEFAULT_LLM_MODEL / DEFAULT_EMBEDDING_MODEL are unset.
DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3",
}
DEFAULT_EMBEDDING_MODELS = {
    "huggingface": "BAAI/bge-m3",
    "openai": "text-embedding-3-small",
    "cohere": "embed-multilingual-v3.0",
}


def default_llm_model(provider: Optional[str]) -> str:
    """Return the default LLM model for a provider."""
    return DEFAULT_LLM_MODELS.get(provider or "openai", DEFAULT_LLM_MODELS["openai"])


def default_embedding_model(provider: Optional[str]) -> str:
    """Return the default embedding model for a provider."""
    return DEFAULT_EMBEDDING_MODELS.get(
        provider or "huggingface", DEFAULT_EMBEDDING_MODELS["huggingface"]
    )


def dotenv_disabled() -> bool:
    """Whether automatic `.env` / `.env.local` discovery is switched off."""
    return os.getenv("RAG_DISABLE_DOTENV", "").strip().lower() in ("1", "true", "yes")


def load_environment(env_file: Optional[str] = None) -> None:
    """
    Load environment files in a predictable order.

    `.env.local` is loaded after `.env` and can override local developer
    secrets such as OPENAI_API_KEY without changing the committed template.

    Set RAG_DISABLE_DOTENV=1 to skip auto-discovery (the test suite does this
    so local secrets never leak into tests). An explicit `env_file` is still
    loaded.
    """
    if env_file:
        load_dotenv(env_file)
        return

    if dotenv_disabled():
        return

    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    local_env_path = Path.cwd() / ".env.local"
    if local_env_path.exists():
        load_dotenv(local_env_path, override=True)


class Config:
    """
    Configuration manager.

    Loads configuration from:
    1. Environment variables
    2. .env file
    3. Default values

    Example:
        config = Config()

        # Get API key
        api_key = config.get("OPENAI_API_KEY")

        # Get with default
        chunk_size = config.get("CHUNK_SIZE", default=500)

        # Get required (raises error if missing)
        api_key = config.get_required("OPENAI_API_KEY")

        # Check if exists
        if config.has("REDIS_URL"):
            redis_url = config.get("REDIS_URL")
    """

    # Default configuration values
    DEFAULTS = {
        # LLM Configuration
        # DEFAULT_LLM_MODEL / DEFAULT_EMBEDDING_MODEL are intentionally absent:
        # when unset they resolve per provider (see DEFAULT_LLM_MODELS).
        "DEFAULT_LLM_PROVIDER": "openai",
        "DEFAULT_TEMPERATURE": "0.7",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "OX_MODEL": "stealth/ox-alpha",
        "OX_TEMPERATURE": "1.0",

        # Embedding Configuration
        "DEFAULT_EMBEDDING_PROVIDER": "huggingface",
        "DEFAULT_RERANKER_MODEL": "AITeamVN/Vietnamese_Reranker",

        # Vector Store Configuration
        "DEFAULT_VECTOR_STORE": "faiss",
        "DEFAULT_COLLECTION_NAME": "default",

        # RAG Configuration
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "50",
        "RETRIEVAL_K": "5",

        # Feature Flags
        "ENABLE_HYBRID_SEARCH": "true",
        "ENABLE_RERANKING": "true",
        "ENABLE_CACHE": "true",
        "ENABLE_API_AUTH": "false",
        "API_KEYS": "",
        "API_RATE_LIMIT": "100",
        "API_RATE_LIMIT_WINDOW": "60",
        "MAX_UPLOAD_SIZE_MB": "25",
        "MAX_MULTIMODAL_FILES": "4",
        "MAX_MULTIMODAL_TOTAL_SIZE_MB": "50",
        "CORS_ALLOW_ORIGINS": "http://localhost:3000,http://localhost:7860,http://localhost:8000",

        # Cache Configuration
        "CACHE_TTL": "3600",

        # Logging
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            env_file: Path to .env file (default: auto-detect)
        """
        # Load .env and .env.local files
        load_environment(env_file)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        # Check environment variables first
        value = os.getenv(key)

        if value is not None:
            return value

        # Check defaults
        if key in self.DEFAULTS:
            return self.DEFAULTS[key]

        return default

    def get_required(self, key: str) -> Any:
        """
        Get required configuration value.

        Args:
            key: Configuration key

        Returns:
            Configuration value

        Raises:
            ValueError: If key is not found
        """
        value = self.get(key)

        if value is None:
            raise ValueError(
                f"Required configuration '{key}' not found. "
                f"Set it in .env file or as environment variable."
            )

        return value

    def get_int(self, key: str, default: int = 0) -> int:
        """
        Get integer configuration value.

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Integer value
        """
        value = self.get(key, default=str(default))
        return int(value)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """
        Get float configuration value.

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Float value
        """
        value = self.get(key, default=str(default))
        return float(value)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get boolean configuration value.

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Boolean value
        """
        value = self.get(key, default=str(default))
        return value.lower() in ("true", "1", "yes", "on")

    def has(self, key: str) -> bool:
        """
        Check if configuration key exists.

        Args:
            key: Configuration key

        Returns:
            True if key exists
        """
        return self.get(key) is not None

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.

        Returns:
            Dict of all configuration
        """
        config = {}

        for key in self.DEFAULTS:
            config[key] = self.get(key)

        return config

    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration.

        Returns:
            Dict with LLM settings
        """
        base_url = self.get("OPENAI_BASE_URL")
        is_openrouter = bool(
            base_url and "openrouter.ai" in str(base_url).casefold()
        )
        api_key = (
            self.get("OPENROUTER_API_KEY") if is_openrouter else None
        ) or self.get("OPENAI_API_KEY") or self.get("ANTHROPIC_API_KEY")

        provider = self.get("DEFAULT_LLM_PROVIDER")
        return {
            "provider": provider,
            "model": self.get("DEFAULT_LLM_MODEL") or default_llm_model(provider),
            "api_key": api_key,
            "base_url": base_url,
            "temperature": self.get_float("DEFAULT_TEMPERATURE"),
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding configuration.

        Returns:
            Dict with embedding settings
        """
        provider = self.get("DEFAULT_EMBEDDING_PROVIDER")
        return {
            "provider": provider,
            "model": self.get("DEFAULT_EMBEDDING_MODEL") or default_embedding_model(provider),
        }

    def get_vector_store_config(self) -> Dict[str, Any]:
        """
        Get vector store configuration.

        Returns:
            Dict with vector store settings
        """
        return {
            "provider": self.get("DEFAULT_VECTOR_STORE"),
            "collection_name": self.get("DEFAULT_COLLECTION_NAME"),
            "persist_directory": self.get("PERSIST_DIRECTORY"),
        }

    def get_rag_config(self) -> Dict[str, Any]:
        """
        Get RAG configuration.

        Returns:
            Dict with RAG settings
        """
        return {
            "chunk_size": self.get_int("CHUNK_SIZE"),
            "chunk_overlap": self.get_int("CHUNK_OVERLAP"),
            "retrieval_k": self.get_int("RETRIEVAL_K"),
            "enable_hybrid_search": self.get_bool("ENABLE_HYBRID_SEARCH"),
            "enable_reranking": self.get_bool("ENABLE_RERANKING"),
        }

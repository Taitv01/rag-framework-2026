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
        "DEFAULT_LLM_PROVIDER": "openai",
        "DEFAULT_LLM_MODEL": "gpt-4o-mini",
        "DEFAULT_TEMPERATURE": "0.7",

        # Embedding Configuration
        "DEFAULT_EMBEDDING_PROVIDER": "huggingface",
        "DEFAULT_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",

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
        # Load .env file
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env file
            env_path = Path.cwd() / ".env"
            if env_path.exists():
                load_dotenv(env_path)

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
        return {
            "provider": self.get("DEFAULT_LLM_PROVIDER"),
            "model": self.get("DEFAULT_LLM_MODEL"),
            "api_key": self.get("OPENAI_API_KEY") or self.get("ANTHROPIC_API_KEY"),
            "temperature": self.get_float("DEFAULT_TEMPERATURE"),
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding configuration.

        Returns:
            Dict with embedding settings
        """
        return {
            "provider": self.get("DEFAULT_EMBEDDING_PROVIDER"),
            "model": self.get("DEFAULT_EMBEDDING_MODEL"),
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

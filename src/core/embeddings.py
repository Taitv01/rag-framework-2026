"""
Embeddings Manager
==================

Abstraction layer for embedding models supporting multiple providers.

Supported providers:
- HuggingFace (local models)
- OpenAI
- Cohere

Usage:
    # Vietnamese embeddings (default, no API key needed)
    embeddings = EmbeddingsManager(provider="huggingface")

    # OpenAI embeddings
    embeddings = EmbeddingsManager(provider="openai", model="text-embedding-3-large")

    # Embed documents
    vectors = embeddings.embed_documents(["text1", "text2"])

    # Embed query
    vector = embeddings.embed_query("search query")
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field

from src.utils.config import load_environment


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    provider: str = "huggingface"
    model_name: str = "BAAI/bge-m3"
    api_key: Optional[str] = None
    batch_size: int = 32
    device: Optional[str] = None
    normalize_embeddings: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class EmbeddingsManager:
    """
    Embedding model manager with multi-provider support.

    Provides a unified interface for different embedding models.

    Example:
        # Vietnamese embeddings (default, free, no API key)
        embeddings = EmbeddingsManager(
            provider="huggingface",
            model_name="keepitreal/vietnamese-sbert"
        )

        # OpenAI embeddings
        embeddings = EmbeddingsManager(
            provider="openai",
            model_name="text-embedding-3-large"
        )

        # Embed documents
        vectors = embeddings.embed_documents(["Hello world", "How are you?"])

        # Embed single query
        vector = embeddings.embed_query("search query")
    """

    # Popular embedding models
    POPULAR_MODELS = {
        "huggingface": {
            # Vietnamese-specific models (recommended for Vietnamese content)
            "keepitreal/vietnamese-sbert": {
                "dimensions": 768,
                "description": "Vietnamese SBERT - best for Vietnamese text (default)",
            },
            "bkai-foundation-models/vietnamese-bi-encoder": {
                "dimensions": 768,
                "description": "PhoBERT-based Vietnamese encoder, high quality",
            },
            "AITeamVN/Vietnamese_Embedding": {
                "dimensions": 1024,
                "description": "bge-m3-based, best Vietnamese retrieval benchmarks",
            },
            # Multilingual models
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
                "dimensions": 384,
                "description": "Multilingual, supports 50+ languages including Vietnamese",
            },
            "BAAI/bge-m3": {
                "dimensions": 1024,
                "description": "[RECOMMENDED] Best multilingual, dense+sparse+ColBERT hybrid, 2026 baseline for Vietnamese RAG",
            },
            "intfloat/multilingual-e5-large": {
                "dimensions": 1024,
                "description": "Strong multilingual, MIT license",
            },
            "intfloat/multilingual-e5-large-instruct": {
                "dimensions": 1024,
                "description": "Top multilingual performer with instruction-tuning, excellent for Vietnamese",
            },
            "Alibaba-NLP/gte-Qwen2-7B-instruct": {
                "dimensions": 3584,
                "description": "Premium 7B parameter model, long context (32k), highest quality multilingual",
            },
            # English models (for English-only content)
            "all-MiniLM-L6-v2": {
                "dimensions": 384,
                "description": "Fast English model, 384 dimensions",
            },
            "all-mpnet-base-v2": {
                "dimensions": 768,
                "description": "Higher quality English, 768 dimensions",
            },
            "BAAI/bge-base-en-v1.5": {
                "dimensions": 768,
                "description": "Best English base model, 768 dimensions",
            },
        },
        "openai": {
            "text-embedding-3-small": {
                "dimensions": 1536,
                "description": "Fast, cost-effective, supports Vietnamese",
            },
            "text-embedding-3-large": {
                "dimensions": 3072,
                "description": "Highest quality, supports Vietnamese",
            },
        },
        "cohere": {
            "embed-multilingual-v3.0": {
                "dimensions": 1024,
                "description": "Multilingual, 100+ languages including Vietnamese",
            },
            "embed-english-v3.0": {
                "dimensions": 1024,
                "description": "English-only",
            },
        },
    }

    def __init__(
        self,
        provider: str = "huggingface",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize embeddings manager.

        Args:
            provider: Embedding provider ('huggingface', 'openai', 'cohere')
            model_name: Model name/identifier
            api_key: API key (for cloud providers)
            batch_size: Batch size for embedding
            device: Device to use ('cpu', 'cuda', 'mps')
            normalize_embeddings: Whether to normalize embeddings
        """
        self.config = EmbeddingConfig(
            provider=provider,
            model_name=model_name or self._get_default_model(provider),
            api_key=api_key,
            batch_size=batch_size,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

        self._embeddings = None

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "huggingface": "BAAI/bge-m3",
            "openai": "text-embedding-3-small",
            "cohere": "embed-multilingual-v3.0",
        }
        return defaults.get(provider, "BAAI/bge-m3")

    @property
    def embeddings(self):
        """Get or create embeddings instance."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def _create_embeddings(self):
        """Create embeddings instance based on provider."""
        if self.config.provider == "huggingface":
            return self._create_huggingface_embeddings()
        elif self.config.provider == "openai":
            return self._create_openai_embeddings()
        elif self.config.provider == "cohere":
            return self._create_cohere_embeddings()
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _create_huggingface_embeddings(self):
        """Create HuggingFace embeddings."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for HuggingFace embeddings. "
                "Install it with: pip install sentence-transformers"
            )

        model_kwargs = {}
        if self.config.device:
            model_kwargs["device"] = self.config.device

        encode_kwargs = {
            "normalize_embeddings": self.config.normalize_embeddings,
            "batch_size": self.config.batch_size,
        }

        return HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def _create_openai_embeddings(self):
        """Create OpenAI embeddings."""
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI embeddings. "
                "Install it with: pip install langchain-openai"
            )

        import os

        load_environment()
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        return OpenAIEmbeddings(
            model=self.config.model_name,
            api_key=api_key,
        )

    def _create_cohere_embeddings(self):
        """Create Cohere embeddings."""
        try:
            from langchain_cohere import CohereEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-cohere is required for Cohere embeddings. "
                "Install it with: pip install langchain-cohere"
            )

        import os

        load_environment()
        api_key = self.config.api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key is required. "
                "Set COHERE_API_KEY environment variable or pass api_key parameter."
            )

        return CohereEmbeddings(
            model=self.config.model_name,
            cohere_api_key=api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)

    @classmethod
    def list_models(cls, provider: Optional[str] = None) -> Dict[str, Dict]:
        """
        List available embedding models.

        Args:
            provider: Filter by provider (None for all)

        Returns:
            Dictionary of available models
        """
        if provider:
            return cls.POPULAR_MODELS.get(provider, {})
        return cls.POPULAR_MODELS

    @classmethod
    def from_provider(cls, provider: str, **kwargs) -> "EmbeddingsManager":
        """
        Create embeddings manager from provider name.

        Args:
            provider: Provider name
            **kwargs: Additional arguments

        Returns:
            EmbeddingsManager instance
        """
        return cls(provider=provider, **kwargs)


# Convenience functions for quick embedding creation
def get_local_embeddings(
    model_name: str = "BAAI/bge-m3",
    device: Optional[str] = None
) -> EmbeddingsManager:
    """
    Get local HuggingFace embeddings (no API key needed).

    Default model is Vietnamese-optimized. For English-only content,
    use model_name="all-MiniLM-L6-v2" for faster performance.

    Args:
        model_name: HuggingFace model name
        device: Device to use ('cpu', 'cuda', 'mps')

    Returns:
        EmbeddingsManager instance
    """
    return EmbeddingsManager(
        provider="huggingface",
        model_name=model_name,
        device=device,
    )


def get_openai_embeddings(
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None
) -> EmbeddingsManager:
    """
    Get OpenAI embeddings.

    Args:
        model_name: OpenAI model name
        api_key: OpenAI API key

    Returns:
        EmbeddingsManager instance
    """
    return EmbeddingsManager(
        provider="openai",
        model_name=model_name,
        api_key=api_key,
    )

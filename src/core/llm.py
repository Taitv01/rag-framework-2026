"""
LLM Manager
===========

Abstraction layer for Large Language Models supporting multiple providers.

Supported providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models (Ollama)

Usage:
    # OpenAI
    llm = LLMManager(provider="openai", model="gpt-4o")

    # Anthropic
    llm = LLMManager(provider="anthropic", model="claude-sonnet-4-20250514")

    # Generate
    response = llm.generate("What is Python?")
"""

import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from src.monitoring.tracing import step, usage_details
from src.utils.config import default_llm_model, load_environment

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMManager:
    """
    LLM manager with multi-provider support.

    Provides a unified interface for different LLM providers.

    Example:
        # OpenAI
        llm = LLMManager(provider="openai", model="gpt-4o")

        # Anthropic
        llm = LLMManager(provider="anthropic", model="claude-sonnet-4-20250514")

        # Generate response
        response = llm.generate("What is Python?")

        # With system prompt
        response = llm.generate(
            "What is Python?",
            system_prompt="You are a helpful Python expert."
        )

        # Streaming
        for chunk in llm.stream("Tell me a story"):
            print(chunk, end="")
    """

    # Popular LLM models
    POPULAR_MODELS = {
        "openai": {
            "gpt-4o": {
                "description": "Most capable GPT-4 model",
                "context_window": 128000,
            },
            "gpt-4o-mini": {
                "description": "Fast, cost-effective GPT-4",
                "context_window": 128000,
            },
            "gpt-4-turbo": {
                "description": "GPT-4 with vision",
                "context_window": 128000,
            },
            "gpt-3.5-turbo": {
                "description": "Fast, affordable",
                "context_window": 16385,
            },
        },
        "anthropic": {
            "claude-sonnet-4-20250514": {
                "description": "Most capable Claude model",
                "context_window": 200000,
            },
            "claude-3-5-sonnet-20241022": {
                "description": "Previous generation Claude",
                "context_window": 200000,
            },
            "claude-3-haiku-20240307": {
                "description": "Fast, affordable Claude",
                "context_window": 200000,
            },
        },
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
    ):
        """
        Initialize LLM manager.

        Args:
            provider: LLM provider ('openai', 'anthropic', 'ollama')
            model: Model name/identifier
            api_key: API key
            base_url: Optional OpenAI-compatible API base URL
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            streaming: Enable streaming
        """
        self.config = LLMConfig(
            provider=provider,
            model=model or self._get_default_model(provider),
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )

        self._llm = None

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        return default_llm_model(provider)

    @property
    def llm(self) -> BaseChatModel:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> BaseChatModel:
        """Create LLM instance based on provider."""
        if self.config.provider == "openai":
            return self._create_openai_llm()
        elif self.config.provider == "anthropic":
            return self._create_anthropic_llm()
        elif self.config.provider == "ollama":
            return self._create_ollama_llm()
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _create_openai_llm(self) -> BaseChatModel:
        """Create OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI. "
                "Install it with: pip install langchain-openai"
            )

        load_environment()
        base_url = self.config.base_url or os.getenv("OPENAI_BASE_URL")
        is_openrouter = bool(base_url and "openrouter.ai" in base_url.casefold())
        api_key = (
            self.config.api_key
            or (os.getenv("OPENROUTER_API_KEY") if is_openrouter else None)
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "An OpenAI-compatible API key is required. Set OPENAI_API_KEY "
                "(or OPENROUTER_API_KEY for OpenRouter) or pass api_key."
            )

        kwargs = {
            "model": self.config.model,
            "api_key": api_key,
            "temperature": self.config.temperature,
            "streaming": self.config.streaming,
        }

        if base_url:
            kwargs["base_url"] = base_url

        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens

        return ChatOpenAI(**kwargs)

    def _create_anthropic_llm(self) -> BaseChatModel:
        """Create Anthropic LLM."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic. "
                "Install it with: pip install langchain-anthropic"
            )

        load_environment()
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        kwargs = {
            "model": self.config.model,
            "api_key": api_key,
            "temperature": self.config.temperature,
            "streaming": self.config.streaming,
        }

        base_url = os.getenv("ANTHROPIC_BASE_URL")
        if base_url:
            kwargs["anthropic_api_url"] = base_url

        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens

        return ChatAnthropic(**kwargs)

    def _create_ollama_llm(self) -> BaseChatModel:
        """Create Ollama LLM (local)."""
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-community is required for Ollama. "
                "Install it with: pip install langchain-community"
            )

        return ChatOllama(
            model=self.config.model,
            temperature=self.config.temperature,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        messages = self._create_messages(prompt, system_prompt)
        with self._generation(prompt) as generation:
            response = self.llm.invoke(messages, **kwargs)
            text = self._content_to_text(response.content)
            generation.update(output=text, usage=usage_details(response))
        return text

    def _generation(self, prompt: Any, current: bool = True):
        """Trace step for one LLM call (see src.monitoring.tracing)."""
        return step(
            "llm",
            as_type="generation",
            input=prompt,
            model=self.config.model,
            metadata={"provider": self.config.provider},
            current=current,
        )

    def generate_multimodal(
        self,
        prompt: str,
        media: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt plus image/video URL or data-URL inputs.

        Media items use the provider-neutral shape ``{"type": "image" | "video",
        "url": "..."}``. They are translated to OpenAI-compatible
        ``image_url`` and ``video_url`` content blocks used by OpenRouter.
        """
        if self.config.provider != "openai":
            raise ValueError(
                "Image/video input currently requires an OpenAI-compatible provider"
            )
        if not media:
            raise ValueError("At least one image or video is required")

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        block_types = {
            "image": "image_url",
            "video": "video_url",
        }

        for item in media:
            media_type = str(item.get("type", "")).casefold()
            url = str(item.get("url", "")).strip()
            block_type = block_types.get(media_type)
            if not block_type:
                raise ValueError(f"Unsupported media type: {media_type or '<empty>'}")
            if not url:
                raise ValueError("Media URL cannot be empty")
            content.append({
                "type": block_type,
                block_type: {"url": url},
            })

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=content))

        # Media can be large data URLs: trace the prompt and a count only.
        with self._generation({"prompt": prompt, "media": len(media)}) as generation:
            response = self.llm.invoke(messages, **kwargs)
            text = self._content_to_text(response.content)
            generation.update(output=text, usage=usage_details(response))
        return text

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Normalize text returned as a string or provider content blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        return str(content)

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate response from message list.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        with self._generation(messages) as generation:
            response = self.llm.invoke(lc_messages, **kwargs)
            generation.update(output=response.content, usage=usage_details(response))
        return response.content

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Stream response from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Response chunks
        """
        messages = self._create_messages(prompt, system_prompt)

        # Consumers may resume this generator in other threads: not a parent step.
        with self._generation(prompt, current=False) as generation:
            parts, usage = [], None
            for chunk in self.llm.stream(messages, **kwargs):
                usage = usage_details(chunk) or usage
                parts.append(self._content_to_text(chunk.content))
                yield chunk.content
            generation.update(output="".join(parts), usage=usage)

    def _create_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List:
        """Create message list from prompt."""
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        return messages

    def bind_tools(self, tools: List[Any]) -> BaseChatModel:
        """
        Bind tools to LLM for function calling.

        Args:
            tools: List of tools to bind

        Returns:
            LLM with bound tools
        """
        return self.llm.bind_tools(tools)

    def with_structured_output(self, schema: Any) -> BaseChatModel:
        """
        Get LLM with structured output.

        Args:
            schema: Pydantic model or JSON schema

        Returns:
            LLM with structured output
        """
        return self.llm.with_structured_output(schema)

    @classmethod
    def list_models(cls, provider: Optional[str] = None) -> Dict[str, Dict]:
        """
        List available models.

        Args:
            provider: Filter by provider (None for all)

        Returns:
            Dictionary of available models
        """
        if provider:
            return cls.POPULAR_MODELS.get(provider, {})
        return cls.POPULAR_MODELS

    def get_context_window(self) -> int:
        """
        Get the context window size for the current model.

        Returns:
            Context window size in tokens. Returns 128000 as default
            if model is not in the known models catalog.
        """
        provider_models = self.POPULAR_MODELS.get(self.config.provider, {})
        model_info = provider_models.get(self.config.model, {})

        if "context_window" in model_info:
            return model_info["context_window"]

        # Fallback: try to infer from model name
        model_lower = self.config.model.lower()
        if "gpt-4" in model_lower:
            return 128000
        elif "gpt-3.5" in model_lower:
            return 16385
        elif "claude" in model_lower:
            return 200000
        elif "llama" in model_lower or "mistral" in model_lower:
            return 8192  # Common local model default

        # Default: assume a generous context window
        logger.warning(
            f"Unknown model '{self.config.model}'. "
            f"Assuming 128000 token context window."
        )
        return 128000

    def get_max_output_tokens(self) -> int:
        """
        Get the maximum output tokens for the current model.

        Returns:
            Max output tokens. Uses configured max_tokens or defaults to 4096.
        """
        if self.config.max_tokens:
            return self.config.max_tokens

        # Default output token limits by provider
        provider_defaults = {
            "openai": 4096,
            "anthropic": 4096,
            "ollama": 2048,
        }
        return provider_defaults.get(self.config.provider, 4096)

    @classmethod
    def from_provider(cls, provider: str, **kwargs) -> "LLMManager":
        """
        Create LLM manager from provider name.

        Args:
            provider: Provider name
            **kwargs: Additional arguments

        Returns:
            LLMManager instance
        """
        return cls(provider=provider, **kwargs)


# Convenience functions
def get_openai_llm(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.7
) -> LLMManager:
    """
    Get OpenAI LLM.

    Args:
        model: Model name
        api_key: API key
        temperature: Temperature

    Returns:
        LLMManager instance
    """
    return LLMManager(
        provider="openai",
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def get_anthropic_llm(
    model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None,
    temperature: float = 0.7
) -> LLMManager:
    """
    Get Anthropic LLM.

    Args:
        model: Model name
        api_key: API key
        temperature: Temperature

    Returns:
        LLMManager instance
    """
    return LLMManager(
        provider="anthropic",
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def get_local_llm(
    model: str = "llama3",
    temperature: float = 0.7
) -> LLMManager:
    """
    Get local Ollama LLM.

    Args:
        model: Model name
        temperature: Temperature

    Returns:
        LLMManager instance
    """
    return LLMManager(
        provider="ollama",
        model=model,
        temperature=temperature,
    )

"""
Context Window Validator
=======================

Validates that assembled prompts fit within LLM context windows.

Features:
- Token counting (tiktoken for OpenAI, estimation for others)
- Automatic truncation with configurable strategy
- Warnings when approaching context limits
- Reserved output token management

Usage:
    from src.utils.context_validator import ContextValidator

    validator = ContextValidator(context_window=128000, max_output_tokens=4096)

    # Check if prompt fits
    result = validator.validate(prompt="..." * 10000, system_prompt="You are helpful.")
    if result.is_too_large:
        print(f"Prompt exceeds limit by {result.overflow_tokens} tokens")
        prompt = result.truncated_prompt

    # Or use the convenience method
    safe_prompt = validator.fit_to_window(prompt, system_prompt="...")
"""

import logging
import math
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Approximate chars per token for different languages
CHARS_PER_TOKEN = {
    "en": 4.0,      # English: ~4 chars per token
    "vi": 3.0,      # Vietnamese: ~3 chars per token (diacritics, compound words)
    "zh": 1.5,      # Chinese: ~1.5 chars per token
    "default": 3.5, # Mixed/default: ~3.5 chars per token
}


@dataclass
class ValidationResult:
    """Result of context window validation."""

    is_valid: bool
    """Whether the prompt fits within the context window."""

    estimated_tokens: int
    """Estimated total tokens (prompt + system + output reservation)."""

    context_window: int
    """The model's context window size in tokens."""

    available_tokens: int
    """Tokens available for the prompt (context_window - output_reservation)."""

    overflow_tokens: int
    """Number of tokens over the limit (0 if valid)."""

    prompt_tokens: int
    """Estimated tokens for the prompt content."""

    system_tokens: int
    """Estimated tokens for the system prompt."""

    truncated_prompt: Optional[str] = None
    """The truncated prompt if overflow occurred, None otherwise."""

    warning: Optional[str] = None
    """Warning message if approaching limits."""

    @property
    def is_too_large(self) -> bool:
        """Alias for not is_valid."""
        return not self.is_valid

    @property
    def usage_ratio(self) -> float:
        """Ratio of used tokens to available tokens (0.0 to 1.0+)."""
        if self.available_tokens == 0:
            return 1.0
        return self.prompt_tokens / self.available_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "estimated_tokens": self.estimated_tokens,
            "context_window": self.context_window,
            "available_tokens": self.available_tokens,
            "overflow_tokens": self.overflow_tokens,
            "prompt_tokens": self.prompt_tokens,
            "system_tokens": self.system_tokens,
            "usage_ratio": round(self.usage_ratio, 3),
            "warning": self.warning,
        }


class ContextValidator:
    """
    Validates and manages context window limits.

    Ensures prompts fit within the LLM's context window, with
    automatic truncation and warning thresholds.

    Example:
        validator = ContextValidator(context_window=128000)

        # Validate a prompt
        result = validator.validate(long_prompt, system_prompt="You are helpful.")
        if result.is_too_large:
            safe_prompt = result.truncated_prompt

        # Or use fit_to_window directly
        safe_prompt = validator.fit_to_window(long_prompt)

        # Get token estimate
        tokens = validator.estimate_tokens("Hello world")
    """

    def __init__(
        self,
        context_window: int = 128000,
        max_output_tokens: int = 4096,
        warning_threshold: float = 0.8,
        language: str = "default",
        reserve_tokens: Optional[int] = None,
    ):
        """
        Initialize context validator.

        Args:
            context_window: Model's context window size in tokens
            max_output_tokens: Maximum tokens reserved for output
            warning_threshold: Threshold (0-1) to trigger warning (e.g., 0.8 = 80%)
            language: Language for token estimation ("en", "vi", "zh", "default")
            reserve_tokens: Custom token reservation (overrides max_output_tokens)
        """
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.warning_threshold = warning_threshold
        self.language = language
        self.reserve_tokens = reserve_tokens or max_output_tokens

        # Available tokens for input (context minus output reservation)
        self.available_tokens = max(0, context_window - self.reserve_tokens)

        # Try to load tiktoken for accurate counting
        self._tiktoken_available = False
        self._encoding = None
        try:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")
            self._tiktoken_available = True
        except ImportError:
            pass

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Uses tiktoken if available (accurate for OpenAI models),
        falls back to character-based estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0

        if self._tiktoken_available and self._encoding:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass

        # Fallback: character-based estimation
        return self.estimate_tokens(text)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens using character-based heuristic.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        chars_per_token = CHARS_PER_TOKEN.get(self.language, CHARS_PER_TOKEN["default"])
        return math.ceil(len(text) / chars_per_token)

    def validate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate that a prompt fits within the context window.

        Args:
            prompt: The main prompt text
            system_prompt: Optional system prompt
            additional_context: Optional additional context to include

        Returns:
            ValidationResult with details about fit and overflow
        """
        # Count tokens
        prompt_tokens = self.count_tokens(prompt)
        system_tokens = self.count_tokens(system_prompt or "")
        additional_tokens = self.count_tokens(additional_context or "")

        total_input_tokens = prompt_tokens + system_tokens + additional_tokens

        # Check if it fits
        overflow = max(0, total_input_tokens - self.available_tokens)
        is_valid = overflow == 0

        # Generate warning if approaching limit
        warning = None
        usage_ratio = total_input_tokens / self.available_tokens if self.available_tokens > 0 else 1.0

        if usage_ratio >= 1.0:
            warning = (
                f"Context EXCEEDS limit: {total_input_tokens:,} tokens "
                f"> {self.available_tokens:,} available "
                f"(context_window={self.context_window:,}, "
                f"output_reserved={self.reserve_tokens:,})"
            )
        elif usage_ratio >= self.warning_threshold:
            warning = (
                f"Context approaching limit: {total_input_tokens:,} / "
                f"{self.available_tokens:,} tokens "
                f"({usage_ratio:.0%} used)"
            )

        # Truncate if needed
        truncated_prompt = None
        if not is_valid:
            truncated_prompt = self._truncate_text(prompt, self.available_tokens - system_tokens - additional_tokens)

        return ValidationResult(
            is_valid=is_valid,
            estimated_tokens=total_input_tokens + self.reserve_tokens,
            context_window=self.context_window,
            available_tokens=self.available_tokens,
            overflow_tokens=overflow,
            prompt_tokens=prompt_tokens,
            system_tokens=system_tokens,
            truncated_prompt=truncated_prompt,
            warning=warning,
        )

    def fit_to_window(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        additional_context: Optional[str] = None,
        truncation_strategy: str = "tail",
    ) -> str:
        """
        Fit a prompt to the context window, truncating if necessary.

        Args:
            prompt: The main prompt text
            system_prompt: Optional system prompt
            additional_context: Optional additional context
            truncation_strategy: "tail" (remove end), "head" (remove start),
                                or "middle" (keep start and end)

        Returns:
            Prompt that fits within the context window
        """
        result = self.validate(prompt, system_prompt, additional_context)

        if result.warning:
            logger.warning(result.warning)

        if result.is_valid:
            return prompt

        # Truncate
        available_for_prompt = self.available_tokens - result.system_tokens
        if additional_context:
            available_for_prompt -= self.count_tokens(additional_context)

        available_for_prompt = max(100, available_for_prompt)  # Minimum 100 tokens

        return self._truncate_text(prompt, available_for_prompt, strategy=truncation_strategy)

    def _truncate_text(
        self,
        text: str,
        max_tokens: int,
        strategy: str = "tail",
    ) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            strategy: Truncation strategy

        Returns:
            Truncated text
        """
        if max_tokens <= 0:
            return ""

        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # Calculate target character count
        chars_per_token = CHARS_PER_TOKEN.get(self.language, CHARS_PER_TOKEN["default"])
        target_chars = int(max_tokens * chars_per_token * 0.95)  # 95% to be safe

        if strategy == "tail":
            truncated = text[:target_chars]
            # Try to break at sentence boundary
            last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
            if last_period > target_chars * 0.8:
                truncated = truncated[:last_period + 1]
            return truncated + "\n[... truncated due to context limit ...]"

        elif strategy == "head":
            truncated = text[-target_chars:]
            # Try to break at sentence boundary
            first_period = truncated.find(". ")
            if first_period < target_chars * 0.2 and first_period > 0:
                truncated = truncated[first_period + 2:]
            return "[... truncated due to context limit ...]\n" + truncated

        elif strategy == "middle":
            half = target_chars // 2
            head = text[:half]
            tail = text[-half:]
            # Adjust boundaries
            last_period = head.rfind(". ")
            if last_period > half * 0.8:
                head = head[:last_period + 1]
            first_period = tail.find(". ")
            if first_period < half * 0.2 and first_period > 0:
                tail = tail[first_period + 2:]
            return head + "\n[... truncated ...]\n" + tail

        return text[:target_chars]

    @classmethod
    def from_llm_manager(cls, llm_manager, **kwargs) -> "ContextValidator":
        """
        Create a ContextValidator from an LLMManager instance.

        Args:
            llm_manager: LLMManager instance
            **kwargs: Additional arguments for ContextValidator

        Returns:
            ContextValidator configured for the model
        """
        context_window = llm_manager.get_context_window()
        max_output = llm_manager.get_max_output_tokens()

        # Detect language from model name
        language = "default"
        model_lower = llm_manager.config.model.lower()
        if any(kw in model_lower for kw in ["vietnamese", "vi-", "pho"]):
            language = "vi"

        return cls(
            context_window=context_window,
            max_output_tokens=max_output,
            language=language,
            **kwargs,
        )

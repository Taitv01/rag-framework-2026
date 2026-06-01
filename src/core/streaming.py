"""
Streaming Response
==================

Streaming support for real-time token output.

Features:
- Token-by-token streaming
- Callback support
- Async streaming

Usage:
    from src.core.streaming import StreamingManager

    streaming = StreamingManager(llm)
    for token in streaming.stream("What is Python?"):
        print(token, end="")
"""

from typing import Iterator, Callable, Optional, AsyncIterator
from dataclasses import dataclass
import asyncio


@dataclass
class StreamToken:
    """Single token from stream."""
    content: str
    index: int
    finish_reason: Optional[str] = None


class StreamingManager:
    """
    Manager for streaming responses.

    Provides real-time token streaming for LLM responses.

    Example:
        streaming = StreamingManager(llm)

        # Synchronous streaming
        for token in streaming.stream("What is Python?"):
            print(token, end="")

        # With callback
        def on_token(token):
            print(token, end="")

        streaming.stream_with_callback("What is Python?", on_token)
    """

    def __init__(self, llm, on_token: Optional[Callable] = None):
        """
        Initialize streaming manager.

        Args:
            llm: LLM instance
            on_token: Optional callback for each token
        """
        self.llm = llm
        self.on_token = on_token

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream response token by token.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Token strings
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        for chunk in self.llm.llm.stream(messages, **kwargs):
            token = chunk.content
            if token:
                if self.on_token:
                    self.on_token(token)
                yield token

    def stream_with_callback(
        self,
        prompt: str,
        callback: Callable[[str], None],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Stream with callback and return full response.

        Args:
            prompt: User prompt
            callback: Callback function for each token
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Full response text
        """
        full_response = []

        for token in self.stream(prompt, system_prompt, **kwargs):
            callback(token)
            full_response.append(token)

        return "".join(full_response)

    async def astream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Async stream response token by token.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Token strings
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        async for chunk in self.llm.llm.astream(messages, **kwargs):
            token = chunk.content
            if token:
                yield token


class StreamingBuffer:
    """
    Buffer for collecting streamed tokens.

    Useful for collecting tokens and processing them in batches.

    Example:
        buffer = StreamingBuffer()

        for token in stream:
            buffer.add(token)

            # Process when buffer is full
            if buffer.is_full():
                process(buffer.flush())
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize streaming buffer.

        Args:
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.buffer = []
        self.full_text = []

    def add(self, token: str) -> None:
        """Add token to buffer."""
        self.buffer.append(token)
        self.full_text.append(token)

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) >= self.max_size

    def flush(self) -> str:
        """Flush buffer and return content."""
        content = "".join(self.buffer)
        self.buffer = []
        return content

    def get_full_text(self) -> str:
        """Get full collected text."""
        return "".join(self.full_text)

    def clear(self) -> None:
        """Clear buffer and full text."""
        self.buffer = []
        self.full_text = []

    def __len__(self) -> int:
        return len(self.buffer)

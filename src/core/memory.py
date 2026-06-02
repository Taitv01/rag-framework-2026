"""
Conversation Memory
===================

Memory management for multi-turn conversations.

Features:
- Buffer memory (full history)
- Summary memory (compressed history)
- Window memory (last N turns)
- Vector memory (semantic search over history)

Usage:
    from src.core.memory import ConversationMemory

    memory = ConversationMemory(type="buffer", max_history=10)
    memory.add_user_message("Hello")
    memory.add_ai_message("Hi there!")
    history = memory.get_history()
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Message:
    """Single message in conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """
    Conversation memory for multi-turn chat.

    Supports multiple memory strategies:
    - buffer: Store full conversation history
    - window: Store last N messages
    - summary: Store summarized history

    Example:
        memory = ConversationMemory(type="buffer", max_history=10)

        memory.add_user_message("What is Python?")
        memory.add_ai_message("Python is a programming language...")

        memory.add_user_message("Tell me more")
        memory.add_ai_message("Python was created by...")

        history = memory.get_history()
        context = memory.get_context_string()
    """

    def __init__(
        self,
        type: str = "buffer",
        max_history: int = 20,
        max_tokens: Optional[int] = None,
        llm=None
    ):
        """
        Initialize conversation memory.

        Args:
            type: Memory type ('buffer', 'window', 'summary')
            max_history: Maximum messages to store
            max_tokens: Maximum tokens for context
            llm: LLM instance (required for 'summary' type)
        """
        self.type = type
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.llm = llm

        self.messages: List[Message] = []
        self.summary: Optional[str] = None

    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add user message to memory."""
        self.messages.append(Message(
            role="user",
            content=content,
            metadata=metadata or {}
        ))
        self._trim_messages()

    def add_ai_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add AI response to memory."""
        self.messages.append(Message(
            role="assistant",
            content=content,
            metadata=metadata or {}
        ))
        self._trim_messages()

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add message to memory."""
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        self._trim_messages()

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history.

        Returns:
            List of message dicts with 'role' and 'content'
        """
        if self.type == "window":
            messages = self.messages[-self.max_history:]
        else:
            messages = self.messages

        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def get_context_string(self) -> str:
        """
        Get conversation history as formatted string.

        Returns:
            Formatted conversation history
        """
        history = self.get_history()

        if not history:
            return ""

        parts = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")

        return "\n".join(parts)

    def get_context_for_prompt(self) -> str:
        """
        Get context formatted for prompt injection.

        Returns:
            Context string for prompt
        """
        if self.type == "summary" and self.summary:
            return f"Previous conversation summary:\n{self.summary}\n\nRecent messages:\n{self.get_context_string()}"

        return self.get_context_string()

    def summarize(self) -> str:
        """
        Summarize conversation history.

        Returns:
            Conversation summary
        """
        if not self.llm:
            raise ValueError("LLM is required for summarization")

        history = self.get_context_string()

        prompt = f"""Summarize the following conversation in a concise manner.
Preserve key information, decisions, and context.
Tóm tắt cuộc trò chuyện sau một cách ngắn gọn.
Giữ lại thông tin quan trọng, quyết định và ngữ cảnh.

Conversation / Cuộc trò chuyện:
{history}

Summary / Tóm tắt:"""

        self.summary = self.llm.generate(prompt)
        return self.summary

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.summary = None

    def _trim_messages(self) -> None:
        """Trim messages based on memory type."""
        if self.type == "window":
            # Keep only last N messages
            if len(self.messages) > self.max_history * 2:
                self.messages = self.messages[-(self.max_history * 2):]
        elif self.type == "buffer":
            # Trim if too many messages
            if len(self.messages) > self.max_history * 2:
                # Summarize old messages and keep recent
                if self.llm:
                    old_messages = self.messages[:-self.max_history]
                    self._summarize_old(old_messages)
                    self.messages = self.messages[-self.max_history:]

    def _summarize_old(self, old_messages: List[Message]) -> None:
        """Summarize old messages."""
        history = "\n".join([f"{m.role}: {m.content}" for m in old_messages])

        prompt = f"""Summarize this conversation history briefly.
Tóm tắt lịch sử cuộc trò chuyện này.

{history}

Summary / Tóm tắt:"""

        new_summary = self.llm.generate(prompt)

        if self.summary:
            self.summary = f"{self.summary}\n{new_summary}"
        else:
            self.summary = new_summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            "type": self.type,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.messages
            ],
            "summary": self.summary,
        }

    def save(self, path: str) -> None:
        """Save memory to file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load memory from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.type = data.get("type", "buffer")
        self.summary = data.get("summary")

        self.messages = []
        for m in data.get("messages", []):
            self.messages.append(Message(
                role=m["role"],
                content=m["content"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
            ))

    @property
    def message_count(self) -> int:
        """Number of messages in memory."""
        return len(self.messages)

    def __len__(self) -> int:
        return len(self.messages)


class ConversationalRAG:
    """
    RAG with conversation memory.

    Extends any RAG pattern with multi-turn conversation support.

    Example:
        from src.core.memory import ConversationalRAG
        from src.rag import AdvancedRAG

        rag = AdvancedRAG()
        conv_rag = ConversationalRAG(rag, memory_type="buffer")

        # Multi-turn conversation
        answer1 = conv_rag.query("What is Python?")
        answer2 = conv_rag.query("Tell me more about it")
        answer3 = conv_rag.query("So sánh với Java")
    """

    def __init__(
        self,
        rag,
        memory_type: str = "buffer",
        max_history: int = 20,
        llm=None
    ):
        """
        Initialize conversational RAG.

        Args:
            rag: RAG instance
            memory_type: Memory type
            max_history: Maximum history messages
            llm: LLM instance for summarization
        """
        self.rag = rag
        self.memory = ConversationMemory(
            type=memory_type,
            max_history=max_history,
            llm=llm or rag.llm
        )

    def query(self, question: str, **kwargs) -> str:
        """
        Query with conversation context.

        Args:
            question: Question to ask
            **kwargs: Additional parameters

        Returns:
            Answer string
        """
        # Get conversation context
        context = self.memory.get_context_for_prompt()

        # Build enhanced question with context
        if context:
            enhanced_question = f"""Previous conversation:
{context}

Current question: {question}"""
        else:
            enhanced_question = question

        # Query RAG
        answer = self.rag.query(enhanced_question, **kwargs)

        # Update memory
        self.memory.add_user_message(question)
        self.memory.add_ai_message(answer)

        return answer

    def stream(self, question: str, **kwargs):
        """
        Stream response with conversation context.

        Args:
            question: Question to ask
            **kwargs: Additional parameters

        Yields:
            Response tokens
        """
        # Get conversation context
        context = self.memory.get_context_for_prompt()

        # Build enhanced question
        if context:
            enhanced_question = f"""Previous conversation:
{context}

Current question: {question}"""
        else:
            enhanced_question = question

        # Stream response
        full_response = []
        for token in self.rag.stream(enhanced_question, **kwargs):
            full_response.append(token)
            yield token

        # Update memory
        self.memory.add_user_message(question)
        self.memory.add_ai_message("".join(full_response))

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.memory.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.memory.get_history()

    def save_history(self, path: str) -> None:
        """Save conversation history."""
        self.memory.save(path)

    def load_history(self, path: str) -> None:
        """Load conversation history."""
        self.memory.load(path)

"""
Agent LLM
=========

An ``llm`` stand-in answered by the AI agent that runs RAG — "the calling
model" (Claude, Codex, Gemini, …) — so RAG needs no API key and no local
model server.

It works in two passes because the agent cannot be called back mid-run:

1. First run: every prompt without an answer is recorded and ``generate``
   returns a neutral placeholder. ``write_requests()`` saves the prompts.
2. The agent reads the requests file and writes one answer per prompt id into
   the answers file, together with its own model name.
3. Second run: ``generate`` replays the answers; ``complete`` is True and the
   caller can publish its report, crediting ``answered_by`` as the model.

Prompts are keyed by the SHA-256 of their text, so a changed prompt is asked
again and answers to prompts that no longer occur are ignored. Only the
standard library is used: callers such as story pipelines can load this file
by path without importing the ``src`` package and its embedding stack.

Pipelines whose prompts depend on earlier answers (rewrite the query, then
grade what it retrieved, then answer) run each unit of work as a chain:
``start_chain()`` before it, and after the chain's first unanswered prompt
the later ones are not requested, because they would be built on a
placeholder. Each run then asks one more layer, until ``complete``. Prompts
that do not depend on each other (grading several documents) can be asked in
the same run inside ``with llm.independent():``.

``AgentChatModel`` plugs an AgentLLM into ``LLMManager`` in place of a
LangChain chat model.

Usage:
    llm = AgentLLM("rag_answers.json", "rag_requests.json")
    issues = ConsistencyChecker(llm).check_chapter(text, 1)
    if not llm.complete:
        llm.write_requests()   # the agent answers, then run again
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ANSWERS_FORMAT = 'JSON {"model": "<your model id>", "answers": {"<id>": "<answer text>"}}'


def prompt_id(prompt: str, system_prompt: str | None = None) -> str:
    """Stable id of one prompt (and the system prompt it is asked under)."""
    text = prompt if system_prompt is None else f"{system_prompt}\n\n{prompt}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AgentLLM:
    """``generate(prompt) -> str`` backed by answers the calling agent writes."""

    def __init__(
        self,
        answers_path: str | Path,
        requests_path: str | Path,
        system_prompt: str | None = None,
        placeholder: str = "OK",
    ):
        """
        Args:
            answers_path: JSON the agent writes (see ``ANSWERS_FORMAT``).
            requests_path: JSON this class writes with the unanswered prompts.
            system_prompt: Instructions the agent must follow for every answer.
            placeholder: Returned for unanswered prompts during the first pass;
                the caller must not publish results while ``complete`` is False.
        """
        self.answers_path = Path(answers_path)
        self.requests_path = Path(requests_path)
        self.system_prompt = system_prompt
        self.placeholder = placeholder
        self.answered_by: str | None = None
        self._answers: dict[str, str] = {}
        self._pending: dict[str, str] = {}
        self._used = 0
        self._deferred = 0
        self._in_chain = False  # gating applies only after start_chain()
        self._chain_blocked = False
        self._in_group = False
        self._group_missing = False
        self._load_answers()

    def _load_answers(self) -> None:
        if not self.answers_path.is_file():
            return
        data = json.loads(self.answers_path.read_text(encoding="utf-8"))
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            # ValueError like json.JSONDecodeError: callers handle one "bad file" error type
            raise ValueError(f"{self.answers_path}: expected {ANSWERS_FORMAT}")  # noqa: TRY004
        self._answers = {str(key): str(value) for key, value in answers.items()}
        model = data.get("model")
        self.answered_by = model.strip() if isinstance(model, str) and model.strip() else None

    def start_chain(self) -> None:
        """Begin a unit of work whose later prompts may depend on earlier answers."""
        self._in_chain = True
        self._chain_blocked = False

    @contextmanager
    def independent(self):
        """Prompts asked inside do not depend on each other's answers.

        All of them are requested in the same run; the chain blocks only after
        the group, if any of them had no answer.
        """
        if not self._in_chain or self._chain_blocked or self._in_group:
            yield
            return
        self._in_group, self._group_missing = True, False
        try:
            yield
        finally:
            self._in_group = False
            if self._group_missing:
                self._chain_blocked = True

    def generate(self, prompt: str) -> str:
        key = prompt_id(prompt, self.system_prompt)
        if key in self._answers:
            self._used += 1
            return self._answers[key]
        if self._chain_blocked:
            # Built after a placeholder in this chain: ask it once the earlier answer exists.
            self._deferred += 1
        else:
            self._pending.setdefault(key, prompt)
            if self._in_group:
                self._group_missing = True
            else:
                self._chain_blocked = self._in_chain
        return self.placeholder

    @property
    def pending(self) -> dict[str, str]:
        """Unanswered prompts seen so far, by id."""
        return dict(self._pending)

    @property
    def complete(self) -> bool:
        """Every prompt of this run had an answer, and the answers name their model."""
        return not self._pending and not self._deferred and self.answered_by is not None

    @property
    def deferred(self) -> int:
        """Prompts skipped because an earlier prompt of their chain had no answer yet."""
        return self._deferred

    @property
    def answers_used(self) -> int:
        return self._used

    def write_requests(self) -> Path:
        """Save the unanswered prompts for the agent; returns the requests path."""
        missing_model = not self._pending and self.answered_by is None
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "answers_file": str(self.answers_path),
            "answers_format": ANSWERS_FORMAT,
            "instructions": (
                "Answer every request below yourself, as the model running this task. "
                "Keep answers already in the answers file, add the new ids, and set "
                '"model" to your own model id. Then run the same command again.'
                + (' The answers file has no "model" value yet.' if missing_model else "")
            ),
            "system_prompt": self.system_prompt,
            "requests": [{"id": key, "prompt": prompt} for key, prompt in self._pending.items()],
        }
        self.requests_path.parent.mkdir(parents=True, exist_ok=True)
        self.requests_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return self.requests_path


class AgentReply:
    """Minimal chat-model response: ``LLMManager`` only reads ``content``."""

    def __init__(self, content: str):
        self.content = content


class AgentChatModel:
    """
    Chat-model stand-in backed by an AgentLLM, for ``LLMManager``:

        manager = LLMManager()
        manager._llm = AgentChatModel(AgentLLM("answers.json", "requests.json"))
        manager.config.provider, manager.config.model = "agent", "agent"
    """

    def __init__(self, agent_llm: AgentLLM):
        self.agent_llm = agent_llm

    def invoke(self, messages, **_kwargs) -> AgentReply:
        return AgentReply(self.agent_llm.generate(messages_to_prompt(messages)))

    def stream(self, messages, **kwargs):
        yield self.invoke(messages, **kwargs)


def messages_to_prompt(messages) -> str:
    """Flatten chat messages into one prompt; a lone user message stays as is."""
    if isinstance(messages, str):
        return messages
    items = list(messages)
    if len(items) == 1:
        return str(getattr(items[0], "content", items[0]))
    parts = []
    for message in items:
        role = getattr(message, "type", None) or type(message).__name__
        parts.append(f"[{role}]\n{getattr(message, 'content', message)}")
    return "\n\n".join(parts)

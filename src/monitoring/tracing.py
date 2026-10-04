"""
Pipeline tracing
================

``step()`` marks one stage of answering a question: rewrite, retrieve, rerank,
grade, generate, and every LLM call. Two optional consumers:

- ``record_steps()`` collects the name and duration of each step run inside
  it (``AdvancedRAG.query_detailed`` reports them as ``steps``).
- Langfuse, once ``LangfuseTracer`` is enabled (keys set): each step becomes an
  observation nested under the request's trace, and LLM calls become
  generations with their model and token usage, from which Langfuse computes
  the cost.

With neither, a step costs a context-variable lookup. Only the standard
library is imported; the Langfuse client is handed in by ``LangfuseTracer``.

Usage:
    with step("retrieve", input={"query": q}) as s:
        docs = search(q)
        s.update(output={"documents": len(docs)})

    with record_steps() as steps:
        rag.query("...")
    # steps == [{"name": "retrieve", "type": "span", "ms": 12.3}, ...]
"""

import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_steps: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
    "rag_trace_steps", default=None
)
_client = None  # Langfuse client while tracing is enabled


def set_langfuse_client(client) -> None:
    """Send steps to this Langfuse client (None stops sending)."""
    global _client
    _client = client


def langfuse_client():
    return _client


@contextmanager
def record_steps() -> Iterator[List[Dict[str, Any]]]:
    """Collect the steps run inside (worker threads started with asyncio.to_thread included)."""
    steps: List[Dict[str, Any]] = []
    token = _steps.set(steps)
    try:
        yield steps
    finally:
        _steps.reset(token)


def usage_details(message: Any) -> Optional[Dict[str, int]]:
    """Token counts of a LangChain message in Langfuse's names, or None if not reported."""
    usage = getattr(message, "usage_metadata", None) or {}
    names = {"input_tokens": "input", "output_tokens": "output", "total_tokens": "total"}
    details = {names[key]: int(value) for key, value in usage.items() if key in names and value is not None}
    return details or None


class Step:
    """Handle on a running step."""

    def __init__(self, record: Dict[str, Any], observation: Any = None):
        self.record = record
        self._observation = observation

    def update(
        self,
        output: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
        model: Optional[str] = None,
    ) -> None:
        if usage:
            self.record["usage"] = usage
        if model:
            self.record["model"] = model
        if self._observation is None:
            return
        fields: Dict[str, Any] = {}
        if output is not None:
            fields["output"] = output
        if metadata is not None:
            fields["metadata"] = metadata
        if usage:
            fields["usage_details"] = usage
        if model:
            fields["model"] = model
        if fields:
            try:
                self._observation.update(**fields)
            except Exception as e:  # tracing must never break a query
                logger.debug(f"Trace update failed: {e}")


@contextmanager
def step(
    name: str,
    as_type: str = "span",
    input: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    current: bool = True,
) -> Iterator[Step]:
    """
    One stage of the pipeline.

    Args:
        name: Step name ("retrieve", "rerank", "generate", ...)
        as_type: Langfuse observation type: "span", "generation", "retriever"...
        input: Input shown in Langfuse
        metadata: Extra fields shown in Langfuse
        model: Model name, for generations
        current: Make the step the parent of steps started inside it. Use
            False inside generators that may resume in other threads (token
            streams): the observation is then ended explicitly.
    """
    record: Dict[str, Any] = {"name": name, "type": as_type}
    if model:
        record["model"] = model
    steps = _steps.get()
    if steps is not None:
        steps.append(record)

    started = time.perf_counter()
    client = _client
    observation, scope = None, None
    if client is not None:
        options = {"name": name, "as_type": as_type, "input": input, "metadata": metadata}
        if model:
            options["model"] = model
        try:
            if current:
                scope = client.start_as_current_observation(**options)
                observation = scope.__enter__()
            else:
                observation = client.start_observation(**options)
        except Exception as e:
            logger.debug(f"Trace step {name!r} not recorded: {e}")
            observation, scope = None, None

    error = None
    try:
        yield Step(record, observation)
    except BaseException as e:
        error = e
        raise
    finally:
        record["ms"] = round((time.perf_counter() - started) * 1000, 1)
        if observation is not None:
            try:
                if error is not None and not isinstance(error, GeneratorExit):
                    observation.update(level="ERROR", status_message=f"{type(error).__name__}: {error}")
                if scope is not None:
                    scope.__exit__(None, None, None)
                else:
                    observation.end()
            except Exception as e:
                logger.debug(f"Trace step {name!r} not closed: {e}")

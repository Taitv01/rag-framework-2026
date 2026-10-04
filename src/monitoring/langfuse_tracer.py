"""
Langfuse Tracer Module
======================

Production tracing for RAG requests with Langfuse (Python SDK 3 or later,
built on OpenTelemetry). Enabled when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are set; otherwise every call is a cheap no-op.

A request is one trace. Inside ``trace()``, the pipeline's steps (rewrite,
retrieve, rerank, grade, generate, see ``src.monitoring.tracing``) nest under
it, including steps run in worker threads, and each LLM call is a generation
with its model and token usage: Langfuse computes the cost from them.

Usage:
    from src.monitoring.langfuse_tracer import LangfuseTracer

    tracer = LangfuseTracer()
    with tracer.trace("query", input={"question": q}) as root:
        answer = rag.query(q)
        root.update(output=answer)
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from src.monitoring.tracing import Step, set_langfuse_client, step

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """
    Tracer for RAG queries and LLM invocations using Langfuse or local logger fallback.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: Optional[bool] = None,
        client: Any = None,
    ):
        """
        Args:
            public_key: Langfuse public key (default: LANGFUSE_PUBLIC_KEY)
            secret_key: Langfuse secret key (default: LANGFUSE_SECRET_KEY)
            host: Langfuse server (default: LANGFUSE_HOST or Langfuse Cloud)
            enabled: Force tracing on or off (default: on when keys are set)
            client: An already configured ``langfuse.Langfuse`` client
        """
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = client is not None or bool(self.public_key and self.secret_key)

        self._client = client
        if self.enabled and self._client is None:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    base_url=self.host,
                )
                logger.info("Langfuse tracer initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse client: {e}. Falling back to local logging.")
                self.enabled = False

        if self.enabled and self._client is not None:
            # Pipeline steps anywhere in the process report to this client.
            set_langfuse_client(self._client)

    @property
    def client(self):
        return self._client if self.enabled else None

    @contextmanager
    def trace(
        self,
        name: str,
        input: Any = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Step]:
        """
        Root of one request. Steps started inside (also in threads started
        with ``asyncio.to_thread``) are its children. Langfuse shows the root's
        input and ``update(output=...)`` as the trace's input and output.
        """
        client = self.client
        with step(name, input=input, metadata=metadata) as root:
            if client is None:
                yield root
                return
            try:
                from langfuse import propagate_attributes

                attributes = propagate_attributes(user_id=user_id, session_id=session_id, trace_name=name)
                attributes.__enter__()
            except Exception as e:
                logger.debug(f"Langfuse trace attributes not set: {e}")
                attributes = None
            try:
                yield root
            finally:
                if attributes is not None:
                    try:
                        attributes.__exit__(None, None, None)
                    except Exception as e:
                        logger.debug(f"Langfuse trace attributes not reset: {e}")

    def flush(self) -> None:
        """Send buffered observations now (they are otherwise sent in the background)."""
        if self.client is not None:
            try:
                self._client.flush()
            except Exception as e:
                logger.error(f"Langfuse flush error: {e}")

    # Detached traces, for callers that cannot wrap their work in trace().

    def start_trace(
        self,
        name: str = "rag_query",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Start a trace record; end it with ``end_trace``. Pipeline steps do not nest under it."""
        trace_data = {
            "name": name,
            "user_id": user_id,
            "metadata": metadata or {},
            "input": input_data,
            "start_time": time.time(),
        }

        if self.client is not None:
            try:
                trace_data["_observation"] = self._client.start_observation(
                    name=name, as_type="span", input=input_data, metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Langfuse trace creation error: {e}")

        return trace_data

    def end_trace(
        self,
        trace_data: Dict[str, Any],
        output: Optional[Any] = None,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> float:
        """End a trace record and log execution metrics."""
        duration = time.time() - trace_data["start_time"]

        observation = trace_data.get("_observation")
        if observation is not None:
            try:
                if metadata_update:
                    observation.update(output=output, metadata=metadata_update)
                else:
                    observation.update(output=output)
                observation.end()
            except Exception as e:
                logger.error(f"Langfuse end trace error: {e}")

        logger.debug(f"Trace '{trace_data['name']}' completed in {duration:.3f}s")
        return duration

    def log_event(self, name: str, payload: Dict[str, Any]):
        """Log a custom event within monitoring."""
        if self.client is not None:
            try:
                self._client.create_event(name=name, metadata=payload)
            except Exception as e:
                logger.error(f"Langfuse event log error: {e}")
        else:
            logger.info(f"RAG Event [{name}]: {payload}")


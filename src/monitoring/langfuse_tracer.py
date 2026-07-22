"""
Langfuse Tracer Module
======================

Provides production tracing, cost tracking, and latency monitoring for LLM and RAG calls.
Integrates with Langfuse if keys are set in environment/config, otherwise degrades gracefully.

Usage:
    from src.monitoring.langfuse_tracer import LangfuseTracer

    tracer = LangfuseTracer()
    trace = tracer.start_trace(name="rag_query", user_id="user_123")
    ...
    tracer.end_trace(trace, output="Answer text")
"""

import os
import time
import logging
from typing import Optional, Dict, Any

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
    ):
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = bool(self.public_key and self.secret_key)

        self._client = None
        if self.enabled:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
                logger.info("Langfuse tracer initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse client: {e}. Falling back to local logging.")
                self.enabled = False

    def start_trace(
        self,
        name: str = "rag_query",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Start a new trace record."""
        trace_data = {
            "name": name,
            "user_id": user_id,
            "metadata": metadata or {},
            "input": input_data,
            "start_time": time.time(),
        }

        if self.enabled and self._client:
            try:
                lf_trace = self._client.trace(
                    name=name,
                    user_id=user_id,
                    metadata=metadata,
                    input=input_data,
                )
                trace_data["_lf_trace"] = lf_trace
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
        
        if self.enabled and "_lf_trace" in trace_data:
            try:
                lf_trace = trace_data["_lf_trace"]
                if metadata_update:
                    lf_trace.update(metadata=metadata_update, output=output)
                else:
                    lf_trace.update(output=output)
            except Exception as e:
                logger.error(f"Langfuse end trace error: {e}")

        logger.debug(f"Trace '{trace_data['name']}' completed in {duration:.3f}s")
        return duration

    def log_event(self, name: str, payload: Dict[str, Any]):
        """Log a custom event within monitoring."""
        if self.enabled and self._client:
            try:
                self._client.event(name=name, metadata=payload)
            except Exception as e:
                logger.error(f"Langfuse event log error: {e}")
        else:
            logger.info(f"RAG Event [{name}]: {payload}")

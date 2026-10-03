"""
FastAPI Application
===================

RESTful API for RAG system with real-time SSE streaming, rate limiting, and Langfuse tracing.

Endpoints:
- POST /query - Query the RAG system
- POST /query/stream - Stream query response via SSE
- POST /query/multimodal - Query with uploaded images/videos
- POST /query/multimodal/url - Query with image/video URLs
- GET /ox/status - Check dedicated Ox Alpha configuration
- POST /ox/chat - Query Ox Alpha with optional RAG context
- POST /ox/analyze - Analyze uploaded images/videos with Ox Alpha
- POST /ox/analyze/url - Analyze image/video URLs with Ox Alpha
- POST /documents - Add documents
- GET /documents - List documents
- POST /ingest - Ingest files
- POST /search - Search documents
- GET /health - Health check
- GET /ready - Readiness check

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from urllib.parse import urlsplit
import base64
import json
import logging
import os
import sys
import shutil
import uuid
import asyncio

# Keep Windows console output UTF-8 without replacing streams owned by pytest,
# notebooks, IDEs, or ASGI process managers.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty() and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src import __version__
from src.core.llm import LLMManager
from src.rag import NaiveRAG, AdvancedRAG
from src.auth import RateLimiter
from src.monitoring import LangfuseTracer
from src.utils.config import Config

logger = logging.getLogger(__name__)

IMAGE_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}
VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/webm",
}
MEDIA_CONTENT_TYPE_ALIASES = {
    "image/jpg": "image/jpeg",
    "video/quicktime": "video/mov",
}
MEDIA_EXTENSION_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mov": "video/mov",
    ".webm": "video/webm",
}


def _split_csv(value: Optional[str]) -> List[str]:
    """Split comma-separated config values."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _extract_api_key(request: Request) -> Optional[str]:
    """Read API key from X-API-Key or Bearer Authorization header."""
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    return None


def _safe_upload_name(filename: Optional[str]) -> str:
    """Return a safe local filename for an uploaded file."""
    safe_name = Path(filename or "upload").name
    safe_name = safe_name.replace("\x00", "").strip()
    return safe_name or "upload"


def _normalize_media_content_type(
    filename: Optional[str],
    content_type: Optional[str],
) -> tuple[str, str]:
    """Return provider media kind and normalized MIME type for an upload."""
    normalized = (content_type or "").split(";", 1)[0].strip().casefold()
    normalized = MEDIA_CONTENT_TYPE_ALIASES.get(normalized, normalized)

    if not normalized or normalized == "application/octet-stream":
        normalized = MEDIA_EXTENSION_TYPES.get(
            Path(filename or "").suffix.casefold(),
            "",
        )

    if normalized in IMAGE_CONTENT_TYPES:
        return "image", normalized
    if normalized in VIDEO_CONTENT_TYPES:
        return "video", normalized
    raise HTTPException(
        status_code=415,
        detail=(
            f"Unsupported media type for '{_safe_upload_name(filename)}': "
            f"{content_type or 'unknown'}"
        ),
    )


async def _read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload incrementally and reject it as soon as it exceeds the limit."""
    chunks = []
    total = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, max_bytes + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File '{_safe_upload_name(upload.filename)}' exceeds upload limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_remote_media_url(url: str) -> str:
    """Accept only remote HTTP(S) media references; the API never fetches them."""
    value = url.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail="Media URLs must use http:// or https://",
        )
    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=400,
            detail="Media URLs must not contain embedded credentials",
        )
    return value


# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="Question to ask")
    k: Optional[int] = Field(default=5, description="Number of documents to retrieve")
    transform_query: Optional[bool] = Field(default=True, description="Transform query for better retrieval")
    grade_documents: Optional[bool] = Field(default=True, description="Grade document relevance")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents")
    citations: List[Dict[str, Any]] = Field(default=[], description="Citation-ready source documents")
    transformed_query: Optional[str] = Field(default=None, description="Transformed query")


class MediaURLItem(BaseModel):
    """Remote image/video reference for a multimodal query."""
    type: Literal["image", "video"]
    url: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="Public HTTP(S) media URL",
    )


class MultimodalURLRequest(BaseModel):
    """Multimodal query using provider-accessible media URLs."""
    question: str = Field(..., min_length=1, description="Question about the media")
    media: List[MediaURLItem] = Field(..., min_length=1, description="Image/video URLs")
    k: int = Field(default=5, ge=1, le=50, description="Retrieved document count")
    use_retrieval: bool = Field(default=True, description="Enrich with indexed documents")


class OxChatRequest(BaseModel):
    """Text query handled explicitly by Ox Alpha."""
    question: str = Field(..., min_length=1, description="Question for Ox AI")
    k: int = Field(default=5, ge=1, le=50, description="Retrieved document count")
    use_retrieval: bool = Field(default=True, description="Enrich with indexed documents")


class OxQueryResponse(BaseModel):
    """Ox answer plus optional RAG citations."""
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None


class MultimodalQueryResponse(OxQueryResponse):
    """Ox/RAG answer plus non-sensitive media metadata."""
    media: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentRequest(BaseModel):
    """Request model for adding documents."""
    texts: List[str] = Field(..., description="List of text documents")
    metadatas: Optional[List[Dict[str, Any]]] = Field(default=None, description="Metadata for each document")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    count: Optional[int] = Field(default=None, description="Number of documents processed")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    rag_type: str = Field(..., description="RAG type being used")
    num_documents: int = Field(..., description="Number of documents in knowledge base")
    tracing_enabled: bool = Field(default=False, description="Whether Langfuse tracing is active")


class SearchResult(BaseModel):
    """Model for search result."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default={}, description="Document metadata")
    score: Optional[float] = Field(default=None, description="Relevance score")


# ============================================================================
# Application Factory
# ============================================================================

def create_app(
    rag_type: str = "advanced",
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
    embedding_provider: str = "huggingface",
    embedding_model: Optional[str] = None,
    vector_store_provider: Optional[str] = None,
    collection_name: Optional[str] = None,
    **kwargs
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        rag_type: Type of RAG ('naive' or 'advanced')
        llm_provider: LLM provider
        llm_model: LLM model name
        embedding_provider: Embedding provider
        embedding_model: Embedding model name
        vector_store_provider: Vector store backend
        collection_name: Vector store collection/index name
        **kwargs: Additional arguments

    Returns:
        FastAPI application
    """
    config = Config()
    ox_api_key = kwargs.pop(
        "ox_api_key",
        config.get("OPENROUTER_API_KEY") or None,
    )
    ox_base_url = kwargs.pop(
        "ox_base_url",
        config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    ox_model = kwargs.pop(
        "ox_model",
        config.get("OX_MODEL", "stealth/ox-alpha"),
    )
    ox_temperature = float(kwargs.pop(
        "ox_temperature",
        config.get_float("OX_TEMPERATURE", default=1.0),
    ))
    cors_origins = kwargs.pop(
        "cors_allow_origins",
        os.getenv("CORS_ORIGINS") or config.get("CORS_ALLOW_ORIGINS", ""),
    )
    if isinstance(cors_origins, str):
        cors_origins = _split_csv(cors_origins) or [
            "http://localhost:3000",
            "http://localhost:7860",
            "http://localhost:8000",
        ]

    enable_auth = kwargs.pop(
        "enable_auth",
        config.get_bool("ENABLE_API_AUTH", default=False),
    )
    api_keys = kwargs.pop("api_keys", None)
    if api_keys is None:
        api_keys = _split_csv(config.get("API_KEYS", ""))
    api_key_set = set(api_keys)

    rate_limit = int(kwargs.pop(
        "rate_limit",
        config.get_int("API_RATE_LIMIT", default=100),
    ))
    rate_limit_window = int(kwargs.pop(
        "rate_limit_window",
        config.get_int("API_RATE_LIMIT_WINDOW", default=60),
    ))
    max_upload_size_mb = max(1, int(kwargs.pop(
        "max_upload_size_mb",
        config.get_int("MAX_UPLOAD_SIZE_MB", default=25),
    )))
    max_upload_size_bytes = max_upload_size_mb * 1024 * 1024
    max_multimodal_files = max(1, int(kwargs.pop(
        "max_multimodal_files",
        config.get_int("MAX_MULTIMODAL_FILES", default=4),
    )))
    max_multimodal_total_size_mb = max(1, int(kwargs.pop(
        "max_multimodal_total_size_mb",
        config.get_int("MAX_MULTIMODAL_TOTAL_SIZE_MB", default=50),
    )))
    max_multimodal_total_size_bytes = max_multimodal_total_size_mb * 1024 * 1024
    vector_store_provider = vector_store_provider or kwargs.pop(
        "vector_store_provider",
        config.get("DEFAULT_VECTOR_STORE", "faiss"),
    )
    collection_name = collection_name or kwargs.pop(
        "collection_name",
        config.get("DEFAULT_COLLECTION_NAME", "default"),
    )
    persist_directory = kwargs.pop(
        "persist_directory",
        config.get("PERSIST_DIRECTORY"),
    )
    vector_store_url = kwargs.pop("vector_store_url", None)
    vector_store_api_key = kwargs.pop("vector_store_api_key", None)
    if vector_store_provider == "qdrant":
        vector_store_url = vector_store_url or config.get("QDRANT_URL") or None
        vector_store_api_key = (
            vector_store_api_key or config.get("QDRANT_API_KEY") or None
        )

    rate_limiter = RateLimiter(
        max_requests=rate_limit,
        window_seconds=rate_limit_window,
    )
    public_paths = {
        "/health", "/ready", "/ox/status", "/docs", "/redoc", "/openapi.json"
    }

    app = FastAPI(
        title="Ultimate RAG API",
        description="RESTful API for Retrieval-Augmented Generation with Streaming & Tracing",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Langfuse Tracer
    tracer = LangfuseTracer()

    @app.middleware("http")
    async def api_guard(request: Request, call_next):
        """Optional API-key auth plus sliding-window rate limiting."""
        path = request.url.path
        if path in public_paths:
            return await call_next(request)

        identity = request.client.host if request.client else "anonymous"

        if enable_auth:
            api_key = _extract_api_key(request)
            if not api_key or api_key not in api_key_set:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid API key"},
                )
            identity = api_key

        if not rate_limiter.is_allowed(identity):
            reset_in = rate_limiter.get_reset_time(identity)
            headers = {}
            if reset_in is not None:
                headers["Retry-After"] = str(max(1, int(reset_in)))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers=headers,
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(
            rate_limiter.get_remaining(identity)
        )
        return response

    # Initialize RAG
    if rag_type == "advanced":
        rag = AdvancedRAG(
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            vector_store_provider=vector_store_provider,
            collection_name=collection_name,
            persist_directory=persist_directory,
            vector_store_url=vector_store_url,
            vector_store_api_key=vector_store_api_key,
            **kwargs
        )
    else:
        rag = NaiveRAG(
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            vector_store_provider=vector_store_provider,
            collection_name=collection_name,
            persist_directory=persist_directory,
            vector_store_url=vector_store_url,
            vector_store_api_key=vector_store_api_key,
            **kwargs
        )

    # Store RAG instance and tracer
    app.state.rag = rag
    app.state.rag_type = rag_type
    app.state.tracer = tracer
    app.state.ox_llm = LLMManager(
        provider="openai",
        model=ox_model,
        api_key=ox_api_key,
        base_url=ox_base_url,
        temperature=ox_temperature,
    )
    app.state.ox_configured = bool(ox_api_key)

    # ========================================================================
    # Endpoints
    # ========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version=__version__,
            rag_type=app.state.rag_type,
            num_documents=app.state.rag.num_documents,
            tracing_enabled=app.state.tracer.enabled,
        )

    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """Readiness check for container orchestration."""
        vector_config = getattr(app.state.rag.vector_store, "config", None)
        vector_provider = getattr(vector_config, "provider", "unknown")
        checks = {
            "rag_initialized": app.state.rag is not None,
            "vector_store_provider": vector_provider,
        }

        if vector_provider == "qdrant":
            try:
                # Same client the RAG uses: server, embedded on-disk or in-memory.
                health = app.state.rag.vector_store.health()
                checks["qdrant"] = "ready"
                checks["qdrant_points"] = health.get("points")
            except Exception as e:
                checks["qdrant"] = "unavailable"
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "checks": checks,
                        "detail": str(e),
                    },
                )

        return {"status": "ready", "checks": checks}

    @app.get("/ox/status", tags=["Ox AI"])
    async def ox_status():
        """Report local Ox configuration without calling or exposing credentials."""
        return {
            "status": "configured" if app.state.ox_configured else "not_configured",
            "configured": app.state.ox_configured,
            "provider": "openrouter",
            "model": app.state.ox_llm.config.model,
            "modalities": ["text", "image", "video"],
        }

    @app.post("/query", response_model=QueryResponse, tags=["RAG"])
    def query_rag(request: QueryRequest):
        """
        Query the RAG system synchronously.
        """
        trace = app.state.tracer.start_trace(
            name="query_rag",
            input_data={"question": request.question, "k": request.k},
        )
        try:
            if app.state.rag_type == "advanced":
                result = app.state.rag.query_detailed(
                    question=request.question,
                    k=request.k,
                    transform_query=request.transform_query,
                    grade_documents=request.grade_documents,
                )
                res = QueryResponse(
                    answer=result["answer"],
                    sources=result["relevant_docs"],
                    citations=result.get("citations", result["relevant_docs"]),
                    transformed_query=result.get("transformed_query"),
                )
            else:
                result = app.state.rag.query_with_sources(
                    question=request.question,
                    k=request.k,
                )
                citations = [
                    {
                        "source_id": f"S{i}",
                        "source": source.get("metadata", {}).get("source", f"Document {i}"),
                        **source,
                    }
                    for i, source in enumerate(result["sources"], 1)
                ]
                res = QueryResponse(
                    answer=result["answer"],
                    sources=result["sources"],
                    citations=citations,
                )

            app.state.tracer.end_trace(trace, output=res.answer)
            return res
        except Exception as e:
            logger.error(f"Query failed: {e}")
            app.state.tracer.end_trace(trace, output=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query/stream", tags=["RAG"])
    async def query_rag_stream(request: QueryRequest):
        """
        Stream RAG query response in real-time using Server-Sent Events (SSE).
        """
        trace = app.state.tracer.start_trace(
            name="query_rag_stream",
            input_data={"question": request.question},
        )

        async def sse_event_generator():
            try:
                # First retrieve sources asynchronously
                docs = await asyncio.to_thread(app.state.rag.retrieve, request.question, request.k)
                sources = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
                
                # Emit sources metadata first
                meta_event = json.dumps({"status": "sources", "sources": sources})
                yield f"data: {meta_event}\n\n"

                # Stream response text
                if hasattr(app.state.rag, "stream"):
                    for token in app.state.rag.stream(request.question):
                        chunk_event = json.dumps({"status": "generating", "chunk": token})
                        yield f"data: {chunk_event}\n\n"
                        await asyncio.sleep(0.01)
                else:
                    # Fallback if streaming not supported
                    ans = await asyncio.to_thread(app.state.rag.query, request.question)
                    chunk_event = json.dumps({"status": "generating", "chunk": ans})
                    yield f"data: {chunk_event}\n\n"

                done_event = json.dumps({"status": "done"})
                yield f"data: {done_event}\n\n"
                app.state.tracer.end_trace(trace, output="SSE streaming finished")
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                err_event = json.dumps({"status": "error", "detail": str(e)})
                yield f"data: {err_event}\n\n"
                app.state.tracer.end_trace(trace, output=str(e))

        return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

    def require_ox_configuration() -> None:
        """Fail before provider access when the dedicated Ox key is absent."""
        if not app.state.ox_configured:
            raise HTTPException(
                status_code=503,
                detail="Ox AI is not configured. Set OPENROUTER_API_KEY.",
            )

    def format_retrieved_documents(docs) -> tuple[str, List[Dict[str, Any]]]:
        """Build citation-ready context shared by the dedicated Ox text route."""
        context_parts = []
        sources = []
        for index, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("url")
                or f"Document {index}"
            )
            context_parts.append(f"[S{index}] Source: {source}\n{doc.page_content}")
            sources.append({
                "source_id": f"S{index}",
                "source": source,
                "content": (
                    doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content
                ),
                "metadata": metadata,
            })
        return "\n\n".join(context_parts), sources

    async def prepare_multimodal_uploads(
        files: List[UploadFile],
    ) -> tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Validate uploads and turn them into provider-compatible data URLs."""
        if not files:
            raise HTTPException(status_code=400, detail="At least one media file is required")
        if len(files) > max_multimodal_files:
            raise HTTPException(
                status_code=413,
                detail=f"At most {max_multimodal_files} media files are allowed",
            )

        media = []
        media_metadata = []
        total_size = 0
        for upload in files:
            media_type, content_type = _normalize_media_content_type(
                upload.filename,
                upload.content_type,
            )
            content = await _read_upload_limited(upload, max_upload_size_bytes)
            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{_safe_upload_name(upload.filename)}' is empty",
                )

            total_size += len(content)
            if total_size > max_multimodal_total_size_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Combined media size exceeds "
                        f"{max_multimodal_total_size_mb} MB limit"
                    ),
                )

            encoded = base64.b64encode(content).decode("ascii")
            media.append({
                "type": media_type,
                "url": f"data:{content_type};base64,{encoded}",
            })
            media_metadata.append({
                "name": _safe_upload_name(upload.filename),
                "type": media_type,
                "content_type": content_type,
                "size_bytes": len(content),
                "source": "upload",
            })
        return media, media_metadata

    def prepare_multimodal_urls(
        request: MultimodalURLRequest,
    ) -> tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Validate remote references without fetching them on this server."""
        if len(request.media) > max_multimodal_files:
            raise HTTPException(
                status_code=413,
                detail=f"At most {max_multimodal_files} media URLs are allowed",
            )
        media = [
            {"type": item.type, "url": _validate_remote_media_url(item.url)}
            for item in request.media
        ]
        metadata = [{"type": item.type, "source": "url"} for item in request.media]
        return media, metadata

    async def execute_multimodal_query(
        question: str,
        media: List[Dict[str, str]],
        media_metadata: List[Dict[str, Any]],
        k: int,
        use_retrieval: bool,
        llm: Optional[LLMManager] = None,
        trace_name: str = "query_rag_multimodal",
    ) -> MultimodalQueryResponse:
        """Run the shared multimodal RAG path without logging media payloads."""
        question = question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        if not hasattr(app.state.rag, "query_multimodal"):
            raise HTTPException(
                status_code=501,
                detail="The configured RAG pipeline does not support multimodal queries",
            )

        trace = app.state.tracer.start_trace(
            name=trace_name,
            input_data={
                "question": question,
                "media_count": len(media),
                "use_retrieval": use_retrieval,
            },
        )
        try:
            query_kwargs = {
                "question": question,
                "media": media,
                "k": k,
                "use_retrieval": use_retrieval,
            }
            if llm is not None:
                query_kwargs["llm"] = llm
            result = await asyncio.to_thread(
                app.state.rag.query_multimodal,
                **query_kwargs,
            )
            response = MultimodalQueryResponse(
                answer=result["answer"],
                sources=result.get("sources", []),
                citations=result.get("citations", result.get("sources", [])),
                media=media_metadata,
                model=result.get("model"),
            )
            app.state.tracer.end_trace(trace, output=response.answer)
            return response
        except HTTPException:
            raise
        except Exception as e:
            error_name = type(e).__name__
            logger.error("Multimodal provider request failed (%s)", error_name)
            app.state.tracer.end_trace(trace, output=f"Provider error: {error_name}")
            raise HTTPException(
                status_code=502,
                detail=f"Multimodal provider request failed ({error_name})",
            )

    @app.post(
        "/query/multimodal",
        response_model=MultimodalQueryResponse,
        tags=["Multimodal"],
    )
    async def query_multimodal_upload(
        question: str = Form(..., min_length=1),
        files: List[UploadFile] = File(...),
        k: int = Form(default=5, ge=1, le=50),
        use_retrieval: bool = Form(default=True),
    ):
        """Analyze uploaded images/videos and optionally enrich with RAG context."""
        try:
            media, media_metadata = await prepare_multimodal_uploads(files)
            return await execute_multimodal_query(
                question.strip(),
                media,
                media_metadata,
                k,
                use_retrieval,
            )
        finally:
            for upload in files:
                try:
                    await upload.close()
                except Exception:
                    pass

    @app.post(
        "/query/multimodal/url",
        response_model=MultimodalQueryResponse,
        tags=["Multimodal"],
    )
    async def query_multimodal_url(request: MultimodalURLRequest):
        """Analyze public image/video URLs without downloading them in this API."""
        media, media_metadata = prepare_multimodal_urls(request)
        return await execute_multimodal_query(
            request.question.strip(),
            media,
            media_metadata,
            request.k,
            request.use_retrieval,
        )

    @app.post("/ox/chat", response_model=OxQueryResponse, tags=["Ox AI"])
    async def ox_chat(request: OxChatRequest):
        """Ask Ox Alpha a text question, optionally grounded in this RAG index."""
        require_ox_configuration()
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        trace = app.state.tracer.start_trace(
            name="ox_chat",
            input_data={
                "question": question,
                "use_retrieval": request.use_retrieval,
            },
        )
        try:
            docs = []
            if request.use_retrieval and app.state.rag.num_chunks:
                docs = await asyncio.to_thread(
                    app.state.rag.retrieve,
                    question,
                    k=request.k,
                )
            context, sources = format_retrieved_documents(docs)
            prompt = (
                "Answer the user's question accurately. "
                "Use the RAG context when it is relevant and cite its source IDs "
                "such as [S1]. Do not invent citations.\n\n"
                f"RAG context:\n{context or '(no retrieved context)'}\n\n"
                f"User question:\n{question}"
            )
            answer = await asyncio.to_thread(app.state.ox_llm.generate, prompt)
            response = OxQueryResponse(
                answer=answer,
                sources=sources,
                citations=sources,
                model=app.state.ox_llm.config.model,
            )
            app.state.tracer.end_trace(trace, output=response.answer)
            return response
        except HTTPException:
            raise
        except Exception as e:
            error_name = type(e).__name__
            logger.error("Ox provider request failed (%s)", error_name)
            app.state.tracer.end_trace(trace, output=f"Provider error: {error_name}")
            raise HTTPException(
                status_code=502,
                detail=f"Ox provider request failed ({error_name})",
            )

    @app.post(
        "/ox/analyze",
        response_model=MultimodalQueryResponse,
        tags=["Ox AI"],
    )
    async def ox_analyze_upload(
        question: str = Form(..., min_length=1),
        files: List[UploadFile] = File(...),
        k: int = Form(default=5, ge=1, le=50),
        use_retrieval: bool = Form(default=True),
    ):
        """Analyze uploaded images/videos with Ox Alpha and optional RAG context."""
        require_ox_configuration()
        try:
            media, media_metadata = await prepare_multimodal_uploads(files)
            return await execute_multimodal_query(
                question.strip(),
                media,
                media_metadata,
                k,
                use_retrieval,
                llm=app.state.ox_llm,
                trace_name="ox_analyze_upload",
            )
        finally:
            for upload in files:
                try:
                    await upload.close()
                except Exception:
                    pass

    @app.post(
        "/ox/analyze/url",
        response_model=MultimodalQueryResponse,
        tags=["Ox AI"],
    )
    async def ox_analyze_url(request: MultimodalURLRequest):
        """Analyze public image/video URLs with Ox Alpha and optional RAG context."""
        require_ox_configuration()
        media, media_metadata = prepare_multimodal_urls(request)
        return await execute_multimodal_query(
            request.question.strip(),
            media,
            media_metadata,
            request.k,
            request.use_retrieval,
            llm=app.state.ox_llm,
            trace_name="ox_analyze_url",
        )

    @app.post("/documents", response_model=DocumentResponse, tags=["Documents"])
    def add_documents(request: DocumentRequest):
        """Add text documents to the knowledge base."""
        try:
            num_chunks = app.state.rag.add_texts(
                texts=request.texts,
                metadatas=request.metadatas,
            )
            return DocumentResponse(
                status="success",
                message=f"Added {num_chunks} chunks from {len(request.texts)} documents",
                count=num_chunks,
            )
        except Exception as e:
            logger.error(f"Add documents failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/documents", tags=["Documents"])
    async def list_documents():
        """List basic information about indexed documents."""
        return {
            "num_documents": app.state.rag.num_documents,
            "num_chunks": app.state.rag.num_chunks,
        }

    @app.post("/ingest", response_model=DocumentResponse, tags=["Documents"])
    async def ingest_files(files: List[UploadFile] = File(...)):
        """Ingest files into the knowledge base."""
        temp_dir = Path("temp_uploads")
        file_paths = []
        try:
            temp_dir.mkdir(exist_ok=True)

            for file in files:
                content = await file.read()
                if len(content) > max_upload_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File '{file.filename}' exceeds {max_upload_size_mb} MB limit",
                    )

                safe_name = _safe_upload_name(file.filename)
                file_path = temp_dir / f"{uuid.uuid4().hex}_{safe_name}"
                file_path.write_bytes(content)
                file_paths.append(str(file_path))

            num_chunks = await asyncio.to_thread(
                app.state.rag.add_documents, file_paths
            )

            return DocumentResponse(
                status="success",
                message=f"Ingested {len(files)} files with {num_chunks} chunks",
                count=num_chunks,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File ingestion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for file_path in file_paths:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    @app.post("/search", tags=["RAG"])
    def search_documents(
        query: str = Query(..., description="Search query"),
        k: int = Query(default=5, description="Number of results"),
    ):
        """Search for relevant documents without generating an answer."""
        try:
            docs = app.state.rag.retrieve(query, k=k)
            results = [
                SearchResult(content=doc.page_content, metadata=doc.metadata)
                for doc in docs
            ]
            return {"results": results, "count": len(results)}
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Default application instance for uvicorn
config = Config()
app = create_app(
    rag_type="advanced",
    llm_provider=config.get("DEFAULT_LLM_PROVIDER", "openai"),
    llm_model=config.get("DEFAULT_LLM_MODEL"),
    embedding_provider=config.get("DEFAULT_EMBEDDING_PROVIDER", "huggingface"),
    embedding_model=config.get("DEFAULT_EMBEDDING_MODEL"),
    vector_store_provider=config.get("DEFAULT_VECTOR_STORE", "faiss"),
    collection_name=config.get("DEFAULT_COLLECTION_NAME", "default"),
    chunk_size=config.get_int("CHUNK_SIZE", default=500),
    chunk_overlap=config.get_int("CHUNK_OVERLAP", default=50),
    retrieval_k=config.get_int("RETRIEVAL_K", default=5),
    use_hybrid=config.get_bool("ENABLE_HYBRID_SEARCH", default=True),
    use_reranking=config.get_bool("ENABLE_RERANKING", default=True),
    use_parent_context=config.get_bool("ENABLE_PARENT_CONTEXT", default=True),
    use_cache=config.get_bool("ENABLE_CACHE", default=False),
    cache_ttl=config.get_int("CACHE_TTL", default=3600),
)

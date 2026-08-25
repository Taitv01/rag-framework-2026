"""
FastAPI Application
===================

RESTful API for RAG system with real-time SSE streaming, rate limiting, and Langfuse tracing.

Endpoints:
- POST /query - Query the RAG system
- POST /query/stream - Stream query response via SSE
- POST /documents - Add documents
- GET /documents - List documents
- POST /ingest - Ingest files
- POST /search - Search documents
- GET /health - Health check
- GET /ready - Readiness check

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import logging
import os
import shutil
import uuid
import asyncio

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.rag import NaiveRAG, AdvancedRAG
from src.auth import RateLimiter
from src.monitoring import LangfuseTracer
from src.utils.config import Config

logger = logging.getLogger(__name__)


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
    llm_model: str = "gpt-4o-mini",
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
    max_upload_size_mb = int(kwargs.pop(
        "max_upload_size_mb",
        config.get_int("MAX_UPLOAD_SIZE_MB", default=25),
    ))
    max_upload_size_bytes = max_upload_size_mb * 1024 * 1024
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
    public_paths = {"/health", "/ready", "/docs", "/redoc", "/openapi.json"}

    app = FastAPI(
        title="Ultimate RAG API",
        description="RESTful API for Retrieval-Augmented Generation with Streaming & Tracing",
        version="1.1.0",
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

    # ========================================================================
    # Endpoints
    # ========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.1.0",
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
                from qdrant_client import QdrantClient

                client = QdrantClient(
                    url=getattr(vector_config, "url", None) or os.getenv("QDRANT_URL"),
                    api_key=(
                        getattr(vector_config, "api_key", None)
                        or os.getenv("QDRANT_API_KEY")
                        or None
                    ),
                    timeout=3,
                )
                client.get_collections()
                checks["qdrant"] = "ready"
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
    llm_model=config.get("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
    embedding_provider=config.get("DEFAULT_EMBEDDING_PROVIDER", "huggingface"),
    embedding_model=config.get("DEFAULT_EMBEDDING_MODEL", "BAAI/bge-m3"),
    vector_store_provider=config.get("DEFAULT_VECTOR_STORE", "faiss"),
    collection_name=config.get("DEFAULT_COLLECTION_NAME", "default"),
    chunk_size=config.get_int("CHUNK_SIZE", default=500),
    chunk_overlap=config.get_int("CHUNK_OVERLAP", default=50),
    retrieval_k=config.get_int("RETRIEVAL_K", default=5),
    use_hybrid=config.get_bool("ENABLE_HYBRID_SEARCH", default=True),
    use_reranking=config.get_bool("ENABLE_RERANKING", default=True),
    use_cache=config.get_bool("ENABLE_CACHE", default=False),
    cache_ttl=config.get_int("CACHE_TTL", default=3600),
)

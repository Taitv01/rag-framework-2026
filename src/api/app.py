"""
FastAPI Application
===================

RESTful API for RAG system.

Endpoints:
- POST /query - Query the RAG system
- POST /documents - Add documents
- GET /documents - List documents
- POST /ingest - Ingest files
- GET /health - Health check

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import io
import logging
import os
import sys
import shutil
import uuid

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.rag import NaiveRAG, AdvancedRAG
from src.auth import RateLimiter
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
    rate_limiter = RateLimiter(
        max_requests=rate_limit,
        window_seconds=rate_limit_window,
    )
    public_paths = {"/health", "/docs", "/redoc", "/openapi.json"}

    app = FastAPI(
        title="Ultimate RAG API",
        description="RESTful API for Retrieval-Augmented Generation",
        version="1.0.0",
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
            **kwargs
        )
    else:
        rag = NaiveRAG(
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            **kwargs
        )

    # Store RAG instance
    app.state.rag = rag
    app.state.rag_type = rag_type

    # ========================================================================
    # Endpoints
    # ========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint.

        Returns service status and basic information.
        """
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            rag_type=app.state.rag_type,
            num_documents=app.state.rag.num_documents,
        )

    @app.post("/query", response_model=QueryResponse, tags=["RAG"])
    def query_rag(request: QueryRequest):
        """
        Query the RAG system.

        Send a question and receive an answer with sources.
        Note: Uses sync def (not async) to avoid blocking the event loop.
        FastAPI runs sync endpoints in a thread pool automatically.
        """
        try:
            if app.state.rag_type == "advanced":
                result = app.state.rag.query_detailed(
                    question=request.question,
                    k=request.k,
                    transform_query=request.transform_query,
                    grade_documents=request.grade_documents,
                )
                return QueryResponse(
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
                return QueryResponse(
                    answer=result["answer"],
                    sources=result["sources"],
                    citations=citations,
                )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/documents", response_model=DocumentResponse, tags=["Documents"])
    def add_documents(request: DocumentRequest):
        """
        Add text documents to the knowledge base.

        Send a list of texts to be indexed.
        Note: Uses sync def to avoid blocking the event loop.
        """
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
        """
        List documents in the knowledge base.

        Returns basic information about indexed documents.
        """
        return {
            "num_documents": app.state.rag.num_documents,
            "num_chunks": app.state.rag.num_chunks,
        }

    @app.post("/ingest", response_model=DocumentResponse, tags=["Documents"])
    async def ingest_files(files: List[UploadFile] = File(...)):
        """
        Ingest files into the knowledge base.

        Upload files (PDF, TXT, MD, etc.) to be indexed.
        """
        temp_dir = Path("temp_uploads")
        file_paths = []
        try:
            # Save uploaded files temporarily
            temp_dir.mkdir(exist_ok=True)

            for file in files:
                content = await file.read()
                if len(content) > max_upload_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File '{file.filename}' exceeds "
                            f"{max_upload_size_mb} MB upload limit"
                        ),
                    )

                safe_name = _safe_upload_name(file.filename)
                file_path = temp_dir / f"{uuid.uuid4().hex}_{safe_name}"
                file_path.write_bytes(content)
                file_paths.append(str(file_path))

            # Add documents (run in thread pool to avoid blocking)
            import asyncio
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
            # Always clean up temp files
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
        """
        Search for relevant documents without generating an answer.

        Returns list of relevant documents.
        Note: Uses sync def to avoid blocking the event loop.
        """
        try:
            docs = app.state.rag.retrieve(query, k=k)

            results = []
            for doc in docs:
                results.append(SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                ))

            return {"results": results, "count": len(results)}
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ============================================================================
# Default Application Instance
# ============================================================================

# Create default app for uvicorn
config = Config()
app = create_app(
    rag_type="advanced",
    llm_provider=config.get("DEFAULT_LLM_PROVIDER", "openai"),
    llm_model=config.get("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
    embedding_provider=config.get("DEFAULT_EMBEDDING_PROVIDER", "huggingface"),
    embedding_model=config.get("DEFAULT_EMBEDDING_MODEL", "keepitreal/vietnamese-sbert"),
)

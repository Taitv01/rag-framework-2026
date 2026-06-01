"""
Example 06: API Server
======================

Start a REST API server for the RAG system.

This example demonstrates:
1. Starting a FastAPI server
2. API endpoints for querying
3. Document upload via API
4. Health check endpoint

Requirements:
    pip install -r requirements.txt

Usage:
    python examples/06_api_server.py

    Then visit:
    - API docs: http://localhost:8000/docs
    - Health: http://localhost:8000/health
"""

import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Start API server."""

    print("=" * 60)
    print("RAG API Server")
    print("=" * 60)

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install uvicorn")
        return

    print("\n🚀 Starting API server...")
    print("\n📚 API Documentation:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("   - Health: http://localhost:8000/health")
    print("\n📌 Available Endpoints:")
    print("   - POST /query - Query the RAG system")
    print("   - POST /documents - Add documents")
    print("   - GET /documents - List documents")
    print("   - POST /ingest - Upload files")
    print("   - POST /search - Search documents")
    print("   - GET /health - Health check")
    print("\n" + "=" * 60)

    # Start server
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()

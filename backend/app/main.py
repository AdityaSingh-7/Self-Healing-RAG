"""
main.py — FastAPI Application Entry Point

THIS IS WHERE EVERYTHING STARTS.
When you run `uvicorn app.main:app`, Python:
1. Imports this file
2. Finds the `app` object
3. Starts listening for HTTP requests on port 8000

WHAT THIS FILE DOES:
- Creates the FastAPI app
- Sets up CORS (so the frontend can talk to us)
- Includes all our route handlers (URL endpoints)
- Defines a health check endpoint for testing
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

# ============================================
# CREATE THE APP
# ============================================
# This is the main FastAPI application object.
# Everything attaches to this — routes, middleware, etc.
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation with semantic search, recency awareness, and hybrid retrieval.",
    version="1.0.0",
)

# ============================================
# CORS MIDDLEWARE
# ============================================
# WHAT: Cross-Origin Resource Sharing
# WHY:  Your frontend runs on localhost:3000
#        Your backend runs on localhost:8000
#        Browsers BLOCK requests between different origins (ports count!)
#        CORS says "it's okay, I trust this origin"
#
# WITHOUT THIS: Frontend fetch() calls get blocked by browser → nothing works
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],  # Only our frontend can talk to us
    allow_credentials=True,                  # Allow cookies/auth headers
    allow_methods=["*"],                     # Allow GET, POST, DELETE, etc.
    allow_headers=["*"],                     # Allow any headers (including Authorization)
)


# ============================================
# HEALTH CHECK ENDPOINT
# ============================================
# This is the simplest possible endpoint.
# It exists so you can verify the server is running:
#   curl http://localhost:8000/health
#   → {"status": "healthy", "version": "1.0.0"}
@app.get("/health")
async def health_check():
    """
    Health dashboard — returns system status, component health, and stats.
    """
    import time
    from app.services.cache import semantic_cache
    from app.services.analytics import get_analytics_summary

    # Check component status
    components = {}

    # Pinecone status
    try:
        from app.services.vectorstore import VectorStoreService
        store = VectorStoreService()
        if store.index is not None:
            components["pinecone"] = "connected"
        else:
            components["pinecone"] = "not_configured"
    except Exception as e:
        components["pinecone"] = f"error: {str(e)}"

    # Embedding model status
    try:
        from app.services.embedder import EmbeddingService
        if EmbeddingService._model is not None:
            components["embedding_model"] = "loaded"
        else:
            components["embedding_model"] = "not_loaded_yet"
    except Exception:
        components["embedding_model"] = "error"

    # Cache stats
    components["semantic_cache"] = {
        "entries": semantic_cache.size,
        "threshold": semantic_cache.threshold,
    }

    # Analytics summary (last 24h)
    try:
        stats = get_analytics_summary(days=1)
    except Exception:
        stats = {}

    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": components,
        "stats_24h": stats,
    }


# ============================================
# ROOT ENDPOINT
# ============================================
@app.get("/")
async def root():
    """
    Base URL — just redirects people to the docs.
    Visit http://localhost:8000/docs for interactive API documentation.
    """
    return {
        "message": "RAG System API",
        "docs": "Visit /docs for interactive API documentation",
        "health": "Visit /health to check server status",
    }


# ============================================
# INCLUDE ROUTERS
# ============================================
# Routers are like chapters in a book — each one handles a group of related URLs.
# `prefix="/ingest"` means all routes in ingest.py start with /ingest
# So ingest.py's @router.post("/upload") becomes POST /ingest/upload

from app.routers import ingest, query, documents, evaluate, analytics_router, export, workspaces, healing_query

app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query (Standard)"])
app.include_router(healing_query.router, prefix="/healing", tags=["Self-Healing Query"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["Evaluation"])
app.include_router(analytics_router.router, prefix="/analytics", tags=["Analytics"])
app.include_router(export.router, prefix="/export", tags=["Export"])
app.include_router(workspaces.router, prefix="/workspaces", tags=["Workspaces"])

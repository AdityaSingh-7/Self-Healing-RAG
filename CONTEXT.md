# RAG System — Claude Code Context File

> **Purpose:** Feed this file to Claude Code at the start of any new session.
> It tells Claude where the project is, what's been built, and what's next.

---

## Project Overview

- **What:** Full-stack RAG (Retrieval-Augmented Generation) system
- **Location:** `~/Desktop/rag-system`
- **Stack:** LlamaIndex, Pinecone, Groq (Llama 3.3 70B), sentence-transformers (all-MiniLM-L6-v2), FastAPI, Next.js, Tailwind, NextAuth

## Features

- [x] Architecture designed
- [x] Backend scaffold (FastAPI + config + health check)
- [x] Document ingestion (PDF parse → chunk → embed → Pinecone upsert)
- [x] Query pipeline (embed → search → recency rerank → Groq LLM → SSE stream)
- [x] Conversation memory (sliding window)
- [x] Query rewriting (LLM reformulates vague follow-ups)
- [x] Auth middleware (JWT verification — not wired to endpoints yet)
- [x] Document management (list/delete endpoints)
- [x] Schemas (Pydantic request/response validation)
- [ ] RAGAS evaluation endpoint
- [ ] Frontend (Next.js + Tailwind + NextAuth)
- [ ] Wire auth to all endpoints (currently hardcoded "default_user")
- [ ] Pinecone connection (needs real API key)
- [ ] Groq connection (needs real API key)
- [ ] Docker Compose + README

## Architecture

```
PDF → Parse (PyMuPDF) → Chunk (LlamaIndex) → Embed (MiniLM) → Pinecone (dense + sparse)
Query → Rewrite → Embed → Hybrid Search → Recency Rerank → Groq/Llama 3 → SSE Stream → Frontend
```

## Endpoints

```
GET      /health              — Server status
GET      /                    — API info
POST     /ingest/upload       — Upload & process PDF
POST     /query/ask           — Ask question (streaming SSE or JSON)
POST     /query/clear-history — Reset conversation
GET      /documents/          — List uploaded docs
DELETE   /documents/{doc_id}  — Delete a document
```

## Key Files

```
backend/
├── app/
│   ├── main.py              — FastAPI app, CORS, router inclusion
│   ├── config.py            — Pydantic settings from .env
│   ├── schemas/             — Request/response Pydantic models
│   │   ├── query.py         — QueryRequest, QueryResponse, etc.
│   │   └── documents.py     — IngestResponse, DeleteResponse
│   ├── routers/             — URL endpoint handlers
│   │   ├── ingest.py        — POST /ingest/upload
│   │   ├── query.py         — POST /query/ask + SSE streaming
│   │   └── documents.py     — GET/DELETE /documents
│   ├── services/            — Business logic
│   │   ├── parser.py        — PDF → text (PyMuPDF)
│   │   ├── chunker.py       — Text → chunks (LlamaIndex SentenceSplitter)
│   │   ├── embedder.py      — Text → vectors (MiniLM, singleton)
│   │   ├── vectorstore.py   — Pinecone upsert/search + recency scoring
│   │   ├── llm.py           — Groq/Llama 3 (sync + streaming)
│   │   ├── memory.py        — Conversation history (sliding window)
│   │   └── rewriter.py      — Query rewriting for follow-ups
│   └── middleware/
│       └── auth.py          — JWT verification (get_current_user dependency)
├── models/
│   └── all-MiniLM-L6-v2/   — Local embedding model (downloaded)
├── requirements.txt
├── .env                     — Secrets (placeholder keys currently)
└── .env.example             — Template for secrets
```

## Important Notes

- Embedding model is stored LOCALLY at `./models/all-MiniLM-L6-v2` (corporate firewall blocks HuggingFace)
- pip is configured to use Artifactory mirror (`~/.pip/pip.conf`)
- Pinecone and Groq API keys are placeholders — need real keys to test full flow
- Server runs on port 8001 (8000 is taken by something else)

## Current Status

**Step:** Backend complete, ready for frontend
**Blockers:** Need Pinecone API key + Groq API key for end-to-end testing
**Last updated:** 2026-06-22

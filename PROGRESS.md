# RAG System — Progress & Learning Journal

> **Purpose:** YOUR reference. Documents what we built, why, and how it works.
> Read this when you need to remember what a piece does or explain the project.

---

## 📅 Day 1 — Project Setup & Backend Scaffold

### What We're Doing Today

Setting up the project folder structure and creating the FastAPI backend — the "brain" of our system that will handle file uploads, search queries, and talk to all the AI services.

---

### Step 1: Project Structure Created

```
~/Desktop/rag-system/
├── CONTEXT.md          ← For Claude Code (session continuity)
├── PROGRESS.md         ← For YOU (this file — learning journal)
├── backend/            ← Python FastAPI server (we're building this now)
│   ├── app/            ← All our Python code lives here
│   │   ├── main.py     ← Entry point — starts the server
│   │   ├── config.py   ← Settings (API keys, etc.) loaded from .env
│   │   ├── routers/    ← URL endpoints (like /ingest, /query)
│   │   └── services/   ← Business logic (parsing, embedding, etc.)
│   ├── requirements.txt ← Python packages we need
│   └── .env.example    ← Template for secret keys
└── frontend/           ← Next.js React app (later)
```

**Why this structure?**
- Separating `routers/` (what URLs exist) from `services/` (what the code does) is called "separation of concerns" — if you want to change HOW you parse a PDF, you only touch `services/parser.py`, not the URL handler
- `.env.example` shows what keys are needed without exposing real secrets

---

### Key Concepts Learned

| Concept | Explanation |
|---------|-------------|
| **FastAPI** | A Python framework for building web APIs. You define URL endpoints and FastAPI handles incoming requests. |
| **Pydantic** | A library for data validation. You define the "shape" of your data and it automatically rejects bad input. |
| **Environment variables** | Secrets (API keys) stored in a `.env` file, NOT in code. Never commit secrets to git. |
| **CORS** | Cross-Origin Resource Sharing — allows your frontend (localhost:3000) to talk to your backend (localhost:8000). Without it, the browser blocks the request. |
| **Uvicorn** | The server that actually runs your FastAPI app. FastAPI defines the routes, uvicorn listens for incoming connections. |

---

# 🩺 Self-Healing RAG

> A RAG system that **validates its own answers**, **automatically retries with adaptive strategies** when confidence is low, and **learns which strategies work** over time.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Llama 3](https://img.shields.io/badge/Llama_3-70B-purple)

## The Problem with Standard RAG

Every RAG system you've seen does this:

```
Question → Search → LLM → Answer → Done
```

**What happens when it's wrong?** Nothing. The system returns confidently incorrect answers with no self-awareness.

## This System Does This Instead

```
Question → Search → LLM → Answer → VALIDATE → Confident? 
                                                    │
                                          YES → Return ✅
                                          NO  → HEAL → Retry → VALIDATE → ...
                                                    │
                                          STILL NO → Admit uncertainty gracefully
```

## How It Works

### 1. Validation (LLM-as-Judge)

After generating an answer, a second LLM call evaluates:
- Does the answer address the **specific** question?
- Is it **grounded** in the retrieved context (no hallucination)?
- Is it **complete** (not missing key information)?

Returns a confidence score (0.0–1.0).

### 2. Healing Strategies

When confidence < 0.8, the system tries these strategies (ordered by cost):

| # | Strategy | What It Does | Targets This Failure |
|---|----------|-------------|---------------------|
| 1 | **Query Expansion** | Adds synonyms/related terms | Vague answers from narrow queries |
| 2 | **Multi-Query** | 3 rephrased variants, merged results | Ambiguous questions |
| 3 | **Keyword Fallback** | Pure term matching (no semantics) | Exact terms missed (codes, names) |
| 4 | **Broader Retrieval** | top-K: 5 → 20 | Incomplete answers |
| 5 | **Chunk Refinement** | Re-chunks at 256 tokens | Noisy/mixed context |

### 3. Adaptive Learning

The system tracks which strategies succeed:

```
After 50+ queries:
  "query_expansion"  → 78% success rate, avg +0.3 confidence
  "keyword_fallback" → 85% success rate, avg +0.4 confidence  ← best!
  "multi_query"      → 60% success rate, avg +0.2 confidence
```

Next time, it **tries keyword_fallback first** instead of the default order.

### 4. Graceful Degradation

If all strategies fail (confidence still < 0.5):

```
⚠️ Low Confidence Answer

I found some information but I'm not fully confident:
[partial answer here]

What might be missing: [specific gaps identified]
Documents searched: [list]
Suggestions: Try uploading more relevant documents or rephrase your question.
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SELF-HEALING PIPELINE                   │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐    │
│  │ RETRIEVE │──►│ GENERATE │──►│    VALIDATE      │    │
│  │          │   │          │   │  (LLM-as-Judge)  │    │
│  └──────────┘   └──────────┘   └────────┬─────────┘    │
│       ▲                                  │              │
│       │                        ┌─────────▼──────────┐   │
│       │                        │  confidence ≥ 0.8? │   │
│       │                        └─────────┬──────────┘   │
│       │                          YES │        │ NO      │
│       │                              ▼        ▼         │
│       │                        ┌────────┐ ┌────────┐    │
│       │                        │ RETURN │ │  HEAL  │    │
│       │                        │   ✅   │ │(select │    │
│       │                        └────────┘ │strategy)│   │
│       │                                   └───┬────┘    │
│       └───────────── retry with ◄─────────────┘        │
│                     new results                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              ADAPTIVE LEARNER                     │   │
│  │  Tracks: strategy success rates, improvements    │   │
│  │  Adapts: reorders strategies by effectiveness    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## API Endpoints

### Self-Healing (new)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/healing/ask` | Ask with self-healing (validates + heals) |
| GET | `/healing/explain` | Last healing report |
| GET | `/healing/strategies` | Strategy performance stats |
| GET | `/healing/history` | Recent healing events |

### Standard RAG (unchanged)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/upload` | Upload documents |
| POST | `/query/ask` | Standard query (no healing) |
| GET | `/documents/` | List documents |
| GET | `/health` | System status |

## Example Response

```json
{
  "answer": "Q3 budget submissions are due September 15th [finance-procedures.pdf, page 8]",
  "confidence": 0.88,
  "healed": true,
  "attempts": 2,
  "strategy_used": "query_expansion",
  "sources": [...],
  "healing_report": {
    "original_confidence": 0.45,
    "final_confidence": 0.88,
    "attempts": [
      {"attempt": 1, "strategy": "none (initial)", "confidence": 0.45, "reason": "Answer discusses fiscal year but not specific deadline"},
      {"attempt": 2, "strategy": "query_expansion", "confidence": 0.88, "reason": "Answer directly states the Q3 deadline with source"}
    ]
  }
}
```

## Quick Start

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add Pinecone + Groq keys
uvicorn app.main:app --port 8000

# Try the self-healing endpoint:
curl -X POST http://localhost:8000/healing/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the deadline for Q3 submissions?"}'
```

## References

- [Corrective RAG (CRAG) — Microsoft Research, 2024](https://arxiv.org/abs/2401.15884)
- [Self-RAG: Learning to Retrieve, Generate, and Critique — UW, 2023](https://arxiv.org/abs/2310.11511)
- [RAGAS: Automated Evaluation of RAG](https://docs.ragas.io/)
- [Adaptive Retrieval-Augmented Generation — Meta, 2024](https://arxiv.org/abs/2403.14403)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| Embeddings | all-MiniLM-L6-v2 (local) |
| Vector DB | Pinecone |
| LLM | Groq + Llama 3.3 70B |
| Validation | LLM-as-Judge (second Groq call) |
| Learning | SQLite (strategy performance tracking) |
| Frontend | Next.js 14 + Tailwind |




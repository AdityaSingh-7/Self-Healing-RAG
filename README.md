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


 python scripts/ingest.py "C:/Users/Aditya Singh/Desktop/hyperfocus.pdf"
============================================================
INGESTING 1 FILE(S)
============================================================

📄 Processing: C:/Users/Aditya Singh/Desktop/hyperfocus.pdf
   Parsed: 232 pages
   Chunked: 316 chunks
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
modules.json: 100%|███████████████████████████████████████████████████████████████████████████████████████| 349/349 [00:00<?, ?B/s]
C:\Users\Aditya Singh\Desktop\Self-Healing-RAG\venv\Lib\site-packages\huggingface_hub\file_download.py:139: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\Aditya Singh\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
config_sentence_transformers.json: 100%|██████████████████████████████████████████████████████████| 116/116 [00:00<00:00, 60.3kB/s]
README.md: 100%|██████████████████████████████████████████████████████████████████████████████████████| 10.5k/10.5k [00:00<?, ?B/s]
sentence_bert_config.json: 100%|████████████████████████████████████████████████████████████████████████| 53.0/53.0 [00:00<?, ?B/s]
config.json: 100%|████████████████████████████████████████████████████████████████████████████████| 612/612 [00:00<00:00, 1.19MB/s]
model.safetensors: 100%|██████████████████████████████████████████████████████████████████████| 90.9M/90.9M [00:06<00:00, 13.0MB/s]
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 4558.16it/s]
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████████████████| 350/350 [00:00<?, ?B/s]
vocab.txt: 100%|████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 15.2MB/s]
tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 39.4MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████| 112/112 [00:00<?, ?B/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<?, ?B/s]
Embedding model loaded!
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:08<00:00,  1.23it/s]
   Embedded: 316 vectors
Creating Pinecone index: rag-system
Index created!
   Stored: 316 vectors in Pinecone
   ⏱️  Time: 49.3s

============================================================
✅ DONE — 1 files ingested
   Total chunks in Pinecone: 316
============================================================

Next step: python scripts/benchmark.py
(venv) PS C:\Users\Aditya Singh\Desktop\Self-Healing-RAG\backend> python scripts\benchmark.py
======================================================================
BENCHMARK: Standard RAG vs Self-Healing RAG
======================================================================
Questions: 20
User namespace: benchmark_user

📚 Documents in index: 316 vectors

----------------------------------------------------------------------

[1/20] What is hyperfocus?
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6904.26it/s]
Embedding model loaded!
   Standard: confidence=1.00, tokens=1250, latency=11384ms
   Healing:  confidence=0.90, tokens=1250, latency=4852ms, attempts=1

[2/20] What is scatterfocus?
   Standard: confidence=1.00, tokens=1250, latency=5147ms
   Healing:  confidence=1.00, tokens=1250, latency=4420ms, attempts=1

[3/20] Why is attention considered a limited resource?
   Standard: confidence=0.80, tokens=1250, latency=5336ms
   Healing:  confidence=0.80, tokens=1250, latency=5650ms, attempts=1

[4/20] What are the two primary modes of attention described in the book?
   Standard: confidence=0.60, tokens=1250, latency=4970ms
   Healing:  confidence=0.60, tokens=4350, latency=35360ms, attempts=3

[5/20] Why does multitasking reduce productivity?
   Standard: confidence=0.80, tokens=1250, latency=13650ms
   Healing:  confidence=0.80, tokens=1250, latency=12234ms, attempts=1

[6/20] How does boredom improve creativity?
   Standard: confidence=0.80, tokens=1250, latency=17907ms
   Healing:  confidence=0.80, tokens=1250, latency=19306ms, attempts=1

[7/20] What role do distractions play in attention?
   Standard: confidence=0.80, tokens=1250, latency=17761ms
   Healing:  confidence=0.80, tokens=2800, latency=37291ms, attempts=2 → HEALED by query_expansion

[8/20] How can someone prepare for a deep focus session?
   Standard: confidence=0.80, tokens=1250, latency=14861ms
   Healing:  confidence=0.80, tokens=1250, latency=14894ms, attempts=1

[9/20] What is attentional space?
   Standard: confidence=1.00, tokens=1250, latency=13083ms
   Healing:  confidence=1.00, tokens=1250, latency=12563ms, attempts=1

[10/20] Why are smartphones harmful to concentration?
   Standard: confidence=0.80, tokens=1250, latency=15744ms
   Healing:  confidence=0.80, tokens=1250, latency=13322ms, attempts=1

[11/20] What is the relationship between dopamine and distraction?
   Standard: confidence=0.60, tokens=1250, latency=19679ms
   Healing:  confidence=0.60, tokens=4350, latency=61890ms, attempts=3

[12/20] Why should difficult work be done during periods of high energy?
   Standard: confidence=0.80, tokens=1250, latency=19566ms
   Healing:  confidence=0.80, tokens=1250, latency=19010ms, attempts=1

[13/20] How long does the author recommend spending in a hyperfocus session?
   Standard: confidence=0.80, tokens=1250, latency=13773ms
   Healing:  confidence=0.70, tokens=4350, latency=39824ms, attempts=3

[14/20] What is the purpose of scatterfocus?
Traceback (most recent call last):
  File "C:\Users\Aditya Singh\Desktop\Self-Healing-RAG\backend\scripts\benchmark.py", line 331, in <module>
    asyncio.run(main())
  File "C:\Users\Aditya Singh\AppData\Local\Programs\Python\Python311\Lib\asyncio\runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "C:\Users\Aditya Singh\AppData\Local\Programs\Python\Python311\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Aditya Singh\AppData\Local\Programs\Python\Python311\Lib\asyncio\base_events.py", line 654, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "C:\Users\Aditya Singh\Desktop\Self-Healing-RAG\backend\scripts\benchmark.py", line 212, in main
    std = await run_standard_pipeline(q, user_id)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Aditya Singh\Desktop\Self-Healing-RAG\backend\scripts\benchmark.py", line 136, in run_standard_pipeline
    answer = await llm.generate(question, results)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Aditya Singh\Desktop\Self-Healing-RAG\backend\app\services\llm.py", line 175, in generate
    response = await self.client.chat.completions.create(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Aditya Singh\Desktop\Self-Healing-RAG\venv\Lib\site-packages\groq\resources\chat\completions.py", line 943, in create       
    return await self._post(
           ^^^^^^^^^^^^^^^^^
  File "C:\Users\Aditya Singh\Desktop\Self-Healing-RAG\venv\Lib\site-packages\groq\_base_client.py", line 1856, in post
    return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Aditya Singh\Desktop\Self-Healing-RAG\venv\Lib\site-packages\groq\_base_client.py", line 1655, in request
    raise self._make_status_error_from_response(err.response) from None
groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kx1gs5cffsfa25x6693msmkh` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99769, Requested 1376. Please try again in 16m29.28s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}

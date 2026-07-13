"""
pipeline/contrastive.py — Contrastive Retrieval

THE NOVEL TECHNIQUE:
Generate a NEGATIVE query (what the question is NOT about) and use it to
filter false positives from the retrieval results.

WHY THIS IS POWERFUL:
Regular search for "What is the 40-second rule?" returns:
  ✓ Chunk about the 40-second rule
  ✗ Chunk about general productivity tips (semantically similar but wrong)
  ✗ Chunk about other numbered rules

Contrastive retrieval:
  Positive query: "40-second rule focus attention"
  Negative query: "general productivity advice not about specific seconds or timing rules"

  Score: positive_similarity - λ * negative_similarity

  The "general tips" chunk scores HIGH on the negative query → gets penalized.
  The "40-second rule" chunk scores LOW on the negative → stays ranked high.

BASED ON: Contrastive learning principles (SimCLR, CLIP) applied to retrieval.
Few production systems do this — genuinely novel implementation.
"""

import numpy as np
from groq import AsyncGroq

from app.config import settings
from app.services.embedder import EmbeddingService


NEGATIVE_QUERY_PROMPT = """Given this search question, generate a NEGATIVE query that describes
what the user is NOT looking for — the type of content that would be a false positive.

RULES:
- The negative should describe content that is semantically similar but NOT the answer
- Keep it short (under 20 words)
- Return ONLY the negative query, nothing else

EXAMPLES:
Question: "What is the 40-second rule?"
Negative: "general productivity tips and time management advice not about specific second-based rules"

Question: "How does dopamine affect focus?"
Negative: "general brain chemistry and neuroscience unrelated to attention or concentration"

Question: "What did the author say about multitasking?"
Negative: "general opinions about work habits not specific to the author or this book"
"""


async def generate_negative_query(question: str) -> str:
    """
    Generate a negative query — what the user is NOT looking for.
    Used to penalize false-positive retrievals.
    """
    client = AsyncGroq(api_key=settings.groq_api_key)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": NEGATIVE_QUERY_PROMPT},
            {"role": "user", "content": f"Question: {question}\nNegative:"},
        ],
        temperature=0.3,
        max_tokens=50,
    )

    return response.choices[0].message.content.strip()


def contrastive_rerank(
    results: list[dict],
    query_embedding: list[float],
    negative_embedding: list[float],
    lambda_penalty: float = 0.3,
) -> list[dict]:
    """
    Rerank results using contrastive scoring.

    score(chunk) = similarity(query, chunk) - λ * similarity(negative, chunk)

    Chunks similar to the NEGATIVE query get penalized.
    This removes false positives that are topically adjacent but not actually relevant.

    Parameters:
    -----------
    results : list[dict]
        Initial retrieval results
    query_embedding : list[float]
        Embedding of the original question
    negative_embedding : list[float]
        Embedding of the negative query (what we DON'T want)
    lambda_penalty : float
        How much to penalize negative-similar results (0.0 = no effect, 1.0 = heavy penalty)

    Returns:
    --------
    list[dict] — reranked results with contrastive_score added
    """
    if not results:
        return results

    embedder = EmbeddingService()
    query_vec = np.array(query_embedding)
    neg_vec = np.array(negative_embedding)

    for result in results:
        # Get chunk embedding (re-embed from text)
        chunk_vec = np.array(embedder.embed_text(result["text"]))

        # Cosine similarities
        pos_sim = float(np.dot(query_vec, chunk_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8
        ))
        neg_sim = float(np.dot(neg_vec, chunk_vec) / (
            np.linalg.norm(neg_vec) * np.linalg.norm(chunk_vec) + 1e-8
        ))

        # Contrastive score: positive minus penalty for negative similarity
        contrastive_score = pos_sim - (lambda_penalty * neg_sim)

        result["positive_similarity"] = round(pos_sim, 4)
        result["negative_similarity"] = round(neg_sim, 4)
        result["contrastive_score"] = round(contrastive_score, 4)

    # Sort by contrastive score (highest first)
    results.sort(key=lambda x: x["contrastive_score"], reverse=True)

    return results


async def contrastive_retrieval(
    question: str,
    results: list[dict],
    query_embedding: list[float],
    lambda_penalty: float = 0.3,
) -> dict:
    """
    Full contrastive retrieval pipeline:
    1. Generate negative query
    2. Embed it
    3. Rerank results contrastively

    Returns:
    --------
    dict with:
        - results: reranked list
        - negative_query: what was generated
        - metadata: scoring details
    """
    # Generate negative query
    negative_query = await generate_negative_query(question)

    # Embed negative query
    embedder = EmbeddingService()
    negative_embedding = embedder.embed_text(negative_query)

    # Rerank with contrastive scoring
    reranked = contrastive_rerank(
        results=results,
        query_embedding=query_embedding,
        negative_embedding=negative_embedding,
        lambda_penalty=lambda_penalty,
    )

    return {
        "results": reranked,
        "negative_query": negative_query,
        "metadata": {
            "lambda_penalty": lambda_penalty,
            "avg_negative_similarity": round(
                sum(r["negative_similarity"] for r in reranked) / max(len(reranked), 1), 4
            ),
        },
    }

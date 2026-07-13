"""
pipeline/hyde.py — Hypothetical Document Embedding (HyDE)

THE IDEA:
Instead of embedding the SHORT QUESTION and searching, we:
1. Ask the LLM to generate a HYPOTHETICAL answer (without retrieval)
2. Embed that hypothetical answer
3. Search with the hypothetical embedding

WHY THIS WORKS:
A question: "What is hyperfocus?" → embedding captures "question about hyperfocus"
A hypothetical: "Hyperfocus is a state of deep concentration where..." → embedding
  captures "explanation of hyperfocus"

The hypothetical LOOKS LIKE a document chunk. So it matches against real chunks
much better than a short question does.

PROVEN: +10-20% retrieval recall in academic benchmarks.
REFERENCE: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al, 2022)
"""

from groq import AsyncGroq
from app.config import settings
from app.services.embedder import EmbeddingService


HYDE_PROMPT = """Given this question, write a short paragraph (3-4 sentences) that would
be the IDEAL answer found in a document. Write as if you're quoting a textbook or manual.

RULES:
- Write in a factual, document-like style
- Don't hedge or say "I think" — write as if it's from a reference document
- Keep it 3-4 sentences max
- It's okay to be approximate — the goal is to SOUND like the right document passage

Return ONLY the hypothetical passage, nothing else."""


async def generate_hypothetical_document(question: str) -> str:
    """
    Generate a hypothetical answer passage for HyDE.

    The LLM makes up a plausible answer. We don't use this answer
    directly — we EMBED it and use the embedding to search.
    """
    client = AsyncGroq(api_key=settings.groq_api_key)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.5,  # Slightly creative for diverse hypotheticals
        max_tokens=150,
    )

    return response.choices[0].message.content.strip()


async def hyde_embed(question: str) -> dict:
    """
    Full HyDE pipeline:
    1. Generate hypothetical document
    2. Embed it
    3. Return the embedding (for use in search)

    Returns:
    --------
    dict with:
        - embedding: list[float] (384-dim vector of the hypothetical)
        - hypothetical: str (the generated text, for debugging)
        - original_embedding: list[float] (regular question embedding for comparison)
    """
    # Generate hypothetical answer
    hypothetical = await generate_hypothetical_document(question)

    # Embed both the hypothetical and the original question
    embedder = EmbeddingService()
    hyde_embedding = embedder.embed_text(hypothetical)
    original_embedding = embedder.embed_text(question)

    return {
        "embedding": hyde_embedding,
        "hypothetical": hypothetical,
        "original_embedding": original_embedding,
    }


async def hyde_search(question: str, user_id: str, top_k: int = 10) -> dict:
    """
    Search using HyDE embedding instead of direct question embedding.

    Returns more results (top_k=10) for downstream reranking.
    """
    from app.services.vectorstore import VectorStoreService

    # Generate HyDE embedding
    hyde_result = await hyde_embed(question)

    # Search with hypothetical embedding
    store = VectorStoreService()
    results = store.search(
        query_embedding=hyde_result["embedding"],
        user_id=user_id,
        top_k=top_k,
    )

    return {
        "results": results,
        "hypothetical": hyde_result["hypothetical"],
        "metadata": {
            "method": "HyDE",
            "hypothetical_preview": hyde_result["hypothetical"][:100],
        },
    }

"""
strategies/chunk_refinement.py — Strategy 5: Chunk Refinement

WHAT: Re-chunks the top retrieved documents at a smaller size (256 tokens)
      for more granular matching.
WHEN: Retrieved chunks are too broad — they contain the answer + noise.
COST: Re-chunking + re-embedding + re-search (most expensive strategy)

EXAMPLE:
  Original 512-token chunk: "PTO policy is 20 days. <blah blah 400 tokens> Carry-over is limited to 5 days."
  After 256-token re-chunk:
    Chunk A: "PTO policy is 20 days..."
    Chunk B: "Carry-over is limited to 5 days..."
  Now each chunk is focused on ONE topic — better precision.
"""

from llama_index.core.node_parser import SentenceSplitter

from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.pipeline.strategies.base import HealingStrategy
from app.config import settings


class ChunkRefinementStrategy(HealingStrategy):
    """Re-chunks retrieved documents at smaller granularity."""

    @property
    def name(self) -> str:
        return "chunk_refinement"

    @property
    def description(self) -> str:
        return "Re-chunk retrieved docs at 256 tokens for finer-grained matching"

    async def execute(self, question: str, original_results: list[dict], user_id: str, validation_issues: str) -> dict:
        if not original_results:
            return {"results": [], "modified_question": question, "metadata": {"note": "No results to refine"}}

        # Step 1: Get the text from original results
        original_texts = [r["text"] for r in original_results]

        # Step 2: Re-chunk at smaller size
        splitter = SentenceSplitter(chunk_size=256, chunk_overlap=30)
        refined_chunks = []

        for i, text in enumerate(original_texts):
            sub_chunks = splitter.split_text(text)
            for sub_chunk in sub_chunks:
                refined_chunks.append({
                    "text": sub_chunk,
                    "metadata": original_results[i].get("metadata", {}),
                })

        if not refined_chunks:
            return {"results": original_results, "modified_question": question, "metadata": {"note": "Re-chunking produced no results"}}

        # Step 3: Re-embed the refined chunks
        embedder = EmbeddingService()
        chunk_texts = [c["text"] for c in refined_chunks]
        chunk_embeddings = embedder.embed_batch(chunk_texts)

        # Step 4: Embed the question and find best matches among refined chunks
        query_embedding = embedder.embed_text(question)

        # Manual similarity calculation (these aren't in Pinecone)
        import numpy as np
        query_vec = np.array(query_embedding)

        scored_results = []
        for chunk, embedding in zip(refined_chunks, chunk_embeddings):
            emb_vec = np.array(embedding)
            similarity = float(np.dot(query_vec, emb_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)))

            scored_results.append({
                "text": chunk["text"],
                "score": similarity,
                "recency_score": 1.0,  # Not applicable for re-ranked
                "final_score": similarity,
                "metadata": chunk["metadata"],
            })

        # Sort by similarity and take top results
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = scored_results[:settings.top_k]

        return {
            "results": top_results,
            "modified_question": question,
            "metadata": {
                "original_chunks": len(original_texts),
                "refined_chunks": len(refined_chunks),
                "chunk_size": 256,
            },
        }

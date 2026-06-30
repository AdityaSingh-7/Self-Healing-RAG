"""
strategies/broader_retrieval.py — Strategy 4: Broader Retrieval

WHAT: Increases top-K from 5 → 20 to get more context.
WHEN: The answer was incomplete — there's relevant info but it wasn't in the top 5.
COST: 1 search (cheap, just asks for more results)

EXAMPLE:
  Original: top_k=5 → gets chunks about PTO but not carry-over rules
  Broader: top_k=20 → finds the carry-over rule in position 12
"""

from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.pipeline.strategies.base import HealingStrategy
from app.config import settings


class BroaderRetrievalStrategy(HealingStrategy):
    """Retrieves more results by increasing top-K."""

    @property
    def name(self) -> str:
        return "broader_retrieval"

    @property
    def description(self) -> str:
        return "Increase retrieval scope (top-K: 5 → 20) for more context"

    async def execute(self, question: str, original_results: list[dict], user_id: str, validation_issues: str) -> dict:
        # Re-search with much higher top-K
        broader_top_k = 20

        embedder = EmbeddingService()
        query_embedding = embedder.embed_text(question)

        store = VectorStoreService()
        results = store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=broader_top_k,
            recency_weight=settings.recency_weight,
        )

        return {
            "results": results,
            "modified_question": question,
            "metadata": {
                "original_top_k": settings.top_k,
                "broader_top_k": broader_top_k,
                "results_found": len(results),
            },
        }

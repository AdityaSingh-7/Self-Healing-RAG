"""
strategies/multi_query.py — Strategy 2: Multi-Query Retrieval

WHAT: Generates 3 different phrasings of the question, searches with each,
      then merges and deduplicates results.
WHEN: The question is ambiguous — different phrasings might find different relevant chunks.
COST: 1 LLM call + 3 searches (medium)

EXAMPLE:
  Original: "How does the system handle errors?"
  Variant 1: "What is the error handling mechanism?"
  Variant 2: "How are exceptions and failures managed?"
  Variant 3: "What happens when something goes wrong in the system?"
"""

from groq import AsyncGroq
from app.config import settings
from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.pipeline.strategies.base import HealingStrategy


MULTI_QUERY_PROMPT = """Generate 3 different phrasings of this question.
Each phrasing should approach the topic from a different angle.
Return ONLY the 3 questions, one per line, numbered 1-3. No explanations."""


class MultiQueryStrategy(HealingStrategy):
    """Generates multiple query variants and merges results."""

    @property
    def name(self) -> str:
        return "multi_query"

    @property
    def description(self) -> str:
        return "Generate 3 query variants, search with each, merge and deduplicate results"

    async def execute(self, question: str, original_results: list[dict], user_id: str, validation_issues: str) -> dict:
        # Step 1: Generate 3 variants
        client = AsyncGroq(api_key=settings.groq_api_key)

        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": MULTI_QUERY_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.7,  # Higher temp = more diverse phrasings
            max_tokens=200,
        )

        # Parse the 3 variants
        lines = response.choices[0].message.content.strip().split("\n")
        variants = []
        for line in lines:
            # Remove numbering (1. 2. 3. or 1) 2) 3))
            cleaned = line.strip().lstrip("0123456789.)- ").strip()
            if cleaned:
                variants.append(cleaned)

        # Ensure we have at least the original
        if not variants:
            variants = [question]
        variants = variants[:3]  # Max 3

        # Step 2: Search with each variant
        embedder = EmbeddingService()
        store = VectorStoreService()
        all_results = []
        seen_ids = set()

        for variant in variants:
            query_embedding = embedder.embed_text(variant)
            results = store.search(
                query_embedding=query_embedding,
                user_id=user_id,
                top_k=5,
            )

            # Deduplicate by text content
            for r in results:
                text_key = r["text"][:100]  # Use first 100 chars as key
                if text_key not in seen_ids:
                    seen_ids.add(text_key)
                    all_results.append(r)

        # Sort by final_score and take top results
        all_results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = all_results[:settings.top_k]

        return {
            "results": top_results,
            "modified_question": question,  # Keep original for LLM
            "metadata": {
                "variants_generated": variants,
                "total_unique_results": len(all_results),
                "results_returned": len(top_results),
            },
        }

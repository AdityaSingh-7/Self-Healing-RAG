"""
pipeline/fusion.py — Reciprocal Rank Fusion (RRF)

WHAT: Mathematically principled way to combine results from multiple search strategies.

THE FORMULA:
  RRF_score(document) = Σ 1 / (k + rank_in_list_i)

  Where k = 60 (standard constant), and we sum across all lists.

WHY NOT JUST AVERAGE SCORES:
- Different strategies use different scoring scales (cosine 0-1, BM25 0-25, cross-encoder -10 to +10)
- You can't average these — the scales don't match
- RRF uses only RANKS (positions), which are scale-free
- If a document ranks #1 in two different strategies → very likely relevant

EXAMPLE:
  Semantic search: [A=rank1, B=rank2, C=rank3, D=rank4]
  HyDE search:     [B=rank1, A=rank2, D=rank3, E=rank4]
  Keyword search:  [A=rank1, C=rank2, E=rank3, F=rank4]

  RRF(A) = 1/(60+1) + 1/(60+2) + 1/(60+1) = 0.0164 + 0.0161 + 0.0164 = 0.0489
  RRF(B) = 1/(60+2) + 1/(60+1) + 0         = 0.0161 + 0.0164 + 0      = 0.0325
  RRF(C) = 1/(60+3) + 0         + 1/(60+2) = 0.0159 + 0       + 0.0161 = 0.0320

  Final ranking: A > B > C (A appeared in all 3 lists, strong consensus)

REFERENCE: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
           (Cormack, Clarke & Büttcher, 2009)
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 5,
    id_key: str = "text",
) -> list[dict]:
    """
    Combine multiple ranked result lists using RRF.

    Parameters:
    -----------
    ranked_lists : list[list[dict]]
        Multiple result lists, each already sorted by relevance.
        Example: [semantic_results, hyde_results, keyword_results]
    k : int
        Constant (standard = 60). Higher k = less weight to top positions.
    top_k : int
        How many final results to return.
    id_key : str
        Which field to use as document identity for merging.
        Default "text" — uses first 100 chars of text as ID.

    Returns:
    --------
    list[dict] — merged and reranked results with rrf_score added.
    """
    # Track RRF scores for each unique document
    rrf_scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}  # Keep the full result dict for each doc

    for result_list in ranked_lists:
        for rank, result in enumerate(result_list):
            # Create a document ID (first 100 chars of text)
            doc_id = result.get(id_key, "")[:100]

            if not doc_id:
                continue

            # RRF formula: 1 / (k + rank)
            # rank is 0-indexed, so rank+1 for 1-indexed positions
            rrf_score = 1.0 / (k + rank + 1)

            # Accumulate across lists
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

            # Keep the version with the best original score
            if doc_id not in doc_data or result.get("final_score", 0) > doc_data[doc_id].get("final_score", 0):
                doc_data[doc_id] = result

    # Sort by RRF score (highest first)
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final results
    final_results = []
    for doc_id, rrf_score in sorted_docs[:top_k]:
        result = doc_data[doc_id].copy()
        result["rrf_score"] = round(rrf_score, 6)
        result["appeared_in_lists"] = sum(
            1 for result_list in ranked_lists
            for r in result_list
            if r.get(id_key, "")[:100] == doc_id
        )
        final_results.append(result)

    return final_results


def weighted_rrf(
    ranked_lists: list[list[dict]],
    weights: list[float],
    k: int = 60,
    top_k: int = 5,
    id_key: str = "text",
) -> list[dict]:
    """
    Weighted RRF — give different strategies different importance.

    Parameters:
    -----------
    weights : list[float]
        Weight for each ranked list. Example: [1.0, 0.8, 0.5]
        means first list is most trusted, third list least.
    """
    if len(weights) != len(ranked_lists):
        weights = [1.0] * len(ranked_lists)

    rrf_scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}

    for list_idx, result_list in enumerate(ranked_lists):
        weight = weights[list_idx]

        for rank, result in enumerate(result_list):
            doc_id = result.get(id_key, "")[:100]
            if not doc_id:
                continue

            # Weighted RRF
            rrf_score = weight * (1.0 / (k + rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

            if doc_id not in doc_data or result.get("final_score", 0) > doc_data[doc_id].get("final_score", 0):
                doc_data[doc_id] = result

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for doc_id, rrf_score in sorted_docs[:top_k]:
        result = doc_data[doc_id].copy()
        result["rrf_score"] = round(rrf_score, 6)
        final_results.append(result)

    return final_results

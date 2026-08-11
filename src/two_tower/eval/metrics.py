"""Ranking metrics. The implementations are yours — this file is the contract.

All three functions take the same two arguments:
  relevant: the ground-truth set of items the user interacted with in the
            eval split
  ranked:   the recommender's ranking for that user, most-liked first,
            limited to the item catalog
"""

from __future__ import annotations

import math


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:

    """Fraction of relevant items that appear in the top-k of `ranked`."""

    hits = sum(1 for item in ranked[:k] if item in relevant)

    if len(relevant) == 0:
        return 0.0
    else:
        return hits/len(relevant)


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Discounted cumulative gain: rewards relevant items ranked earlier."""

    dcg = 0

    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            dcg += 1 / math.log2(i+2)

    ideal = sum(1 / math.log2(pos + 1) for pos in range(1, min(k, len(relevant)) + 1))

    if k <= 0 or len(relevant) == 0:
        return 0.0
    return dcg/ideal


def mrr_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Reciprocal rank of the first relevant item within the top-k; 0 if none"""

    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            return 1/(i+1)

    return 0.0

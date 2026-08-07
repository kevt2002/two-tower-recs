"""Ranking metrics. The implementations are yours — this file is the contract.

All three functions take the same two arguments:
  relevant: the ground-truth set of items the user interacted with in the
            eval split
  ranked:   the recommender's ranking for that user, most-liked first,
            limited to the item catalog
"""

from __future__ import annotations


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k of `ranked`.

    Edge cases to handle: k > len(ranked), and empty `relevant` (return 0.0?
    skip the user entirely? Decide and document it — it changes the mean).
    """
    raise NotImplementedError


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Discounted cumulative gain: rewards relevant items ranked earlier.

    Standard definition: sum of 1 / log2(position + 1) over relevant items
    within the top-k, normalized by the ideal DCG.

    Decide: what is the ideal ranking when |relevant| > k, or when a relevant
    item sits below position k?
    """
    raise NotImplementedError


def mrr_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Reciprocal rank of the first relevant item within the top-k; 0 if none."""
    raise NotImplementedError

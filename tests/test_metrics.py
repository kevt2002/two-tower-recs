"""Property tests for the ranking metrics.

Contract (from src/two_tower/eval/metrics.py):
  - all three take (relevant: set of ground-truth items, ranked: list of
    predicted items, k: retrieval budget) and return a float in [0, 1]
  - empty `relevant` -> 0.0 for all three (caller skips those users in the mean)
  - perfect ranking (all relevant items first) -> recall 1.0, ndcg 1.0
"""

import random

import pytest

from two_tower.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def make_ranking(n_items: int = 30, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    items = [f"i{i}" for i in range(n_items)]
    rng.shuffle(items)
    return items


def test_everything_in_unit_interval() -> None:
    rng = random.Random(0)
    for k in [0, 1, 5, 50]:
        for _ in range(20):
            ranked = make_ranking(seed=rng.randint(0, 1000))
            relevant = set(rng.sample(ranked, rng.randint(0, 5)))
            for metric in (recall_at_k, ndcg_at_k, mrr_at_k):
                score = metric(relevant, ranked, k)
                assert 0.0 <= score <= 1.0


def test_empty_relevant_is_zero() -> None:
    ranked = make_ranking()
    for k in [0, 1, 10, 100]:
        assert recall_at_k(set(), ranked, k) == 0.0
        assert ndcg_at_k(set(), ranked, k) == 0.0
        assert mrr_at_k(set(), ranked, k) == 0.0


def test_perfect_ranking_scores_one() -> None:
    relevant = {"a", "b", "c"}
    ranked = ["a", "b", "c", "x", "y", "z"]
    assert recall_at_k(relevant, ranked, k=3) == 1.0
    assert ndcg_at_k(relevant, ranked, k=3) == 1.0
    assert mrr_at_k(relevant, ranked, k=3) == 1.0


def test_recall_full_catalog_is_one() -> None:
    ranked = make_ranking()
    relevant = set(random.Random(1).sample(ranked, 10))
    assert recall_at_k(relevant, ranked, k=len(ranked)) == 1.0


def test_recall_non_decreasing_in_k() -> None:
    rng = random.Random(0)
    ranked = make_ranking(seed=2)
    relevant = set(rng.sample(ranked, 8))
    recalls = [recall_at_k(relevant, ranked, k) for k in range(0, len(ranked) + 1, 5)]
    assert recalls == sorted(recalls)


def test_recall_known_answers() -> None:
    relevant = {"a", "b"}
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(relevant, ranked, k=1) == 0.0
    assert recall_at_k(relevant, ranked, k=2) == 0.5
    assert recall_at_k(relevant, ranked, k=3) == 0.5
    assert recall_at_k(relevant, ranked, k=4) == 1.0


def test_ndcg_known_answers() -> None:
    import math

    relevant = {"a", "b"}
    ranked = ["x", "a", "y", "b"]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(relevant, ranked, k=4) == pytest.approx(dcg / ideal)


def test_ndcg_penalizes_late_hits() -> None:
    relevant = {"a", "b"}
    early = ndcg_at_k(relevant, ["a", "b", "x", "y"], k=4)
    late = ndcg_at_k(relevant, ["a", "x", "y", "b"], k=4)
    assert early == 1.0
    assert late < early


def test_ndcg_ideal_capped_at_k() -> None:
    import math

    relevant = {"a", "b", "c", "d", "e"}
    ranked = ["a", "b", "x", "y"]
    assert ndcg_at_k(relevant, ranked, k=1) == pytest.approx(1.0)
    assert ndcg_at_k(relevant, ranked, k=2) == pytest.approx(1.0)
    dcg = 1 + 1 / math.log2(3)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(relevant, ranked, k=3) == pytest.approx(dcg / ideal)


def test_mrr_known_answers() -> None:
    relevant = {"a", "b"}
    ranked = ["x", "a", "b", "y"]
    assert mrr_at_k(relevant, ranked, k=1) == 0.0
    assert mrr_at_k(relevant, ranked, k=2) == 0.5
    assert mrr_at_k(relevant, ranked, k=4) == 0.5
    assert mrr_at_k({"x"}, ranked, k=4) == 1.0


def test_mrr_only_first_hit_counts() -> None:
    relevant = {"a", "b", "c"}
    ranked = ["x", "a", "b", "c"]
    assert mrr_at_k(relevant, ranked, k=4) == pytest.approx(0.5)

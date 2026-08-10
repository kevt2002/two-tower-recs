"""Property tests for the popularity baseline.

Contract (from src/two_tower/baselines/popularity.py):
  - popularity = interaction count per item, computed from the TRAIN frame only
  - recommend(user_id, k) returns the k most popular item ids, highest first,
    ignoring user_id entirely
  - deterministic: ties broken by ascending item id
"""

import numpy as np
import pandas as pd
import pytest

from two_tower.baselines.popularity import PopularityRecommender


def make_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_asin": ["B", "A", "B", "C", "A", "A", "D"],
            "rating": [5, 1, 5, 3, 2, 4, 5],
        }
    )


def test_known_answer_ordering() -> None:
    rec = PopularityRecommender()
    rec.fit(make_train())
    assert rec.recommend(k=10) == ["A", "B", "C", "D"]


def test_top_k_respected() -> None:
    rec = PopularityRecommender()
    rec.fit(make_train())
    assert rec.recommend(k=2) == ["A", "B"]
    assert rec.recommend(k=1) == ["A"]


def test_k_larger_than_catalog() -> None:
    rec = PopularityRecommender()
    rec.fit(make_train())
    assert len(rec.recommend(k=100)) == 4


def test_k_zero_and_negative() -> None:
    rec = PopularityRecommender()
    rec.fit(make_train())
    assert rec.recommend(k=0) == []
    assert rec.recommend(k=-1) == []


def test_user_agnostic() -> None:
    rec = PopularityRecommender()
    rec.fit(make_train())
    assert rec.recommend(user_id="u1") == rec.recommend(user_id="u2") == rec.recommend()


def test_deterministic_across_calls() -> None:
    rec = PopularityRecommender()
    rec.fit(make_train())
    assert rec.recommend(k=3) == rec.recommend(k=3)


def test_returns_strings_only() -> None:
    rec = PopularityRecommender()
    rec.fit(make_train())
    ranking = rec.recommend(k=100)
    assert all(isinstance(item, str) for item in ranking)
    assert len(ranking) == len(set(ranking))


def test_no_test_leakage() -> None:
    train = pd.DataFrame({"parent_asin": ["A", "A", "B"]})
    test = pd.DataFrame({"parent_asin": ["C", "C", "C", "D"]})
    rec = PopularityRecommender()
    rec.fit(train)
    test_only = set(test["parent_asin"]) - set(train["parent_asin"])
    ranking = set(rec.recommend(k=100))
    assert ranking.isdisjoint(test_only)


def test_ranking_matches_counts_on_random_data() -> None:
    rng = np.random.default_rng(0)
    items = np.array([f"i{i}" for i in range(50)])
    rows = rng.choice(items, size=500, replace=True)
    train = pd.DataFrame({"parent_asin": rows})
    counts = train["parent_asin"].value_counts()
    rec = PopularityRecommender()
    rec.fit(train)
    ranking = rec.recommend(k=50)
    for first, second in zip(ranking[:-1], ranking[1:], strict=True):
        assert counts[first] >= counts[second]


def test_ties_broken_by_ascending_id() -> None:
    train = pd.DataFrame({"parent_asin": ["z", "a", "m", "z", "a"]})
    rec = PopularityRecommender()
    rec.fit(train)
    ranking = rec.recommend(k=3)
    assert ranking == ["a", "z", "m"]


def test_recommend_before_fit_raises() -> None:
    rec = PopularityRecommender()
    with pytest.raises(TypeError):
        rec.recommend()


def test_fit_does_not_mutate_input() -> None:
    train = make_train()
    before = train.copy(deep=True)
    rec = PopularityRecommender()
    rec.fit(train)
    pd.testing.assert_frame_equal(train, before)

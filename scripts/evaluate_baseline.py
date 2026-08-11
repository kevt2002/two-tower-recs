"""Evaluate the popularity baseline on the test split.

Produces the numbers the two-tower model has to beat: mean Recall@K,
NDCG@K and MRR@K over test users, for a few values of K.

Run:  python scripts/evaluate_baseline.py
"""

from __future__ import annotations

import pandas as pd

from two_tower.baselines.popularity import PopularityRecommender
from two_tower.data.load import load_interactions
from two_tower.data.split import temporal_split
from two_tower.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k

K_VALUES = [1, 5, 10, 20, 50]
MAX_K = max(K_VALUES)


def relevant_per_user(test: pd.DataFrame) -> dict[str, set[str]]:

    return test.groupby("user_id")["parent_asin"].apply(set).to_dict()


def evaluate_popularity(
    rec: PopularityRecommender, relevant: dict[str, set[str]]
) -> dict[str, float]:

    totals = {}
    counts = {}

    for user_id, relevant_set in relevant.items():
        if len(relevant_set) == 0:
            continue
        ranked = rec.recommend(user_id=user_id, k=MAX_K)
        for k in K_VALUES:
            for metric, fn in (("recall", recall_at_k), ("mrr", mrr_at_k), ("ndcg", ndcg_at_k)):
                key = metric_name(metric=metric, k=k)
                totals[key] = totals.get(key, 0.0) + fn(relevant_set, ranked, k)
                counts[key] = counts.get(key, 0) + 1

    return {name: totals[name] / counts[name] for name in totals}


def metric_name(metric: str, k: int) -> str:
    """e.g. ("recall", 10) -> "recall@10"."""
    return f"{metric}@{k}"


def print_results(results: dict[str, float]) -> None:

    print("Popularity baseline | test split | mean per user")
    print("-" * 40)

    for k,v in results.items():
        print(f"{k} {v:.4f}")


def main() -> None:
    df = load_interactions()
    train, val, test = temporal_split(df)

    rec = PopularityRecommender()
    rec.fit(train)

    relevant = relevant_per_user(test)
    results = evaluate_popularity(rec, relevant)
    print_results(results)


if __name__ == "__main__":
    main()

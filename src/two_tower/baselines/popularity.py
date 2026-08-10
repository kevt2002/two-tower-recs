"""Popularity baseline: every user gets the same top-K by train popularity."""

from __future__ import annotations

import pandas as pd


class PopularityRecommender:

    def __init__(self):
        self._ranking = None

    def fit(self, train: pd.DataFrame) -> None:

        unique_count = train["parent_asin"].value_counts()
        unique_count = unique_count.to_frame("count")
        unique_count = unique_count.sort_values(
            by=["count", "parent_asin"], ascending=[False, True]
        )
        self._ranking = unique_count.index.tolist()


    def recommend(self, user_id: str | None = None, k: int = 10) -> list[str]:

        if k <= 0:
            return []
        return self._ranking[:k]

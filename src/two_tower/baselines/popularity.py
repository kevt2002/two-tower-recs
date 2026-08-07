"""Popularity baseline: every user gets the same top-K by train popularity."""

from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """Recommend the most popular items from the TRAIN set only.

    Hints:
      - Popularity = interaction count per item in train. The `rating` value
        doesn't matter here: an interaction is an interaction.
      - Ties: two items with the same count — what breaks the tie? Pick
        something deterministic, and think about whether it can even matter.
      - fit() must never see test data. Leaking test popularity into this
        baseline is how a "trivial" baseline accidentally beats a real model.
    """

    def fit(self, train: pd.DataFrame) -> None:
        raise NotImplementedError

    def recommend(self, user_id: str | None = None, k: int = 10) -> list[str]:
        """Return the k most popular item ids (highest to lowest).

        Note: `user_id` is accepted and ignored. This baseline is
        user-agnostic — that is the point of it.
        """
        raise NotImplementedError

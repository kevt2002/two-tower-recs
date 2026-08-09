"""Temporal train/val/test split: global cutoff on timestamp."""

from __future__ import annotations

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """Split interactions by time, not by row count.

    Contract:
      - every interaction lands in exactly one of the three splits
      - all timestamps in train <= all timestamps in val <= all timestamps in test
      - returns (train, val, test)

    Rule (row-based): sort by timestamp, then cut on row positions.
    The first int(n * train_frac) rows are train, the next int(n * val_frac)
    rows are val, the rest are test. Boundaries are exact row positions, so
    split sizes are exact. A timestamp shared by rows on both sides of a
    boundary may straddle two splits.
    """
    n = len(df)
    df = df.sort_values(by="timestamp")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test

"""Property tests for the temporal train/val/test split contract.

Contract (from src/two_tower/data/split.py):
  - every interaction lands in exactly one of the three splits
  - all timestamps in train <= all timestamps in val <= all timestamps in test
  - returns (train, val, test) DataFrames with the same columns as input
"""

import numpy as np
import pandas as pd
import pytest

from two_tower.data.split import temporal_split


def make_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "asin": rng.integers(0, 1000, n),
            "user": rng.integers(0, 10000, n),
            "timestamp": rng.integers(0, 1_000_000, n),
        }
    )


def test_partition_is_exact() -> None:
    df = make_frame()
    train, val, test = temporal_split(df)
    all_indices = np.concatenate(
        [train.index.to_numpy(), val.index.to_numpy(), test.index.to_numpy()]
    )
    assert len(all_indices) == len(df)
    assert len(np.unique(all_indices)) == len(df)


def test_temporal_ordering() -> None:
    df = make_frame()
    train, val, test = temporal_split(df)
    splits = [train, val, test]
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()
    for split in splits:
        assert not split.empty


def test_columns_preserved() -> None:
    df = make_frame()
    train, val, test = temporal_split(df)
    for split in (train, val, test):
        assert list(split.columns) == list(df.columns)


def test_exact_row_proportions() -> None:
    df = make_frame(n=1000)
    train, val, test = temporal_split(df)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_input_not_mutated() -> None:
    df = make_frame()
    before = df.copy(deep=True)
    temporal_split(df)
    pd.testing.assert_frame_equal(df, before)


def test_deterministic_across_calls() -> None:
    df = make_frame()
    first = temporal_split(df)
    second = temporal_split(df)
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_known_answer_small() -> None:
    df = pd.DataFrame({"timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    train, val, test = temporal_split(df)
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1
    assert list(train["timestamp"]) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert list(val["timestamp"]) == [9]
    assert list(test["timestamp"]) == [10]


def test_known_answer_with_ties() -> None:
    df = pd.DataFrame({"timestamp": [1, 1, 1, 2, 2, 3]})
    train, val, test = temporal_split(df, train_frac=0.4, val_frac=0.2)
    assert list(train["timestamp"]) == [1, 1]
    assert list(val["timestamp"]) == [1]
    assert list(test["timestamp"]) == [2, 2, 3]
    assert train["timestamp"].max() <= val["timestamp"].min()


def test_defaults_sum_to_full_frame() -> None:
    df = make_frame()
    train, val, test = temporal_split(df)
    assert len(train) + len(val) + len(test) == len(df)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ordering_on_multiple_seeds(seed: int) -> None:
    df = make_frame(seed=seed)
    train, val, test = temporal_split(df)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()

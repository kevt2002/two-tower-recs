"""Week 1 EDA: get a feel for the Video Games interactions before modeling.

Run:  python scripts/eda.py
First run downloads ~1.2GB (cached by Hugging Face afterwards).
"""

from __future__ import annotations

from two_tower.data.load import load_interactions


def main() -> None:
    df = load_interactions()

    print(df.shape)
    print(df.dtypes)
    print(df.head())
    print(f"time range: {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(df["rating"].value_counts().sort_index())
    print(f"users: {df['user_id'].nunique()}  items: {df['asin'].nunique()}")

    # Things worth digging into before you build anything:
    # - interactions per user: histogram — how long is the tail?
    # - duplicate (user, item) rows? How many?
    # - how many users / items appear exactly once? (your cold-start material)
    # - do you keep `rating`, or binarize everything to implicit feedback?
    #   That's a real modeling decision — say why either way.


if __name__ == "__main__":
    main()

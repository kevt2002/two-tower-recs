"""Load Amazon Reviews 2023 (Video Games) interactions into a DataFrame."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
REVIEW_PATH = "raw/review_categories/Video_Games.jsonl"
COLUMNS = ["user_id", "asin", "parent_asin", "rating", "timestamp"]


def load_interactions(cache_dir: Path = Path("data")) -> pd.DataFrame:
    """Download (once) and parse the raw Video Games reviews.

    The Hugging Face `datasets` library no longer supports this dataset's
    loading script, so we fetch the raw JSONL directly via huggingface_hub
    (it caches the download) and stream-parse only the columns in COLUMNS.
    Streaming matters: the raw lines also contain review text and image
    URLs, which we don't want materialized in memory.

    Timestamps are Unix epoch MILLISECONDS in the raw data.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        DATASET_ID,
        REVIEW_PATH,
        repo_type="dataset",
        local_dir=cache_dir,
    )
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rows.append(
                (
                    record["user_id"],
                    record["asin"],
                    record["parent_asin"],
                    record["rating"],
                    record["timestamp"],
                )
            )
    return pd.DataFrame(rows, columns=COLUMNS)

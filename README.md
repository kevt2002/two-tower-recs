# two-tower-recs

Two-tower retrieval recommender in PyTorch: a dual-encoder model that learns
separate user and item embeddings, trained on Amazon Reviews 2023 (Video Games
category), benchmarked against popularity and matrix-factorization baselines,
and served via FAISS + FastAPI.

Architecturally this is the same encode-query / encode-corpus-once / ANN-search
pattern that underlies retrieval in production RAG systems, applied to users and
items instead of queries and documents.

## Setup

Requires Python 3.10+ (developed on 3.14).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e .` installs the pinned core stack (PyTorch CPU, FAISS,
sentence-transformers, pandas, datasets, FastAPI, W&B). The matrix-factorization
baseline lives behind an extra and is only needed if you build it:

```bash
pip install -e ".[mf]"
```

## Verify

```bash
ruff check .
pytest
```

GitHub Actions runs both on every push.

## Layout

- `src/two_tower/` — package source
- `tests/` — pytest suite
- `DECISIONS.md` — running log of design decisions
- `data/`, `models/` — datasets and artifacts (gitignored)

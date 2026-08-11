# Popularity baseline results

Evaluated on the test split (2022-04-06 .. 2023-09-12), mean over test users
with non-empty ground truth. Metrics return 0.0 for empty-relevant users,
who are skipped in the mean (see DECISIONS.md).

Reproduce: `python scripts/evaluate_baseline.py`

| metric | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|
| recall | 0.0006 | 0.0092 | 0.0099 | 0.0143 | 0.0232 |
| ndcg | 0.0008 | 0.0053 | 0.0055 | 0.0066 | 0.0084 |
| mrr | 0.0008 | 0.0045 | 0.0046 | 0.0049 | 0.0052 |

## What the two-tower model must beat

- recall@10 ~ 0.010 (of 100 test interactions, ~1 emerges in the top-10)
- mrr@20 ~ 0.005 (first relevant item typically not found in the top-20)
- ndcg@10 ~ 0.006

These are the reference numbers for the comparison table.
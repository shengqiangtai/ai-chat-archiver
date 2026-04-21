from __future__ import annotations


def compute_recall_at_k(*, expected_chunk_ids: list[str], ranked_chunk_ids: list[str], k: int) -> float:
    if k <= 0 or not expected_chunk_ids:
        return 0.0

    expected = set(expected_chunk_ids)
    ranked = ranked_chunk_ids[:k]
    hits = len(expected.intersection(ranked))
    return hits / len(expected)


def compute_hit_rate_at_k(*, expected_chunk_ids: list[str], ranked_chunk_ids: list[str], k: int) -> float:
    if k <= 0 or not expected_chunk_ids:
        return 0.0
    expected = set(expected_chunk_ids)
    ranked = set(ranked_chunk_ids[:k])
    return 1.0 if expected.intersection(ranked) else 0.0


def compute_mrr(*, expected_chunk_ids: list[str], ranked_chunk_ids: list[str], k: int) -> float:
    if k <= 0 or not expected_chunk_ids:
        return 0.0

    expected = set(expected_chunk_ids)
    for index, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        if chunk_id in expected:
            return 1.0 / index
    return 0.0

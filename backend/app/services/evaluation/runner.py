from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.services.evaluation.metrics import (
    compute_hit_rate_at_k,
    compute_mrr,
    compute_recall_at_k,
)
from app.services.evaluation.models import BenchmarkCase, RetrievalEvalResult


def evaluate_retrieval_case(
    *,
    case: BenchmarkCase,
    ranked_chunk_ids: Sequence[Any],
    mode: str,
    elapsed_seconds: float | None = None,
) -> RetrievalEvalResult:
    normalized_ranked_chunk_ids = _normalize_ranked_chunk_ids(ranked_chunk_ids)
    recall_at_5 = compute_recall_at_k(
        expected_chunk_ids=case.expected_chunk_ids,
        ranked_chunk_ids=normalized_ranked_chunk_ids,
        k=5,
    )
    hit_rate_at_5 = compute_hit_rate_at_k(
        expected_chunk_ids=case.expected_chunk_ids,
        ranked_chunk_ids=normalized_ranked_chunk_ids,
        k=5,
    )
    recall_at_10 = compute_recall_at_k(
        expected_chunk_ids=case.expected_chunk_ids,
        ranked_chunk_ids=normalized_ranked_chunk_ids,
        k=10,
    )
    mrr_at_10 = compute_mrr(
        expected_chunk_ids=case.expected_chunk_ids,
        ranked_chunk_ids=normalized_ranked_chunk_ids,
        k=10,
    )
    return RetrievalEvalResult(
        case_id=case.id,
        question=case.question,
        question_type=case.question_type,
        difficulty=case.difficulty,
        source_type=case.source_type,
        mode=mode,
        expected_chunk_ids=list(case.expected_chunk_ids),
        ranked_chunk_ids=normalized_ranked_chunk_ids,
        recall_at_5=recall_at_5,
        hit_rate_at_5=hit_rate_at_5,
        recall_at_10=recall_at_10,
        mrr_at_10=mrr_at_10,
        elapsed_seconds=elapsed_seconds,
        requires_relation_reasoning=case.requires_relation_reasoning,
        requires_context_resolution=case.requires_context_resolution,
    )


def _normalize_ranked_chunk_ids(ranked_chunk_ids: Sequence[Any]) -> list[str]:
    normalized: list[str] = []
    for item in ranked_chunk_ids:
        chunk_id = _extract_chunk_id(item)
        if chunk_id:
            normalized.append(chunk_id)
    return normalized


def _extract_chunk_id(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        return str(item.get("chunk_id") or "").strip()
    chunk_id = getattr(item, "chunk_id", None)
    if chunk_id is not None:
        return str(chunk_id).strip()
    return ""

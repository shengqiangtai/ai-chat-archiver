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
    relevance_mode = _select_relevance_mode(case)
    expected_relevance_ids = _expected_relevance_ids(case, relevance_mode)
    normalized_ranked_chunk_ids = _normalize_ranked_relevance_ids(ranked_chunk_ids, relevance_mode)
    recall_at_5 = compute_recall_at_k(
        expected_chunk_ids=expected_relevance_ids,
        ranked_chunk_ids=normalized_ranked_chunk_ids,
        k=5,
    )
    hit_rate_at_5 = compute_hit_rate_at_k(
        expected_chunk_ids=expected_relevance_ids,
        ranked_chunk_ids=normalized_ranked_chunk_ids,
        k=5,
    )
    recall_at_10 = compute_recall_at_k(
        expected_chunk_ids=expected_relevance_ids,
        ranked_chunk_ids=normalized_ranked_chunk_ids,
        k=10,
    )
    mrr_at_10 = compute_mrr(
        expected_chunk_ids=expected_relevance_ids,
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
        expected_source_titles=list(case.expected_source_titles),
        ranked_chunk_ids=normalized_ranked_chunk_ids,
        recall_at_5=recall_at_5,
        hit_rate_at_5=hit_rate_at_5,
        recall_at_10=recall_at_10,
        mrr_at_10=mrr_at_10,
        elapsed_seconds=elapsed_seconds,
        requires_relation_reasoning=case.requires_relation_reasoning,
        requires_context_resolution=case.requires_context_resolution,
    )


def _select_relevance_mode(case: BenchmarkCase) -> str:
    if case.expected_chunk_ids:
        return "chunk_id"
    if case.expected_source_titles:
        return "source_title"
    return "none"


def _expected_relevance_ids(case: BenchmarkCase, relevance_mode: str) -> list[str]:
    if relevance_mode == "chunk_id":
        return [f"chunk_id::{chunk_id}" for chunk_id in case.expected_chunk_ids if chunk_id]
    if relevance_mode == "source_title":
        return [f"source_title::{title}" for title in case.expected_source_titles if title]
    return []


def _normalize_ranked_relevance_ids(ranked_chunk_ids: Sequence[Any], relevance_mode: str) -> list[str]:
    normalized: list[str] = []
    for index, item in enumerate(ranked_chunk_ids):
        if relevance_mode == "chunk_id":
            normalized.append(_canonical_id("chunk_id", _extract_chunk_id(item), index))
        elif relevance_mode == "source_title":
            normalized.append(_canonical_id("source_title", _extract_title(item), index))
        else:
            normalized.append(f"__non_relevant__::{index}")
    return normalized


def _canonical_id(namespace: str, value: str, index: int) -> str:
    if value:
        return f"{namespace}::{value}"
    return f"__non_relevant__::{index}"


def _extract_title(item: Any) -> str:
    if isinstance(item, str):
        return ""
    if isinstance(item, Mapping):
        return str(item.get("title") or "").strip()
    title = getattr(item, "title", None)
    if title is not None:
        return str(title).strip()
    return ""


def _extract_chunk_id(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        return str(item.get("chunk_id") or "").strip()
    chunk_id = getattr(item, "chunk_id", None)
    if chunk_id is not None:
        return str(chunk_id).strip()
    return ""

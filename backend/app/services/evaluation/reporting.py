from __future__ import annotations

from statistics import mean
from typing import Iterable

from app.services.evaluation.models import EvaluationSummary, RetrievalEvalResult


def build_evaluation_summary(*, mode: str, cases: Iterable[RetrievalEvalResult]) -> EvaluationSummary:
    results = list(cases)
    case_count = len(results)
    recall_at_5 = _average(result.recall_at_5 for result in results)
    hit_rate_at_5 = _average(result.hit_rate_at_5 for result in results)
    recall_at_10 = _average(result.recall_at_10 for result in results)
    mrr_at_10 = _average(result.mrr_at_10 for result in results)
    avg_elapsed_seconds = _average_optional(
        result.elapsed_seconds for result in results if result.elapsed_seconds is not None
    )
    return EvaluationSummary(
        mode=mode,
        case_count=case_count,
        recall_at_5=recall_at_5,
        hit_rate_at_5=hit_rate_at_5,
        recall_at_10=recall_at_10,
        mrr_at_10=mrr_at_10,
        avg_elapsed_seconds=avg_elapsed_seconds,
        cases=results,
    )


def _average(values: Iterable[float]) -> float:
    collected = list(values)
    return mean(collected) if collected else 0.0


def _average_optional(values: Iterable[float]) -> float | None:
    collected = list(values)
    return mean(collected) if collected else None

from __future__ import annotations

from typing import Iterable

from app.services.evaluation.models import EvaluationSummary, RetrievalEvalResult


def build_evaluation_summary(
    *,
    mode: str,
    cases: Iterable[RetrievalEvalResult],
    total_cases: int | None = None,
) -> EvaluationSummary:
    results = list(cases)
    total = max(total_cases if total_cases is not None else len(results), 0)
    evaluated_cases = len(results)
    failed_cases = max(total - evaluated_cases, 0)
    recall_at_5 = _average_over_total((result.recall_at_5 for result in results), total)
    hit_rate_at_5 = _average_over_total((result.hit_rate_at_5 for result in results), total)
    recall_at_10 = _average_over_total((result.recall_at_10 for result in results), total)
    mrr_at_10 = _average_over_total((result.mrr_at_10 for result in results), total)
    avg_elapsed_seconds = _average_optional(
        result.elapsed_seconds for result in results if result.elapsed_seconds is not None
    )
    return EvaluationSummary(
        mode=mode,
        total_cases=total,
        evaluated_cases=evaluated_cases,
        failed_cases=failed_cases,
        recall_at_5=recall_at_5,
        hit_rate_at_5=hit_rate_at_5,
        recall_at_10=recall_at_10,
        mrr_at_10=mrr_at_10,
        avg_elapsed_seconds=avg_elapsed_seconds,
        cases=results,
    )


def _average_over_total(values: Iterable[float], total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(values) / total


def _average_optional(values: Iterable[float]) -> float | None:
    collected = list(values)
    return (sum(collected) / len(collected)) if collected else None


def format_markdown_summary(summary: EvaluationSummary) -> str:
    graph_route_rate = summary.metadata.get("graph_route_rate")
    relation_win_rate = summary.metadata.get("relation_win_rate")
    avg_graph_hits = summary.metadata.get("avg_graph_hits")

    lines = [
        f"## Mode: {summary.mode}",
        f"- Total cases: {summary.total_cases}",
        f"- Evaluated cases: {summary.evaluated_cases}",
        f"- Failed cases: {summary.failed_cases}",
        f"- Recall@5: {summary.recall_at_5:.3f}",
        f"- HitRate@5: {summary.hit_rate_at_5:.3f}",
        f"- Recall@10: {summary.recall_at_10:.3f}",
        f"- MRR@10: {summary.mrr_at_10:.3f}",
        f"- Avg elapsed seconds: {summary.avg_elapsed_seconds:.3f}" if summary.avg_elapsed_seconds is not None else "- Avg elapsed seconds: n/a",
    ]
    if graph_route_rate is not None:
        lines.append(f"- Graph route rate: {float(graph_route_rate):.3f}")
    if relation_win_rate is not None:
        lines.append(f"- Relation win rate: {float(relation_win_rate):.3f}")
    if avg_graph_hits is not None:
        lines.append(f"- Avg graph hits: {float(avg_graph_hits):.3f}")
    return "\n".join(lines)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    question_type: str
    difficulty: str
    source_type: str
    requires_relation_reasoning: bool
    requires_context_resolution: bool
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_source_titles: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass(frozen=True)
class RetrievalEvalResult:
    case_id: str
    question: str
    question_type: str
    difficulty: str
    source_type: str
    mode: str
    recall_at_5: float
    hit_rate_at_5: float
    recall_at_10: float
    mrr_at_10: float
    elapsed_seconds: float | None
    requires_relation_reasoning: bool
    requires_context_resolution: bool
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_source_titles: list[str] = field(default_factory=list)
    ranked_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationSummary:
    mode: str
    total_cases: int
    evaluated_cases: int
    failed_cases: int
    recall_at_5: float
    hit_rate_at_5: float
    recall_at_10: float
    mrr_at_10: float
    avg_elapsed_seconds: float | None
    cases: list[RetrievalEvalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

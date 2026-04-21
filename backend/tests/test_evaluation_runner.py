from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.evaluation.metrics import (
    compute_hit_rate_at_k,
    compute_mrr,
    compute_recall_at_k,
)
from app.services.evaluation.models import BenchmarkCase, RetrievalEvalResult
from app.services.evaluation.reporting import build_evaluation_summary
from app.services.evaluation.runner import evaluate_retrieval_case


def test_recall_hits_when_expected_chunk_present() -> None:
    assert compute_recall_at_k(
        expected_chunk_ids=["chunk-b"],
        ranked_chunk_ids=["chunk-a", "chunk-b", "chunk-c"],
        k=2,
    ) == 1.0


def test_mrr_uses_first_relevant_rank() -> None:
    assert compute_mrr(
        expected_chunk_ids=["chunk-b"],
        ranked_chunk_ids=["chunk-a", "chunk-b", "chunk-c"],
        k=10,
    ) == 0.5


def test_hit_rate_is_binary() -> None:
    assert compute_hit_rate_at_k(
        expected_chunk_ids=["chunk-b"],
        ranked_chunk_ids=["chunk-a", "chunk-b", "chunk-c"],
        k=1,
    ) == 0.0


def test_evaluate_retrieval_case_preserves_relation_flags() -> None:
    case = BenchmarkCase(
        id="relation-001",
        question="Which docs connect the two features?",
        expected_chunk_ids=["chunk-b"],
        question_type="relation",
        difficulty="medium",
        source_type="chat",
        requires_relation_reasoning=True,
        requires_context_resolution=False,
    )

    result = evaluate_retrieval_case(
        case=case,
        ranked_chunk_ids=["chunk-a", "chunk-b", "chunk-c"],
        mode="mix",
        elapsed_seconds=1.25,
    )

    assert result.recall_at_5 == 1.0
    assert result.mrr_at_10 == 0.5
    assert result.requires_relation_reasoning is True
    assert result.requires_context_resolution is False


def test_evaluate_retrieval_case_uses_source_title_fallback() -> None:
    case = BenchmarkCase(
        id="source-title-001",
        question="Where is the install guide?",
        question_type="installation",
        difficulty="easy",
        source_type="chat",
        requires_relation_reasoning=False,
        requires_context_resolution=False,
        expected_source_titles=["Codex Skills 安装指南"],
    )

    result = evaluate_retrieval_case(
        case=case,
        ranked_chunk_ids=[
            {"chunk_id": "doc-1_0", "title": "Other Guide"},
            {"chunk_id": "doc-2_0", "title": "Codex Skills 安装指南"},
        ],
        mode="mix",
        elapsed_seconds=1.25,
    )

    assert result.recall_at_5 == 1.0
    assert result.mrr_at_10 == 0.5
    assert result.expected_source_titles == ["Codex Skills 安装指南"]


def test_summary_metrics_do_not_ignore_failed_cases() -> None:
    summary = build_evaluation_summary(
        mode="mix",
        total_cases=2,
        cases=[
            RetrievalEvalResult(
                case_id="ok-001",
                question="Where is the install guide?",
                question_type="installation",
                difficulty="easy",
                source_type="chat",
                mode="mix",
                recall_at_5=1.0,
                hit_rate_at_5=1.0,
                recall_at_10=1.0,
                mrr_at_10=1.0,
                elapsed_seconds=1.2,
                requires_relation_reasoning=False,
                requires_context_resolution=False,
            )
        ],
    )

    assert summary.total_cases == 2
    assert summary.evaluated_cases == 1
    assert summary.failed_cases == 1
    assert summary.recall_at_5 == 0.5
    assert summary.hit_rate_at_5 == 0.5
    assert summary.recall_at_10 == 0.5
    assert summary.mrr_at_10 == 0.5

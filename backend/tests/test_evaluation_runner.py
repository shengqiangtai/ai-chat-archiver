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
from app.services.evaluation.models import BenchmarkCase
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

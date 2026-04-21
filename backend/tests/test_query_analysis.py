from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.retrieval.query_analysis import analyze_query


def test_symbolic_path_query_skips_graph_and_rewrite() -> None:
    analysis = analyze_query("backend/app/services/qa/pipeline.py")

    assert analysis.query_type == "symbolic"
    assert analysis.enable_rewrite is False
    assert analysis.enable_graph is False


def test_relation_query_enables_graph_and_rerank() -> None:
    analysis = analyze_query("Which module depends on the reranker?")

    assert analysis.query_type == "relation"
    assert analysis.enable_graph is True
    assert analysis.enable_rerank is True


def test_follow_up_query_enables_rewrite() -> None:
    analysis = analyze_query("那它为什么会超时？")

    assert analysis.enable_rewrite is True

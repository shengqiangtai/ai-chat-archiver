from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.evaluation.models import EvaluationSummary
from app.services.evaluation.reporting import format_markdown_summary


def test_markdown_summary_includes_graph_metrics() -> None:
    summary = EvaluationSummary(
        mode="mix_graph",
        total_cases=10,
        evaluated_cases=10,
        failed_cases=0,
        recall_at_5=0.7,
        hit_rate_at_5=0.8,
        recall_at_10=0.9,
        mrr_at_10=0.6,
        avg_elapsed_seconds=0.4,
        metadata={
            "graph_route_rate": 0.5,
            "relation_win_rate": 0.75,
            "avg_graph_hits": 1.2,
        },
    )

    markdown = format_markdown_summary(summary)

    assert "Graph route rate" in markdown
    assert "Relation win rate" in markdown
    assert "Avg graph hits" in markdown

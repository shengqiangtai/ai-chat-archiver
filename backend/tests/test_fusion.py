from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models.schemas import RetrievalHit
from app.services.retrieval.fusion import fuse_candidates


def _hit(
    chunk_id: str,
    *,
    score: float = 0.0,
    keyword_score: float | None = None,
    entity_score: float | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        score=score,
        rerank_score=None,
        platform="ChatGPT",
        title=f"Title {chunk_id}",
        excerpt=f"Excerpt {chunk_id}",
        path=f"/tmp/{chunk_id}.md",
        created_at="2024-01-01",
        keyword_score=keyword_score,
        entity_score=entity_score,
    )


def test_keyword_exact_match_stays_ahead_of_graph_only_hit_on_symbolic_query() -> None:
    hits = fuse_candidates(
        dense_hits=[],
        keyword_hits=[_hit("chunk-a", keyword_score=0.92)],
        entity_hits=[_hit("chunk-c", entity_score=0.20)],
        retrieval_mode="mix",
    )

    assert [hit.chunk_id for hit in hits] == ["chunk-a", "chunk-c"]


def test_graph_score_is_capped_but_can_help_relation_query() -> None:
    hits = fuse_candidates(
        dense_hits=[_hit("chunk-a", score=0.54), _hit("chunk-b", score=0.51)],
        keyword_hits=[],
        entity_hits=[_hit("chunk-b", entity_score=1.0), _hit("chunk-c", entity_score=1.0)],
        retrieval_mode="mix",
    )

    assert [hit.chunk_id for hit in hits] == ["chunk-b", "chunk-a", "chunk-c"]

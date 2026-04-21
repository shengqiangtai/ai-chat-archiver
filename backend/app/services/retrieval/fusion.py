"""Candidate fusion helpers for retrieval."""

from __future__ import annotations

from typing import Dict, List, Optional

from app.models.schemas import RetrievalHit

GRAPH_SCORE_CAP = 0.05


def fuse_candidates(
    dense_hits: List[RetrievalHit],
    keyword_hits: List[RetrievalHit],
    entity_hits: List[RetrievalHit],
    retrieval_mode: str,
) -> List[RetrievalHit]:
    mode = (retrieval_mode or "mix").strip().lower()
    if mode == "vector":
        return _normalize_scores(dense_hits, attr="score")
    if mode == "keyword":
        return _normalize_scores(keyword_hits, attr="keyword_score")
    if mode == "entity":
        return _normalize_scores(entity_hits, attr="entity_score")
    return _fuse_mix_candidates(dense_hits, keyword_hits, entity_hits)


def _fuse_mix_candidates(
    dense_hits: List[RetrievalHit],
    keyword_hits: List[RetrievalHit],
    entity_hits: List[RetrievalHit],
) -> List[RetrievalHit]:
    merged: Dict[str, RetrievalHit] = {}

    for hit in dense_hits:
        merged[hit.chunk_id] = hit
        hit.fused_score = float(hit.score)

    for hit in keyword_hits:
        keyword_score = _source_score(hit.keyword_score, hit.score)
        existing = merged.get(hit.chunk_id)
        if existing is None:
            hit.keyword_score = keyword_score
            hit.fused_score = keyword_score
            merged[hit.chunk_id] = hit
            continue

        existing.keyword_score = keyword_score
        existing.fused_score = _add_score(existing.fused_score, keyword_score)

    for hit in entity_hits:
        entity_score = _source_score(hit.entity_score, hit.score)
        graph_score = _graph_contribution(entity_score)
        existing = merged.get(hit.chunk_id)
        if existing is None:
            hit.entity_score = entity_score
            hit.fused_score = graph_score
            merged[hit.chunk_id] = hit
            continue

        existing.entity_score = entity_score
        existing.entity_names = list(dict.fromkeys([*existing.entity_names, *hit.entity_names]))
        existing.fused_score = _add_score(existing.fused_score, graph_score)

    combined = sorted(
        merged.values(),
        key=lambda hit: hit.fused_score or hit.score or hit.keyword_score or hit.entity_score or 0.0,
        reverse=True,
    )
    return _normalize_scores(combined, attr="fused_score")


def _source_score(primary: Optional[float], fallback: Optional[float]) -> float:
    if primary is not None:
        return float(primary)
    if fallback is not None:
        return float(fallback)
    return 0.0


def _add_score(current: Optional[float], increment: float) -> float:
    return float(current or 0.0) + float(increment)


def _graph_contribution(entity_score: float) -> float:
    bounded = max(0.0, min(float(entity_score), 1.0))
    return bounded * GRAPH_SCORE_CAP


def _normalize_scores(hits: List[RetrievalHit], attr: str) -> List[RetrievalHit]:
    if not hits:
        return []

    raw_scores = [float(getattr(hit, attr) or 0.0) for hit in hits]
    max_score = max(raw_scores)
    min_score = min(raw_scores)

    normalized: list[RetrievalHit] = []
    for hit, raw in zip(hits, raw_scores):
        if max_score > min_score:
            hit.score = (raw - min_score) / (max_score - min_score)
        else:
            hit.score = 1.0
        normalized.append(hit)

    normalized.sort(key=lambda hit: hit.score, reverse=True)
    return normalized

"""Lightweight graph-assisted retrieval over persisted relation metadata."""

from __future__ import annotations

from app.db.sqlite import get_db
from app.models.schemas import RetrievalHit
from app.services.ingest.entity_extractor import extract_query_entities


def retrieve_graph_candidates(
    query: str,
    *,
    top_k: int,
    platform_filter: str | None = None,
    model_filter: str | None = None,
    tag_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[RetrievalHit]:
    query_entities = extract_query_entities(query)
    if not query_entities:
        return []

    db = get_db()
    placeholders = ",".join("?" for _ in query_entities)
    filters: list[str] = []
    params: list[object] = [*query_entities, *query_entities]

    if platform_filter:
        filters.append("c.platform = ?")
        params.append(platform_filter)
    if model_filter:
        filters.append("c.model_name = ?")
        params.append(model_filter)
    if tag_filter:
        filters.append("c.tags LIKE ?")
        params.append(f"%{tag_filter}%")
    if date_from:
        filters.append("c.created_at >= ?")
        params.append(date_from)
    if date_to:
        filters.append("c.created_at <= ?")
        params.append(date_to)

    where_sql = f" AND {' AND '.join(filters)}" if filters else ""
    with db._conn() as conn:
        rows = conn.execute(
            f"""SELECT c.*,
                       COUNT(DISTINCT r.relation_id) AS graph_match_count,
                       GROUP_CONCAT(DISTINCT r.source_entity) AS graph_sources,
                       GROUP_CONCAT(DISTINCT r.target_entity) AS graph_targets
                FROM kb_graph_relations r
                JOIN kb_chunks c ON c.chunk_id = r.chunk_id
                WHERE (r.source_norm_name IN ({placeholders}) OR r.target_norm_name IN ({placeholders})){where_sql}
                GROUP BY c.chunk_id
                ORDER BY graph_match_count DESC, c.created_at DESC
                LIMIT ?""",
            [*params, max(1, top_k)],
        ).fetchall()

    hits: list[RetrievalHit] = []
    for row in rows:
        item = dict(row)
        tags_val = item.get("tags") or []
        tags = tags_val if isinstance(tags_val, list) else [tag for tag in str(tags_val).split(",") if tag]
        entity_names = list(
            dict.fromkeys(
                [
                    *[name for name in str(item.get("graph_sources") or "").split(",") if name],
                    *[name for name in str(item.get("graph_targets") or "").split(",") if name],
                ]
            )
        )
        hits.append(
            RetrievalHit(
                chunk_id=str(item.get("chunk_id") or ""),
                doc_id=str(item.get("doc_id") or ""),
                score=0.0,
                rerank_score=None,
                platform=str(item.get("platform") or "Unknown"),
                title=str(item.get("title") or "Untitled"),
                excerpt=str(item.get("content") or item.get("text") or item.get("excerpt") or ""),
                path=str(item.get("source_path") or item.get("path") or ""),
                created_at=str(item.get("created_at") or ""),
                url=str(item.get("url") or "") or None,
                entity_score=float(item.get("graph_match_count") or 0.0),
                role_summary=str(item.get("role_summary") or ""),
                message_range=str(item.get("message_range") or ""),
                model_name=str(item.get("model_name") or "") or None,
                tags=tags,
                entity_names=entity_names,
                turn_index=int(item.get("turn_index") or 0),
                chunk_index=int(item.get("chunk_index") or 0),
            )
        )
    return hits

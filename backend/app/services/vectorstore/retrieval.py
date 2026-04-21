"""检索模块 — 封装 hybrid retrieval + rerank + 邻近轮次扩展流程。"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Optional

from app.core.config import (
    DEFAULT_RERANK_MODE,
    RERANK_CANDIDATE_LIMIT,
    RERANK_TIMEOUT_MS,
    RERANK_TOP_N,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from app.db.sqlite import get_db
from app.models.schemas import QueryAnalysis, RetrievalHit
from app.services.graph.retrieval import retrieve_graph_candidates
from app.services.retrieval.fusion import fuse_candidates
from app.services.cache.query_cache import get_cache
from app.services.embedding.embedder import get_embedder
from app.services.ingest.entity_extractor import extract_query_entities, normalize_entity_name
from app.services.retrieval.query_analysis import analyze_query
from app.services.vectorstore.chroma_store import get_store

logger = logging.getLogger("archiver.retrieval")


def _rerank_reason_label(reason: str) -> str:
    mapping = {
        "disabled": "已关闭 rerank",
        "no_candidates": "没有可精排的候选",
        "candidate_limit_exceeded": "候选数超过阈值，已跳过 rerank",
        "within_limit": "候选数在阈值内，准备执行 rerank",
        "forced": "已强制执行 rerank",
        "forced_clipped": "已强制执行 rerank，但只处理前几条候选",
        "model_unavailable": "reranker 不可用，已回退到原始检索顺序",
        "timeout": "rerank 超时，已回退到原始检索顺序",
        "ok": "rerank 已执行",
        "cache_hit": "命中缓存，未重新执行 rerank",
    }
    if reason.startswith("error:"):
        return f"rerank 执行失败，已回退到原始检索顺序（{reason.split(':', 1)[1]}）"
    return mapping.get(reason, reason or "未知原因")


def _rerank_status(*, applied: bool, fallback: bool) -> str:
    if applied:
        return "applied"
    if fallback:
        return "fallback"
    return "skipped"


def _effective_rerank_mode(*, use_rerank: bool, requested_mode: str) -> str:
    if not use_rerank:
        return "off"
    if requested_mode == "off":
        return "off"
    if requested_mode == "auto":
        return "auto"
    return "on"


def _query_analysis_debug(query_analysis: QueryAnalysis) -> dict[str, object]:
    return {
        "query_analysis": asdict(query_analysis),
        "analysis_scope": "retrieval_query",
    }


def _graph_debug(*, graph_routed: bool, graph_hits: list[RetrievalHit]) -> dict[str, object]:
    return {
        "graph_routed": graph_routed,
        "graph_hit_count": len(graph_hits),
        "graph_hits": [asdict(hit) for hit in graph_hits[:5]],
    }


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    top_n: int = RERANK_TOP_N,
    platform_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    tag_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    use_rerank: bool = True,
    retrieval_mode: str = "mix",
    expand_neighbors: bool = True,
    neighbor_turn_window: int = 1,
    use_cache: bool = True,
    rerank_mode: str = DEFAULT_RERANK_MODE,
    rerank_timeout_ms: int = RERANK_TIMEOUT_MS,
    rerank_candidate_limit: int = RERANK_CANDIDATE_LIMIT,
) -> list[RetrievalHit]:
    hits, _ = _retrieve_impl(
        query=query,
        top_k=top_k,
        top_n=top_n,
        platform_filter=platform_filter,
        model_filter=model_filter,
        tag_filter=tag_filter,
        date_from=date_from,
        date_to=date_to,
        score_threshold=score_threshold,
        use_rerank=use_rerank,
        retrieval_mode=retrieval_mode,
        expand_neighbors=expand_neighbors,
        neighbor_turn_window=neighbor_turn_window,
        use_cache=use_cache,
        rerank_mode=rerank_mode,
        rerank_timeout_ms=rerank_timeout_ms,
        rerank_candidate_limit=rerank_candidate_limit,
        include_debug=False,
    )
    return hits


def retrieve_debug(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    top_n: int = RERANK_TOP_N,
    platform_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    tag_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    use_rerank: bool = True,
    retrieval_mode: str = "mix",
    expand_neighbors: bool = False,
    neighbor_turn_window: int = 1,
    use_cache: bool = True,
    rerank_mode: str = DEFAULT_RERANK_MODE,
    rerank_timeout_ms: int = RERANK_TIMEOUT_MS,
    rerank_candidate_limit: int = RERANK_CANDIDATE_LIMIT,
) -> dict:
    hits, debug = _retrieve_impl(
        query=query,
        top_k=top_k,
        top_n=top_n,
        platform_filter=platform_filter,
        model_filter=model_filter,
        tag_filter=tag_filter,
        date_from=date_from,
        date_to=date_to,
        score_threshold=score_threshold,
        use_rerank=use_rerank,
        retrieval_mode=retrieval_mode,
        expand_neighbors=expand_neighbors,
        neighbor_turn_window=neighbor_turn_window,
        use_cache=use_cache,
        rerank_mode=rerank_mode,
        rerank_timeout_ms=rerank_timeout_ms,
        rerank_candidate_limit=rerank_candidate_limit,
        include_debug=True,
    )
    return {
        "hits": [asdict(hit) for hit in hits],
        "debug": debug,
    }


def _retrieve_impl(
    *,
    query: str,
    top_k: int,
    top_n: int,
    platform_filter: Optional[str],
    model_filter: Optional[str],
    tag_filter: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    score_threshold: float,
    use_rerank: bool,
    retrieval_mode: str,
    expand_neighbors: bool,
    neighbor_turn_window: int,
    use_cache: bool,
    rerank_mode: str,
    rerank_timeout_ms: int,
    rerank_candidate_limit: int,
    include_debug: bool,
) -> tuple[list[RetrievalHit], dict]:
    user_query = preprocess_query(query)
    if not user_query:
        return [], {}

    retrieval_mode = (retrieval_mode or "mix").strip().lower()
    if retrieval_mode not in {"hybrid", "vector", "keyword", "entity", "mix"}:
        retrieval_mode = "mix"
    requested_rerank_mode = (rerank_mode or DEFAULT_RERANK_MODE).strip().lower()
    if requested_rerank_mode not in {"auto", "off", "on"}:
        requested_rerank_mode = DEFAULT_RERANK_MODE
    query_analysis = analyze_query(user_query)
    effective_use_rerank = use_rerank and query_analysis.enable_rerank
    effective_expand_neighbors = expand_neighbors
    effective_rerank_mode = _effective_rerank_mode(
        use_rerank=effective_use_rerank,
        requested_mode=requested_rerank_mode,
    )

    cache_options = {
        "top_k": top_k,
        "top_n": top_n,
        "platform_filter": platform_filter,
        "model_filter": model_filter,
        "tag_filter": tag_filter,
        "date_from": date_from,
        "date_to": date_to,
        "score_threshold": score_threshold,
        "use_rerank": effective_use_rerank,
        "retrieval_mode": retrieval_mode,
        "expand_neighbors": expand_neighbors,
        "neighbor_turn_window": neighbor_turn_window,
        "rerank_mode": requested_rerank_mode,
        "rerank_timeout_ms": rerank_timeout_ms,
        "rerank_candidate_limit": rerank_candidate_limit,
    }

    if use_cache:
        cached = get_cache().get_retrieval(user_query, cache_options)
        if cached is not None:
            hits = [RetrievalHit(**item) for item in cached]
            debug = {
                "cache_hit": True,
                "retrieval_mode": retrieval_mode,
                "dense_count": None,
                "keyword_count": None,
                "entity_count": None,
                "candidate_count": len(hits),
                "final_count": len(hits),
                "query_entities": [],
                "expanded_entities": [],
                "rerank_requested_mode": requested_rerank_mode,
                "rerank_effective_mode": effective_rerank_mode,
                "rerank_applied": False,
                "rerank_status": "skipped",
                "rerank_reason": "cache_hit",
                "rerank_message": _rerank_reason_label("cache_hit"),
                "rerank_fallback": False,
                "rerank_timed_out": False,
                "rerank_elapsed_ms": 0,
                "rerank_candidate_limit": rerank_candidate_limit,
                "rerank_candidate_count": len(hits),
                "dense_hits": [],
                "keyword_hits": [],
                "entity_hits": [],
                "candidate_hits": [],
                "final_hits": [asdict(hit) for hit in hits],
                **_graph_debug(graph_routed=False, graph_hits=[]),
                **_query_analysis_debug(query_analysis),
            } if include_debug else {}
            logger.info("命中检索缓存: query=%r", user_query[:50])
            return hits, debug

    t0 = time.time()
    dense_hits: list[RetrievalHit] = []
    keyword_hits: list[RetrievalHit] = []
    entity_hits: list[RetrievalHit] = []
    graph_hits: list[RetrievalHit] = []
    query_entities: list[str] = []
    expanded_entities: list[str] = []
    t_embed = 0.0
    t_search = 0.0
    rerank_info = {
        "requested_mode": requested_rerank_mode,
        "effective_mode": "off",
        "applied": False,
        "reason": "disabled",
        "fallback": False,
        "timed_out": False,
        "elapsed_ms": 0,
        "candidate_limit": max(1, rerank_candidate_limit),
        "candidate_count": 0,
        "message": _rerank_reason_label("disabled"),
        "status": "skipped",
    }

    if retrieval_mode in {"hybrid", "vector", "mix"}:
        t_embed_start = time.time()
        embedder = get_embedder()
        query_embedding = embedder.encode_query(user_query)
        t_embed = time.time() - t_embed_start

        dense_hits = get_store().query(
            query_embedding=query_embedding,
            top_k=max(top_k, RETRIEVAL_TOP_K),
            filter_platform=platform_filter,
        )
        dense_hits = _apply_metadata_filters(
            dense_hits,
            model_filter=model_filter,
            tag_filter=tag_filter,
            date_from=date_from,
            date_to=date_to,
        )

    if retrieval_mode in {"hybrid", "keyword", "mix"}:
        keyword_rows = get_db().search_kb_chunks(
            query=user_query,
            platform=platform_filter,
            model_name=model_filter,
            tag=tag_filter,
            date_from=date_from,
            date_to=date_to,
            limit=max(top_k, RETRIEVAL_TOP_K),
        )
        keyword_hits = [_row_to_hit(row) for row in keyword_rows]

    if retrieval_mode in {"entity", "mix"} and query_analysis.enable_graph:
        query_entities = extract_query_entities(user_query)
        seed_entities = get_db().search_entities(query_entities, limit=max(4, top_k))
        expanded_entities = [normalize_entity_name(str(item.get("norm_name") or "")) for item in seed_entities]
        if seed_entities:
            related = get_db().get_related_entities(
                [str(item["entity_id"]) for item in seed_entities],
                limit=max(4, top_k // 2),
            )
            expanded_entities.extend(
                normalize_entity_name(str(item.get("norm_name") or "")) for item in related
            )
        expanded_entities = [name for name in dict.fromkeys(expanded_entities) if name]
        entity_rows = get_db().search_entity_chunks(
            entity_names=expanded_entities or query_entities,
            platform=platform_filter,
            model_name=model_filter,
            tag=tag_filter,
            date_from=date_from,
            date_to=date_to,
            limit=max(top_k, RETRIEVAL_TOP_K),
        )
        entity_hits = [_row_to_hit(row) for row in entity_rows]
        graph_hits = retrieve_graph_candidates(
            user_query,
            top_k=max(top_k, RETRIEVAL_TOP_K),
            platform_filter=platform_filter,
            model_filter=model_filter,
            tag_filter=tag_filter,
            date_from=date_from,
            date_to=date_to,
        )
        entity_hits.extend(graph_hits)

    t_search = time.time() - t0 - t_embed
    candidates = fuse_candidates(dense_hits, keyword_hits, entity_hits, retrieval_mode)
    candidate_snapshot = [asdict(hit) for hit in candidates[:8]] if include_debug else []
    rerank_info["candidate_count"] = len(candidates)

    rerank_plan = _plan_rerank(
        use_rerank=effective_use_rerank,
        requested_mode=requested_rerank_mode,
        candidate_count=len(candidates),
        candidate_limit=max(1, rerank_candidate_limit),
    )
    rerank_info.update(
        {
            "effective_mode": rerank_plan["effective_mode"],
            "applied": False,
            "reason": rerank_plan["reason"],
            "message": _rerank_reason_label(str(rerank_plan["reason"])),
            "status": "skipped",
        }
    )

    if rerank_plan["apply"] and candidates:
        logger.info(
            "准备执行 rerank: query=%r, requested=%s, effective=%s, candidates=%d/%d, reason=%s",
            user_query[:50],
            requested_rerank_mode,
            rerank_plan["effective_mode"],
            len(candidates),
            max(1, rerank_candidate_limit),
            rerank_plan["reason"],
        )
        try:
            from app.services.rerank.reranker import get_reranker

            reranker = get_reranker()
            rerank_count = rerank_plan["rerank_count"]
            rerank_return_n = min(rerank_count, max(top_n * 2, top_n))
            rerank_subset = candidates[:rerank_count]
            reranked_hits, rerank_meta = reranker.rerank(
                user_query,
                rerank_subset,
                top_n=rerank_return_n,
                timeout_ms=rerank_timeout_ms,
            )
            if rerank_plan["effective_mode"] == "on" and rerank_count < len(candidates):
                candidates = reranked_hits + candidates[rerank_count:]
            else:
                candidates = reranked_hits
            rerank_info.update(
                {
                    "applied": rerank_meta.get("applied", False),
                    "reason": rerank_meta.get("reason") or rerank_plan["reason"],
                    "message": rerank_meta.get("message")
                    or _rerank_reason_label(str(rerank_meta.get("reason") or rerank_plan["reason"])),
                    "fallback": bool(rerank_meta.get("fallback")),
                    "timed_out": bool(rerank_meta.get("timed_out")),
                    "elapsed_ms": rerank_meta.get("elapsed_ms", 0),
                    "candidate_count": rerank_count,
                    "status": rerank_meta.get("status")
                    or _rerank_status(
                        applied=bool(rerank_meta.get("applied")),
                        fallback=bool(rerank_meta.get("fallback")),
                    ),
                }
            )
            logger.info(
                "Rerank 结果: query=%r, status=%s, reason=%s, fallback=%s, timed_out=%s, elapsed=%sms, scored=%d",
                user_query[:50],
                rerank_info["status"],
                rerank_info["reason"],
                rerank_info["fallback"],
                rerank_info["timed_out"],
                rerank_info["elapsed_ms"],
                rerank_meta.get("scored_candidates", 0),
            )
        except Exception as e:
            logger.warning("Rerank 失败并回退: query=%r, error=%s", user_query[:50], e)
            rerank_info.update(
                {
                    "applied": False,
                    "reason": f"error:{type(e).__name__}",
                    "message": _rerank_reason_label(f"error:{type(e).__name__}"),
                    "fallback": True,
                    "status": "fallback",
                }
            )
    elif candidates:
        logger.info(
            "跳过 rerank: query=%r, requested=%s, effective=%s, candidates=%d/%d, reason=%s",
            user_query[:50],
            requested_rerank_mode,
            rerank_plan["effective_mode"],
            len(candidates),
            max(1, rerank_candidate_limit),
            rerank_plan["reason"],
        )

    filtered = _filter_hits(candidates, score_threshold=score_threshold, final_limit=max(1, top_n))
    if effective_expand_neighbors and filtered:
        filtered = _expand_neighbor_turns(filtered, window=neighbor_turn_window)

    total_time = time.time() - t0
    logger.info(
        "检索完成: query=%r, mode=%s, dense=%d, keyword=%d, entity=%d, 最终=%d, embed=%.2fs, search=%.2fs, total=%.2fs",
        user_query[:50],
        retrieval_mode,
        len(dense_hits),
        len(keyword_hits),
        len(entity_hits),
        len(filtered),
        t_embed,
        t_search,
        total_time,
    )

    if use_cache:
        get_cache().set_retrieval(
            user_query,
            [asdict(hit) for hit in filtered],
            cache_options,
        )

    debug = {
        "cache_hit": False,
        "retrieval_mode": retrieval_mode,
        "dense_count": len(dense_hits),
        "keyword_count": len(keyword_hits),
        "entity_count": len(entity_hits),
        "candidate_count": len(candidates),
        "final_count": len(filtered),
        "query_entities": query_entities,
        "expanded_entities": expanded_entities,
        **_graph_debug(
            graph_routed=retrieval_mode in {"entity", "mix"} and query_analysis.enable_graph,
            graph_hits=graph_hits,
        ),
        "rerank_requested_mode": rerank_info["requested_mode"],
        "rerank_effective_mode": rerank_info["effective_mode"],
        "rerank_applied": rerank_info["applied"],
        "rerank_status": rerank_info["status"],
        "rerank_reason": rerank_info["reason"],
        "rerank_message": rerank_info["message"],
        "rerank_fallback": rerank_info["fallback"],
        "rerank_timed_out": rerank_info["timed_out"],
        "rerank_elapsed_ms": rerank_info["elapsed_ms"],
        "rerank_candidate_limit": rerank_info["candidate_limit"],
        "rerank_candidate_count": rerank_info["candidate_count"],
        **_query_analysis_debug(query_analysis),
        "embed_time": round(t_embed, 3),
        "search_time": round(t_search, 3),
        "total_time": round(total_time, 3),
        "dense_hits": [asdict(hit) for hit in dense_hits[:5]],
        "keyword_hits": [asdict(hit) for hit in keyword_hits[:5]],
        "entity_hits": [asdict(hit) for hit in entity_hits[:5]],
        "candidate_hits": candidate_snapshot,
        "final_hits": [asdict(hit) for hit in filtered[:8]],
    } if include_debug else {}

    return filtered, debug


def _plan_rerank(
    *,
    use_rerank: bool,
    requested_mode: str,
    candidate_count: int,
    candidate_limit: int,
) -> dict[str, object]:
    if not use_rerank:
        return {"apply": False, "effective_mode": "off", "reason": "disabled", "rerank_count": 0}
    if candidate_count <= 0:
        return {"apply": False, "effective_mode": requested_mode, "reason": "no_candidates", "rerank_count": 0}
    if requested_mode == "off":
        return {"apply": False, "effective_mode": "off", "reason": "disabled", "rerank_count": 0}
    if requested_mode == "auto":
        if candidate_count > candidate_limit:
            return {
                "apply": False,
                "effective_mode": "auto",
                "reason": "candidate_limit_exceeded",
                "rerank_count": 0,
            }
        return {
            "apply": True,
            "effective_mode": "auto",
            "reason": "within_limit",
            "rerank_count": candidate_count,
        }
    rerank_count = min(candidate_count, candidate_limit)
    reason = "forced" if rerank_count == candidate_count else "forced_clipped"
    return {"apply": True, "effective_mode": "on", "reason": reason, "rerank_count": rerank_count}


def preprocess_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""

    prefixes_to_remove = [
        "请问", "帮我", "请帮我", "我想知道", "我想问",
        "你能告诉我", "能不能帮我", "帮忙", "麻烦",
    ]
    for prefix in prefixes_to_remove:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    replacements = {
        "这次": "", "之前": "", "上次": "", "那个聊天": "",
        "那次对话": "", "那个对话": "",
    }
    for old, new in replacements.items():
        q = q.replace(old, new)

    return q.strip()


def _apply_metadata_filters(
    hits: list[RetrievalHit],
    *,
    model_filter: str | None,
    tag_filter: str | None,
    date_from: str | None,
    date_to: str | None,
) -> list[RetrievalHit]:
    filtered = []
    for hit in hits:
        if model_filter and hit.model_name != model_filter:
            continue
        if tag_filter and tag_filter not in hit.tags:
            continue
        if date_from and hit.created_at and hit.created_at < date_from:
            continue
        if date_to and hit.created_at and hit.created_at > date_to:
            continue
        filtered.append(hit)
    return filtered


def _filter_hits(
    candidates: list[RetrievalHit],
    *,
    score_threshold: float,
    final_limit: int,
) -> list[RetrievalHit]:
    filtered: list[RetrievalHit] = []
    per_doc_count: dict[str, int] = {}

    for hit in candidates:
        effective_score = hit.rerank_score if hit.rerank_score is not None else hit.score
        if hit.rerank_score is None and effective_score < score_threshold:
            continue
        used = per_doc_count.get(hit.doc_id, 0)
        if used >= 2:
            continue
        per_doc_count[hit.doc_id] = used + 1
        filtered.append(hit)
        if len(filtered) >= final_limit:
            break

    return filtered


def _expand_neighbor_turns(hits: list[RetrievalHit], window: int = 1) -> list[RetrievalHit]:
    db = get_db()
    expanded: list[RetrievalHit] = []
    seen: set[str] = set()

    for seed in hits:
        rows = db.get_chunks_in_turn_window(seed.doc_id, seed.turn_index, window=window)
        if not rows:
            rows = [_hit_to_row(seed)]
        for row in rows:
            hit = _row_to_hit(row, fallback=seed)
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            expanded.append(hit)

    return expanded


def _row_to_hit(row: dict, fallback: RetrievalHit | None = None) -> RetrievalHit:
    tags_val = row.get("tags") or []
    tags = tags_val if isinstance(tags_val, list) else [t for t in str(tags_val).split(",") if t]
    return RetrievalHit(
        chunk_id=str(row.get("chunk_id") or (fallback.chunk_id if fallback else "")),
        doc_id=str(row.get("doc_id") or (fallback.doc_id if fallback else "")),
        score=float(row.get("score") or (fallback.score if fallback else 0.0)),
        rerank_score=row.get("rerank_score") if row.get("rerank_score") is not None else (
            fallback.rerank_score if fallback else None
        ),
        platform=str(row.get("platform") or (fallback.platform if fallback else "Unknown")),
        title=str(row.get("title") or (fallback.title if fallback else "Untitled")),
        excerpt=str(row.get("content") or row.get("excerpt") or row.get("snippet") or (
            fallback.excerpt if fallback else ""
        )),
        path=str(row.get("source_path") or row.get("path") or (fallback.path if fallback else "")),
        created_at=str(row.get("created_at") or (fallback.created_at if fallback else "")),
        url=str(row.get("url") or "") or (fallback.url if fallback else None),
        keyword_score=float(row.get("keyword_score")) if row.get("keyword_score") is not None else (
            (-float(row.get("rank_score"))) if row.get("rank_score") is not None else None
        ),
        fused_score=float(row.get("fused_score")) if row.get("fused_score") is not None else None,
        entity_score=float(row.get("entity_score")) if row.get("entity_score") is not None else (
            fallback.entity_score if fallback else None
        ),
        role_summary=str(row.get("role_summary") or (fallback.role_summary if fallback else "")),
        message_range=str(row.get("message_range") or (fallback.message_range if fallback else "")),
        model_name=str(row.get("model_name") or "") or (fallback.model_name if fallback else None),
        tags=tags or (fallback.tags if fallback else []),
        entity_names=[
            n for n in (
                row.get("entity_names")
                if isinstance(row.get("entity_names"), list)
                else [t for t in str(row.get("entity_names") or "").split(",") if t]
            )
        ] or (fallback.entity_names if fallback else []),
        turn_index=int(row.get("turn_index") or (fallback.turn_index if fallback else 0)),
        chunk_index=int(row.get("chunk_index") or (fallback.chunk_index if fallback else 0)),
    )


def _hit_to_row(hit: RetrievalHit) -> dict:
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "platform": hit.platform,
        "title": hit.title,
        "content": hit.excerpt,
        "source_path": hit.path,
        "created_at": hit.created_at,
        "url": hit.url or "",
        "role_summary": hit.role_summary,
        "message_range": hit.message_range,
        "model_name": hit.model_name or "",
        "tags": hit.tags,
        "turn_index": hit.turn_index,
        "chunk_index": hit.chunk_index,
        "score": hit.score,
        "rerank_score": hit.rerank_score,
        "keyword_score": hit.keyword_score,
        "fused_score": hit.fused_score,
        "entity_score": hit.entity_score,
        "entity_names": hit.entity_names,
    }

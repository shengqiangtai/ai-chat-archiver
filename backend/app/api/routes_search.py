"""语义检索 API。"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from app.models.schemas import KbSearchRequest
from app.services.qa.query_rewrite import rewrite_query
from app.services.vectorstore.retrieval import retrieve, retrieve_debug

router = APIRouter(prefix="/api/kb", tags=["knowledge-base-search"])


@router.post("/search")
async def api_kb_search(data: KbSearchRequest):
    """
    语义检索（不调用 LLM）。
    返回 top_k 个最相关的 chunk。
    """
    try:
        rewrite = await rewrite_query(data.query, enable_llm=data.rewrite_query)
        query_for_retrieval = rewrite.rewritten_query or data.query

        if data.include_debug:
            result = await asyncio.to_thread(
                retrieve_debug,
                query=query_for_retrieval,
                top_k=data.top_k,
                top_n=data.top_k,
                platform_filter=data.platform_filter,
                model_filter=data.model_filter,
                tag_filter=data.tag_filter,
                date_from=data.date_from,
                date_to=data.date_to,
                score_threshold=data.score_threshold,
                use_rerank=data.rerank_mode != "off",
                retrieval_mode=data.retrieval_mode,
                expand_neighbors=False,
                neighbor_turn_window=1,
                use_cache=True,
                rerank_mode=data.rerank_mode,
            )
            return {
                "query": data.query,
                "rewritten_query": rewrite.rewritten_query,
                "rewrite_applied": rewrite.applied,
                "rewrite_strategy": rewrite.strategy,
                "hits": result["hits"],
                "total": len(result["hits"]),
                "debug": {
                    **result["debug"],
                    "original_query": data.query,
                    "rewritten_query": rewrite.rewritten_query,
                    "rewrite_applied": rewrite.applied,
                    "rewrite_strategy": rewrite.strategy,
                },
            }

        hits = await asyncio.to_thread(
            retrieve,
            query=query_for_retrieval,
            top_k=data.top_k,
            top_n=data.top_k,
            platform_filter=data.platform_filter,
            model_filter=data.model_filter,
            tag_filter=data.tag_filter,
            date_from=data.date_from,
            date_to=data.date_to,
            score_threshold=data.score_threshold,
            use_rerank=data.rerank_mode != "off",
            retrieval_mode=data.retrieval_mode,
            expand_neighbors=False,
            neighbor_turn_window=1,
            use_cache=True,
            rerank_mode=data.rerank_mode,
        )
        return {
            "query": data.query,
            "rewritten_query": rewrite.rewritten_query,
            "rewrite_applied": rewrite.applied,
            "rewrite_strategy": rewrite.strategy,
            "hits": [asdict(h) for h in hits],
            "total": len(hits),
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err

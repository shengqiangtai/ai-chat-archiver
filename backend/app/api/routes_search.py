"""语义检索 API。"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from app.models.schemas import KbSearchRequest
from app.services.vectorstore.retrieval import retrieve

router = APIRouter(prefix="/api/kb", tags=["knowledge-base-search"])


@router.post("/search")
def api_kb_search(data: KbSearchRequest):
    """
    语义检索（不调用 LLM）。
    返回 top_k 个最相关的 chunk。
    """
    try:
        hits = retrieve(
            query=data.query,
            top_k=data.top_k,
            platform_filter=data.platform_filter,
            score_threshold=data.score_threshold,
        )
        return {
            "query": data.query,
            "hits": [asdict(h) for h in hits],
            "total": len(hits),
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err

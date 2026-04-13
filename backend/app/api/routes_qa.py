"""知识库问答 API。"""

from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import QARequest
from app.services.qa.pipeline import qa_answer, qa_answer_stream

router = APIRouter(prefix="/api/kb", tags=["knowledge-base-qa"])


def _sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/qa")
async def api_kb_qa(data: QARequest):
    """非流式问答。"""
    try:
        result = await qa_answer(
            query=data.query,
            mode=data.mode,
            top_k=data.top_k,
            top_n=data.top_n,
            include_debug=True,
        )
        return {
            "answer": result.answer,
            "citations": [asdict(c) for c in result.citations],
            "uncertainty": result.uncertainty,
            "sources": [asdict(s) for s in result.sources],
            "debug": result.debug,
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/qa/stream")
async def api_kb_qa_stream(data: QARequest):
    """流式问答（SSE）。"""
    async def event_stream():
        try:
            async for piece in qa_answer_stream(
                query=data.query,
                mode=data.mode,
                top_k=data.top_k,
                top_n=data.top_n,
            ):
                if "[SOURCES_JSON]" in piece:
                    _, sources_json = piece.split("[SOURCES_JSON]", 1)
                    try:
                        sources = json.loads(sources_json)
                    except json.JSONDecodeError:
                        sources = []
                    yield _sse_data({"type": "sources", "sources": sources})
                else:
                    yield _sse_data({"type": "token", "content": piece})

            yield _sse_data({"type": "done"})
        except Exception as err:
            yield _sse_data({"type": "error", "message": str(err)})
            yield _sse_data({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

"""OpenAI-compatible public API routes."""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.schemas import OpenAIChatCompletionRequest, OpenAIChatMessage
from app.services.qa.pipeline import qa_answer, qa_answer_stream

DEFAULT_MODEL_ID = "ai-chat-archiver-rag"
SOURCES_MARKER = "[SOURCES_JSON]"

router = APIRouter(tags=["openai-compatible"])


def _error(
    message: str,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


def _content_to_text(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _extract_query(messages: list[OpenAIChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = _content_to_text(message.content).strip()
        if text:
            return text
    return None


def _extract_instruction_context(messages: list[OpenAIChatMessage]) -> str | None:
    parts: list[str] = []
    for message in messages:
        if message.role not in {"system", "developer"}:
            continue
        text = _content_to_text(message.content).strip()
        if text:
            parts.append(f"{message.role}: {text}")
    if not parts:
        return None
    return "\n\n".join(parts)


def _usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None = None,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _sse_data(payload: dict | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "ai-chat-archiver",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def create_chat_completion(data: OpenAIChatCompletionRequest):
    query = _extract_query(data.messages)
    if not query:
        return _error("A non-empty user message is required.")
    instruction_context = _extract_instruction_context(data.messages)

    model = data.model or DEFAULT_MODEL_ID
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if data.stream:
        return StreamingResponse(
            _stream_chat_completion(
                query,
                instruction_context,
                completion_id,
                created,
                model,
            ),
            media_type="text/event-stream",
        )

    try:
        result = await qa_answer(
            query=query,
            instruction_context=instruction_context,
            rewrite_query_enabled=False,
            raise_generation_errors=True,
        )
    except Exception as err:
        return _error(str(err), status_code=500, error_type="server_error")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(),
    }


async def _stream_chat_completion(
    query: str,
    instruction_context: str | None,
    completion_id: str,
    created: int,
    model: str,
) -> AsyncIterator[str]:
    try:
        yield _sse_data(_chunk(completion_id, created, model, {"role": "assistant"}))
        async for piece in qa_answer_stream(
            query=query,
            instruction_context=instruction_context,
            rewrite_query_enabled=False,
            raise_generation_errors=True,
        ):
            if SOURCES_MARKER in piece:
                piece, _ = piece.split(SOURCES_MARKER, 1)
                if not piece.strip():
                    continue
            yield _sse_data(_chunk(completion_id, created, model, {"content": piece}))
        yield _sse_data(_chunk(completion_id, created, model, {}, finish_reason="stop"))
        yield _sse_data("[DONE]")
    except Exception as err:
        yield _sse_data(
            {
                "error": {
                    "message": str(err),
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            }
        )
        yield _sse_data("[DONE]")

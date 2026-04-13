"""文档管理 + LLM 后端管理 + 聊天归档 API。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import (
    LMSTUDIO_BASE_URL,
    OLLAMA_BASE_URL,
    get_current_lmstudio_model,
    get_current_ollama_model,
    get_generator_backend,
    set_current_lmstudio_model,
    set_current_ollama_model,
    set_generator_backend,
)
from app.db.sqlite import get_db
from app.models.schemas import OllamaModelUpdateRequest, SaveRequest, SearchRequest
from app.services.llm.generator import LMStudioGenerator, OllamaGenerator

router = APIRouter(tags=["documents"])


# ═══════════════════════════════════════════════════════════════════════════
# 聊天记录 CRUD（兼容旧 API）
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/save")
def api_save_chat(data: SaveRequest):
    try:
        db = get_db()
        return db.save_chat(data)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/chats")
def api_get_chats(platform: Optional[str] = None, limit: int = 50, offset: int = 0):
    try:
        db = get_db()
        chats = db.get_chat_list(platform=platform, limit=limit, offset=offset)
        return {"chats": chats, "count": len(chats)}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/chats/{chat_id}")
def api_get_chat(chat_id: str):
    db = get_db()
    chat = db.get_chat_by_id(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.delete("/chats/{chat_id}")
def api_delete_chat(chat_id: str):
    db = get_db()
    ok = db.delete_chat(chat_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"ok": True, "id": chat_id}


@router.post("/search")
def api_search(data: SearchRequest):
    try:
        db = get_db()
        results = db.search_chats(query=data.query, platform=data.platform, limit=data.limit)
        return {"results": results, "count": len(results)}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/stats")
def api_stats():
    try:
        db = get_db()
        return db.get_stats()
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "AI Chat Archiver", "version": "3.0.0"}


# ═══════════════════════════════════════════════════════════════════════════
# 知识库文档管理
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/api/kb/documents")
def api_kb_documents(platform: Optional[str] = None, limit: int = 50, offset: int = 0):
    """分页返回已索引文档。"""
    db = get_db()
    chats = db.get_chat_list(platform=platform, limit=limit, offset=offset)
    return {"documents": chats, "count": len(chats)}


@router.get("/api/kb/documents/{doc_id}")
def api_kb_document_detail(doc_id: str):
    """返回文档详情。"""
    db = get_db()
    chat = db.get_chat_by_id(doc_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Document not found")
    return chat


# ═══════════════════════════════════════════════════════════════════════════
# LLM 后端管理（LM Studio + Ollama + transformers）
# ═══════════════════════════════════════════════════════════════════════════

class BackendSwitchRequest(BaseModel):
    backend: str  # "lmstudio" | "ollama" | "transformers"
    model: str | None = None


@router.get("/api/kb/llm/status")
async def api_llm_status():
    """返回所有 LLM 后端的状态。"""
    current_backend = get_generator_backend()

    # LM Studio
    lm = LMStudioGenerator(base_url=LMSTUDIO_BASE_URL)
    lm_available = await lm.is_available()
    lm_models: list[str] = []
    if lm_available:
        try:
            lm_models = await lm.list_models()
        except Exception:
            pass

    # Ollama
    ollama = OllamaGenerator(base_url=OLLAMA_BASE_URL)
    ollama_available = await ollama.is_available()
    ollama_models: list[str] = []
    if ollama_available:
        try:
            ollama_models = await ollama.list_models()
        except Exception:
            pass

    return {
        "current_backend": current_backend,
        "lmstudio": {
            "available": lm_available,
            "models": lm_models,
            "current_model": get_current_lmstudio_model(),
            "base_url": LMSTUDIO_BASE_URL,
        },
        "ollama": {
            "available": ollama_available,
            "models": ollama_models,
            "current_model": get_current_ollama_model(),
            "base_url": OLLAMA_BASE_URL,
        },
        "transformers": {
            "available": True,
            "current_model": "Qwen3.5-0.8B (本地)",
        },
    }


@router.put("/api/kb/llm/backend")
def api_switch_backend(data: BackendSwitchRequest):
    """切换 LLM 后端。"""
    backend = (data.backend or "").strip().lower()
    if backend not in ("lmstudio", "ollama", "transformers"):
        raise HTTPException(status_code=400, detail="backend 必须为 lmstudio / ollama / transformers")

    set_generator_backend(backend)

    if data.model:
        if backend == "lmstudio":
            set_current_lmstudio_model(data.model)
        elif backend == "ollama":
            set_current_ollama_model(data.model)

    return {"ok": True, "current_backend": backend, "model": data.model}


# 兼容旧 API
@router.get("/api/kb/ollama/status")
async def api_ollama_status():
    current_model = get_current_ollama_model()
    client = OllamaGenerator(base_url=OLLAMA_BASE_URL, model=current_model)
    available = await client.is_available()
    models: list[str] = []
    if available:
        try:
            models = await client.list_models()
        except Exception:
            pass
    return {
        "available": available,
        "models": models,
        "current_model": current_model,
        "base_url": OLLAMA_BASE_URL,
    }


@router.put("/api/kb/ollama/model")
def api_ollama_model(data: OllamaModelUpdateRequest):
    model = (data.model or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model 不能为空")
    set_current_ollama_model(model)
    return {"ok": True, "current_model": model, "base_url": OLLAMA_BASE_URL}

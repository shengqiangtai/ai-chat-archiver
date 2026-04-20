"""知识库索引管理 API。"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.core.config import STORAGE_ROOT, load_runtime_config, set_last_index_time
from app.services.qa.pipeline import delete_doc_index, incremental_index, rebuild_index
from app.services.vectorstore.chroma_store import get_store
from app.db.sqlite import get_db

router = APIRouter(prefix="/api/kb", tags=["knowledge-base-index"])

index_tasks: dict[str, dict] = {}


def _new_task(task_id: str) -> None:
    index_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "total_files": 0,
        "processed_files": 0,
        "total_chunks": 0,
        "elapsed_seconds": 0.0,
        "error": None,
        "started_at": time.monotonic(),
        "current_file": None,
    }


def _progress_updater(task_id: str):
    def _update(processed: int, total: int, chunks: int, current_file: str) -> None:
        task = index_tasks.get(task_id)
        if not task:
            return
        task["processed_files"] = processed
        task["total_files"] = total
        task["total_chunks"] = chunks
        task["current_file"] = current_file
        task["elapsed_seconds"] = round(time.monotonic() - task["started_at"], 2)
    return _update


async def _run_index_task(task_id: str, is_incremental: bool) -> None:
    task = index_tasks[task_id]
    task["status"] = "running"
    try:
        progress_cb = _progress_updater(task_id)
        if is_incremental:
            runtime = load_runtime_config()
            result = await asyncio.to_thread(
                incremental_index, runtime.get("last_index_time"), progress_cb
            )
        else:
            result = await asyncio.to_thread(rebuild_index, progress_cb)

        task["status"] = "done"
        task["total_files"] = result["total_files"]
        task["processed_files"] = result["processed_files"]
        task["total_chunks"] = result["total_chunks"]
        task["elapsed_seconds"] = round(time.monotonic() - task["started_at"], 2)
    except Exception as err:
        task["status"] = "error"
        task["error"] = str(err)
        task["elapsed_seconds"] = round(time.monotonic() - task["started_at"], 2)


@router.post("/reindex")
async def api_reindex():
    """全量重建索引。"""
    task_id = str(uuid.uuid4())
    _new_task(task_id)
    asyncio.create_task(_run_index_task(task_id, is_incremental=False))
    return {
        "task_id": task_id,
        "status": "started",
        "message": "全量索引已开始，可通过 GET /api/kb/reindex/progress/{task_id} 查询进度",
    }


@router.post("/reindex/incremental")
async def api_reindex_incremental():
    """增量索引。"""
    task_id = str(uuid.uuid4())
    _new_task(task_id)
    asyncio.create_task(_run_index_task(task_id, is_incremental=True))
    return {
        "task_id": task_id,
        "status": "started",
        "message": "增量索引已开始",
    }


@router.get("/reindex/progress/{task_id}")
def api_reindex_progress(task_id: str):
    """查询索引任务进度。"""
    task = index_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task["task_id"],
        "status": task["status"],
        "total_files": task["total_files"],
        "processed_files": task["processed_files"],
        "total_chunks": task["total_chunks"],
        "elapsed_seconds": task["elapsed_seconds"],
        "error": task["error"],
    }


@router.get("/status")
def api_kb_status():
    """返回知识库状态。"""
    try:
        store = get_store()
        chunk_stats = store.get_stats()
        runtime = load_runtime_config()
        db = get_db()
        chat_stats = db.get_stats()
        entity_stats = db.get_entity_stats()

        chroma_size = 0
        from app.core.config import CHROMA_PATH
        if CHROMA_PATH.exists():
            for path in CHROMA_PATH.rglob("*"):
                if path.is_file():
                    chroma_size += path.stat().st_size

        is_indexing = any(
            t.get("status") in {"pending", "running"} for t in index_tasks.values()
        )

        return {
            "total_chats": chat_stats.get("total", 0),
            "total_chunks": chunk_stats.get("total_chunks", 0),
            "total_entities": entity_stats.get("total_entities", 0),
            "top_entities": entity_stats.get("top_entities", []),
            "by_platform": chunk_stats.get("by_platform", {}),
            "vectorstore_size_bytes": chroma_size,
            "last_index_time": runtime.get("last_index_time"),
            "is_indexing": is_indexing,
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.delete("/documents/{doc_id}/index")
def api_delete_doc_index(doc_id: str):
    """删除某个文档的索引。"""
    try:
        deleted = delete_doc_index(doc_id)
        return {"ok": True, "deleted": deleted, "doc_id": doc_id}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err

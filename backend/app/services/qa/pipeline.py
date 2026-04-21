"""QA Pipeline — 编排检索→rerank→上下文→生成→引用验证的完整流程。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

from app.core.config import (
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
    STORAGE_ROOT,
    UNLOAD_GENERATOR_AFTER_INFERENCE,
    set_last_index_time,
)
from app.db.sqlite import get_db
from app.models.schemas import AnswerResult, Citation, SourceRef
from app.services.cache.query_cache import get_cache
from app.services.embedding.embedder import get_embedder
from app.services.ingest.chunker import chunk_document
from app.services.ingest.deduper import (
    clear_file_index,
    deduplicate_chunks,
    get_skip_reason,
    register_chunks,
    register_file,
    should_skip_file,
)
from app.services.ingest.entity_extractor import (
    extract_entities_from_chunks,
    extract_graph_relations_from_chunks,
)
from app.services.ingest.loader import load_document, scan_chat_files
from app.services.llm.generator import get_generator, unload_generator
from app.services.llm.prompt_builder import (
    build_fallback_answer,
    build_user_prompt,
    get_system_prompt,
)
from app.services.qa.citation import parse_llm_output
from app.services.qa.query_rewrite import rewrite_query
from app.services.vectorstore.chroma_store import get_store
from app.services.vectorstore.retrieval import retrieve

logger = logging.getLogger("archiver.qa.pipeline")

ProgressCallback = Callable[[int, int, int, str, Optional[dict]], None]


def _persist_graph_metadata(db, chunks, created_at: str) -> int:
    relations = extract_graph_relations_from_chunks(chunks)
    if not relations:
        return 0
    return int(db.upsert_graph_relations(relations, created_at=created_at) or 0)


# ═══════════════════════════════════════════════════════════════════════════
# 问答流程
# ═══════════════════════════════════════════════════════════════════════════

async def qa_answer(
    query: str,
    mode: str = "concise",
    top_k: int = RETRIEVAL_TOP_K,
    top_n: int = RERANK_TOP_N,
    platform_filter: str | None = None,
    model_filter: str | None = None,
    tag_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    retrieval_mode: str = "mix",
    rerank_mode: str = "auto",
    rewrite_query_enabled: bool = True,
    include_debug: bool = False,
) -> AnswerResult:
    """
    完整 RAG 问答流程（非流式）。
    """
    t0 = time.time()

    rewrite = await rewrite_query(query, enable_llm=rewrite_query_enabled)
    retrieval_query = rewrite.rewritten_query or query

    hits = await asyncio.to_thread(
        retrieve,
        query=retrieval_query,
        top_k=top_k,
        top_n=top_n,
        platform_filter=platform_filter,
        model_filter=model_filter,
        tag_filter=tag_filter,
        date_from=date_from,
        date_to=date_to,
        score_threshold=0.0,
        use_rerank=rerank_mode != "off",
        retrieval_mode=retrieval_mode,
        expand_neighbors=True,
        neighbor_turn_window=1,
        use_cache=True,
        rerank_mode=rerank_mode,
    )
    t_retrieve = time.time() - t0

    if not hits:
        return AnswerResult(
            answer="知识库中未找到相关记录。",
            citations=[],
            uncertainty=None,
            sources=[],
            debug={"total_time": time.time() - t0} if include_debug else {},
        )

    # 检索失败兜底：所有分数都太低
    best_score = max(
        (h.rerank_score if h.rerank_score is not None else h.score) for h in hits
    )
    if best_score < 0.25:
        fallback = build_fallback_answer(hits)
        result = parse_llm_output(fallback, hits)
        result.uncertainty = "未找到足够相关内容，以下为最相关的候选片段"
        return result

    source_chunk_ids = [hit.chunk_id for hit in hits]
    cached_answer = get_cache().get_answer(query, source_chunk_ids, mode=mode)
    if cached_answer:
        result = AnswerResult(
            answer=str(cached_answer.get("answer") or ""),
            citations=[Citation(**c) for c in (cached_answer.get("citations") or [])],
            uncertainty=cached_answer.get("uncertainty"),
            sources=[
                SourceRef(
                    chunk_id=str(s.get("chunk_id") or ""),
                    source_id=str(s.get("source_id") or ""),
                    platform=str(s.get("platform") or "Unknown"),
                    title=str(s.get("title") or "Untitled"),
                    path=str(s.get("path") or ""),
                    score=float(s.get("score") or 0.0),
                    rerank_score=s.get("rerank_score"),
                    url=s.get("url"),
                    excerpt=str(s.get("excerpt") or ""),
                    message_range=str(s.get("message_range") or ""),
                    turn_index=int(s.get("turn_index") or 0),
                )
                for s in (cached_answer.get("sources") or [])
            ],
            debug=cached_answer.get("debug") or {},
        )
        if include_debug:
            result.debug = {
                **result.debug,
                "original_query": query,
                "rewritten_query": rewrite.rewritten_query,
                "rewrite_applied": rewrite.applied,
                "rewrite_strategy": rewrite.strategy,
                "retrieved_count": len(hits),
                "retrieve_time": round(t_retrieve, 3),
                "total_time": round(time.time() - t0, 3),
                "answer_cache_hit": True,
                "retrieved": [asdict(h) for h in hits],
            }
        return result

    user_prompt = build_user_prompt(query, hits, mode)
    system_prompt = get_system_prompt()
    t_prompt = time.time() - t0 - t_retrieve

    try:
        generator = get_generator()
        raw_answer = await generator.generate(user_prompt, mode=mode, system_prompt=system_prompt)
        t_generate = time.time() - t0 - t_retrieve - t_prompt
    except Exception as e:
        logger.warning("生成失败，使用降级回答: %s", e)
        raw_answer = build_fallback_answer(hits)
        t_generate = time.time() - t0 - t_retrieve - t_prompt
    finally:
        # 低内存模式：生成完毕立即卸载模型，释放约 1.8 GB
        if UNLOAD_GENERATOR_AFTER_INFERENCE:
            unload_generator()

    result = parse_llm_output(raw_answer, hits)

    if include_debug:
        result.debug = {
            "original_query": query,
            "rewritten_query": rewrite.rewritten_query,
            "rewrite_applied": rewrite.applied,
            "rewrite_strategy": rewrite.strategy,
            "retrieved_count": len(hits),
            "retrieve_time": round(t_retrieve, 3),
            "prompt_time": round(t_prompt, 3),
            "generate_time": round(t_generate, 3),
            "total_time": round(time.time() - t0, 3),
            "low_memory_mode": UNLOAD_GENERATOR_AFTER_INFERENCE,
            "retrieval_mode": retrieval_mode,
            "rerank_mode": rerank_mode,
            "retrieved": [asdict(h) for h in hits],
        }

    get_cache().set_answer(
        query,
        source_chunk_ids,
        {
            "answer": result.answer,
            "citations": [asdict(c) for c in result.citations],
            "uncertainty": result.uncertainty,
            "sources": [asdict(s) for s in result.sources],
            "debug": result.debug,
        },
        mode=mode,
    )

    total = time.time() - t0
    logger.info("QA 完成: query=%r, hits=%d, time=%.2fs", query[:50], len(hits), total)
    return result


async def qa_answer_stream(
    query: str,
    mode: str = "concise",
    top_k: int = RETRIEVAL_TOP_K,
    top_n: int = RERANK_TOP_N,
    platform_filter: str | None = None,
    model_filter: str | None = None,
    tag_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    retrieval_mode: str = "mix",
    rerank_mode: str = "auto",
    rewrite_query_enabled: bool = True,
) -> AsyncGenerator[str, None]:
    """
    流式 QA 问答。
    最后 yield 一个 [SOURCES_JSON] 标记携带来源信息。
    """
    rewrite = await rewrite_query(query, enable_llm=rewrite_query_enabled)
    retrieval_query = rewrite.rewritten_query or query

    hits = await asyncio.to_thread(
        retrieve,
        query=retrieval_query,
        top_k=top_k,
        top_n=top_n,
        platform_filter=platform_filter,
        model_filter=model_filter,
        tag_filter=tag_filter,
        date_from=date_from,
        date_to=date_to,
        score_threshold=0.0,
        use_rerank=rerank_mode != "off",
        retrieval_mode=retrieval_mode,
        expand_neighbors=True,
        neighbor_turn_window=1,
        use_cache=True,
        rerank_mode=rerank_mode,
    )

    if not hits:
        yield "知识库中未找到相关记录。"
        yield "\n\n[SOURCES_JSON]" + json.dumps([], ensure_ascii=False)
        return

    from app.services.qa.citation import _build_sources
    sources = _build_sources(hits)
    sources_data = [asdict(s) for s in sources]

    best_score = max(
        (h.rerank_score if h.rerank_score is not None else h.score) for h in hits
    )
    if best_score < 0.25:
        yield build_fallback_answer(hits)
        yield "\n\n[SOURCES_JSON]" + json.dumps(sources_data, ensure_ascii=False)
        return

    user_prompt = build_user_prompt(query, hits, mode)
    system_prompt = get_system_prompt()

    try:
        generator = get_generator()
        async for token in generator.generate_stream(user_prompt, mode=mode, system_prompt=system_prompt):
            yield token
    except Exception as e:
        logger.warning("流式生成失败: %s", e)
        yield build_fallback_answer(hits)
    finally:
        # 低内存模式：流式生成完毕后卸载模型
        if UNLOAD_GENERATOR_AFTER_INFERENCE:
            unload_generator()

    yield "\n\n[SOURCES_JSON]" + json.dumps(sources_data, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# 索引管理
# ═══════════════════════════════════════════════════════════════════════════

def rebuild_index(progress_cb: ProgressCallback | None = None) -> dict:
    """全量重建索引。"""
    store = get_store()
    store.reset()
    db = get_db()
    db.clear_kb_chunks()
    get_cache().clear_all()
    embedder = get_embedder()

    files = scan_chat_files(STORAGE_ROOT)
    total_files = len(files)
    processed = 0
    total_chunks = 0
    skipped_chunks = 0

    logger.info("开始全量索引: %d 个文件", total_files)

    for md_path, meta_path in files:
        try:
            doc = load_document(md_path, meta_path)
            chunks = chunk_document(doc)

            if chunks:
                texts = [c.text for c in chunks]
                embeddings = embedder.encode_docs(texts)
                written = store.upsert_chunks(chunks, embeddings)
                total_chunks += written
                db.upsert_kb_chunks(chunks)
                db.upsert_entity_mentions(
                    extract_entities_from_chunks(chunks),
                    created_at=doc.created_at,
                )
                _persist_graph_metadata(db, chunks, doc.created_at)
                register_chunks(chunks)
                register_file(doc, len(chunks))
        except Exception as e:
            logger.error("索引文件失败: %s — %s", md_path, e)

        processed += 1
        if progress_cb:
            progress_cb(
                processed,
                total_files,
                total_chunks,
                str(md_path),
                {
                    "scanned_files": total_files,
                    "skipped_files": 0,
                    "skip_reasons": {},
                },
            )

    set_last_index_time(datetime.now().isoformat(timespec="seconds"))
    logger.info("全量索引完成: files=%d, chunks=%d", total_files, total_chunks)

    # 索引完成后触发 GC，释放索引期间积累的中间对象
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    return {
        "total_files": total_files,
        "processed_files": processed,
        "total_chunks": total_chunks,
    }


def incremental_index(
    last_index_time: str | None,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    """增量索引 — 只处理新增或修改的文件。"""
    files = scan_chat_files(STORAGE_ROOT)
    store = get_store()
    db = get_db()
    get_cache().clear_all()
    embedder = get_embedder()

    selected: list[tuple[Path, Path]] = []
    skipped_files = 0
    skip_reasons: dict[str, int] = {}
    for md_path, meta_path in files:
        doc = load_document(md_path, meta_path)
        reason = get_skip_reason(doc)
        if reason:
            skipped_files += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        selected.append((md_path, meta_path))

    total_files = len(selected)
    processed = 0
    total_chunks = 0

    logger.info(
        "开始增量索引: %d / %d 个文件需要更新，跳过 %d 个文件",
        total_files,
        len(files),
        skipped_files,
    )

    if progress_cb:
        progress_cb(
            0,
            total_files,
            total_chunks,
            "",
            {
                "scanned_files": len(files),
                "skipped_files": skipped_files,
                "skip_reasons": skip_reasons,
            },
        )

    for md_path, meta_path in selected:
        try:
            doc = load_document(md_path, meta_path)
            clear_file_index(doc.doc_id, doc.path)
            store.delete_by_doc_id(doc.doc_id)
            db.delete_kb_chunks_by_doc(doc.doc_id)

            chunks = chunk_document(doc)
            if chunks:
                unique_chunks = deduplicate_chunks(chunks, doc.doc_id)
                if unique_chunks:
                    texts = [c.text for c in unique_chunks]
                    embeddings = embedder.encode_docs(texts)
                    written = store.upsert_chunks(unique_chunks, embeddings)
                    total_chunks += written
                    db.upsert_kb_chunks(unique_chunks)
                    db.upsert_entity_mentions(
                        extract_entities_from_chunks(unique_chunks),
                        created_at=doc.created_at,
                    )
                    _persist_graph_metadata(db, unique_chunks, doc.created_at)
                    register_chunks(unique_chunks)
                register_file(doc, len(chunks))
        except Exception as e:
            logger.error("增量索引文件失败: %s — %s", md_path, e)

        processed += 1
        if progress_cb:
            progress_cb(
                processed,
                total_files,
                total_chunks,
                str(md_path),
                {
                    "scanned_files": len(files),
                    "skipped_files": skipped_files,
                    "skip_reasons": skip_reasons,
                },
            )

    set_last_index_time(datetime.now().isoformat(timespec="seconds"))
    logger.info("增量索引完成: files=%d, chunks=%d", total_files, total_chunks)

    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    return {
        "scanned_files": len(files),
        "total_files": total_files,
        "processed_files": processed,
        "skipped_files": skipped_files,
        "skip_reasons": skip_reasons,
        "total_chunks": total_chunks,
    }


def delete_doc_index(doc_id: str) -> int:
    """删除某个文档的索引。"""
    store = get_store()
    deleted = store.delete_by_doc_id(doc_id)
    get_db().delete_kb_chunks_by_doc(doc_id)
    get_cache().clear_all()
    return deleted


def cleanup_index_integrity() -> dict:
    """清理失效的聊天元数据与索引残留，不触碰仍存在的归档目录。"""
    db = get_db()
    store = get_store()

    orphan_chats = db.list_orphan_chats()
    stale_file_records = db.list_stale_file_records()
    chat_ids = db.list_chat_ids()
    kb_docs = db.list_kb_doc_sources()
    chunk_hash_docs = db.list_chunk_hash_docs()

    orphan_doc_ids: set[str] = set()
    for item in orphan_chats:
        orphan_doc_ids.add(str(item["id"]))
    for item in stale_file_records:
        doc_id = str(item.get("doc_id") or "")
        if doc_id:
            orphan_doc_ids.add(doc_id)
    for item in kb_docs:
        doc_id = str(item.get("doc_id") or "")
        source_path = str(item.get("source_path") or "")
        if doc_id and (doc_id not in chat_ids or (source_path and not Path(source_path).exists())):
            orphan_doc_ids.add(doc_id)
    for doc_id in chunk_hash_docs:
        if doc_id not in chat_ids:
            orphan_doc_ids.add(doc_id)

    removed_chats = 0
    for item in orphan_chats:
        if db.purge_chat_metadata(str(item["id"])):
            removed_chats += 1

    removed_file_records = 0
    processed_doc_ids: set[str] = set()
    for item in stale_file_records:
        file_path = str(item.get("file_path") or "")
        doc_id = str(item.get("doc_id") or "")
        if file_path:
            db.delete_file_record(file_path)
            removed_file_records += 1
        if doc_id:
            processed_doc_ids.add(doc_id)

    removed_chunks = 0
    removed_vector_docs = 0
    removed_chunk_hash_docs = 0
    for doc_id in sorted(orphan_doc_ids):
        removed_chunks += db.delete_kb_chunks_by_doc(doc_id)
        removed_vector_docs += store.delete_by_doc_id(doc_id)
        db.delete_chunk_hashes_by_doc(doc_id)
        removed_chunk_hash_docs += 1
        if doc_id not in processed_doc_ids:
            removed_file_records += db.delete_file_records_by_doc(doc_id)

    get_cache().clear_all()

    return {
        "orphan_chat_count": len(orphan_chats),
        "stale_file_record_count": len(stale_file_records),
        "orphan_doc_count": len(orphan_doc_ids),
        "removed_chats": removed_chats,
        "removed_file_records": removed_file_records,
        "removed_vector_docs": removed_vector_docs,
        "removed_chunks": removed_chunks,
        "removed_chunk_hash_docs": removed_chunk_hash_docs,
        "sample_orphan_chat_ids": [str(item["id"]) for item in orphan_chats[:5]],
        "sample_orphan_doc_ids": sorted(orphan_doc_ids)[:5],
    }

"""QA Pipeline — 编排检索→rerank→上下文→生成→引用验证的完整流程。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional

from app.core.config import (
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
    STORAGE_ROOT,
    UNLOAD_GENERATOR_AFTER_INFERENCE,
    set_last_index_time,
)
from app.models.schemas import AnswerResult, Chunk, RetrievalHit
from app.services.embedding.embedder import get_embedder
from app.services.ingest.chunker import chunk_document
from app.services.ingest.deduper import (
    clear_file_index,
    deduplicate_chunks,
    register_chunks,
    register_file,
    should_skip_file,
)
from app.services.ingest.loader import load_document, scan_chat_files
from app.services.llm.generator import get_generator, unload_generator
from app.services.llm.prompt_builder import (
    build_fallback_answer,
    build_qa_prompt,
    build_user_prompt,
    get_system_prompt,
)
from app.services.qa.citation import parse_llm_output
from app.services.vectorstore.chroma_store import get_store
from app.services.vectorstore.retrieval import retrieve

logger = logging.getLogger("archiver.qa.pipeline")

ProgressCallback = Callable[[int, int, int, str], None]


# ═══════════════════════════════════════════════════════════════════════════
# 问答流程
# ═══════════════════════════════════════════════════════════════════════════

async def qa_answer(
    query: str,
    mode: str = "concise",
    top_k: int = RETRIEVAL_TOP_K,
    top_n: int = RERANK_TOP_N,
    include_debug: bool = False,
) -> AnswerResult:
    """
    完整 RAG 问答流程（非流式）。
    """
    t0 = time.time()

    hits = await asyncio.to_thread(retrieve, query, top_k, top_n)
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

    user_prompt = build_user_prompt(query, hits, mode)
    system_prompt = get_system_prompt()
    t_prompt = time.time() - t0 - t_retrieve

    try:
        generator = get_generator()
        raw_answer = await generator.generate(user_prompt, mode, system_prompt=system_prompt)
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
            "retrieved_count": len(hits),
            "retrieve_time": round(t_retrieve, 3),
            "prompt_time": round(t_prompt, 3),
            "generate_time": round(t_generate, 3),
            "total_time": round(time.time() - t0, 3),
            "low_memory_mode": UNLOAD_GENERATOR_AFTER_INFERENCE,
            "retrieved": [asdict(h) for h in hits],
        }

    total = time.time() - t0
    logger.info("QA 完成: query=%r, hits=%d, time=%.2fs", query[:50], len(hits), total)
    return result


async def qa_answer_stream(
    query: str,
    mode: str = "concise",
    top_k: int = RETRIEVAL_TOP_K,
    top_n: int = RERANK_TOP_N,
) -> AsyncGenerator[str, None]:
    """
    流式 QA 问答。
    最后 yield 一个 [SOURCES_JSON] 标记携带来源信息。
    """
    hits = await asyncio.to_thread(retrieve, query, top_k, top_n)

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
        async for token in generator.generate_stream(user_prompt, mode, system_prompt=system_prompt):
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
                register_chunks(chunks)
                register_file(doc, len(chunks))
        except Exception as e:
            logger.error("索引文件失败: %s — %s", md_path, e)

        processed += 1
        if progress_cb:
            progress_cb(processed, total_files, total_chunks, str(md_path))

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
    embedder = get_embedder()

    selected: list[tuple[Path, Path]] = []
    for md_path, meta_path in files:
        doc = load_document(md_path, meta_path)
        if not should_skip_file(doc):
            selected.append((md_path, meta_path))

    total_files = len(selected)
    processed = 0
    total_chunks = 0

    logger.info("开始增量索引: %d / %d 个文件需要更新", total_files, len(files))

    for md_path, meta_path in selected:
        try:
            doc = load_document(md_path, meta_path)
            clear_file_index(doc.doc_id, doc.path)
            store.delete_by_doc_id(doc.doc_id)

            chunks = chunk_document(doc)
            if chunks:
                unique_chunks = deduplicate_chunks(chunks, doc.doc_id)
                if unique_chunks:
                    texts = [c.text for c in unique_chunks]
                    embeddings = embedder.encode_docs(texts)
                    written = store.upsert_chunks(unique_chunks, embeddings)
                    total_chunks += written
                    register_chunks(unique_chunks)
                register_file(doc, len(chunks))
        except Exception as e:
            logger.error("增量索引文件失败: %s — %s", md_path, e)

        processed += 1
        if progress_cb:
            progress_cb(processed, total_files, total_chunks, str(md_path))

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
        "total_files": total_files,
        "processed_files": processed,
        "total_chunks": total_chunks,
    }


def delete_doc_index(doc_id: str) -> int:
    """删除某个文档的索引。"""
    store = get_store()
    return store.delete_by_doc_id(doc_id)

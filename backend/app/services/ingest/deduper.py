"""去重器 — 文件级去重 + chunk 级去重。"""

from __future__ import annotations

import logging

from app.db.sqlite import get_db
from app.models.schemas import Chunk, Document
from app.utils.hashing import file_hash

logger = logging.getLogger("archiver.ingest.deduper")


def should_skip_file(doc: Document) -> bool:
    """
    文件级去重：基于文件路径 + hash 判断是否需要跳过。
    如果文件内容未变化则跳过。
    """
    db = get_db()
    record = db.get_file_record(doc.path)
    if record is None:
        return False

    if record["file_hash"] == doc.file_hash:
        logger.debug("文件未变化，跳过: %s", doc.path)
        return True

    return False


def deduplicate_chunks(chunks: list[Chunk], doc_id: str) -> list[Chunk]:
    """
    chunk 级去重：基于 text_hash 判断是否已存在。
    返回去重后的 chunk 列表。
    """
    db = get_db()
    unique: list[Chunk] = []
    skipped = 0

    for chunk in chunks:
        if db.has_chunk_hash(chunk.text_hash):
            skipped += 1
            continue
        unique.append(chunk)

    if skipped > 0:
        logger.info("文档 %s: 去重跳过 %d 个重复 chunk", doc_id, skipped)

    return unique


def register_chunks(chunks: list[Chunk]) -> None:
    """将 chunk 的 hash 注册到去重表。"""
    db = get_db()
    for chunk in chunks:
        db.add_chunk_hash(chunk.text_hash, chunk.chunk_id, chunk.doc_id)


def register_file(doc: Document, chunk_count: int) -> None:
    """记录文件索引状态。"""
    db = get_db()
    db.upsert_file_record(
        file_path=doc.path,
        file_hash=doc.file_hash,
        modified_time=doc.modified_time,
        doc_id=doc.doc_id,
        chunk_count=chunk_count,
    )


def clear_file_index(doc_id: str, file_path: str) -> None:
    """清除文件的索引记录和 chunk hash 记录（用于重建时）。"""
    db = get_db()
    db.delete_file_record(file_path)
    db.delete_chunk_hashes_by_doc(doc_id)

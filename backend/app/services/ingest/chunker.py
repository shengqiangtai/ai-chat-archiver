"""切块器 — 实现两种切块模式：dialogue_chunk 和 semantic_window_chunk。

适配聊天语料特点：
- 优先按"问答轮次"切块
- 短轮次合并
- 长轮次用滑动窗口再切分
- 每个 chunk 保留完整元数据
"""

from __future__ import annotations

import logging

from app.core.config import CHUNK_MAX_SIZE, CHUNK_MIN_MERGE, CHUNK_OVERLAP, CHUNK_TARGET_SIZE
from app.models.schemas import Chunk, Document, ParsedMessage
from app.services.ingest.normalizer import build_chunk_prefix, normalize_chunk_text
from app.services.ingest.parser import group_into_turns, parse_chat_markdown
from app.utils.hashing import text_hash

logger = logging.getLogger("archiver.ingest.chunker")


def chunk_document(doc: Document) -> list[Chunk]:
    """
    对一个 Document 进行切块，返回 Chunk 列表。
    策略：
    1. 解析为消息列表
    2. 按轮次分组
    3. dialogue_chunk 切块
    4. 短块合并
    5. 长块滑动窗口切分
    """
    messages = parse_chat_markdown(doc.raw_markdown)
    if not messages:
        return []

    turns = group_into_turns(messages)
    raw_chunks = _dialogue_chunk(turns, doc)
    merged = _merge_short_chunks(raw_chunks)
    final = _split_long_chunks(merged, doc)

    logger.debug("文档 %s 切块结果: %d 个 chunk", doc.doc_id, len(final))
    return final


def _dialogue_chunk(turns: list[list[ParsedMessage]], doc: Document) -> list[_RawChunk]:
    """按问答轮次创建原始切块。"""
    raw_chunks: list[_RawChunk] = []

    for i, turn in enumerate(turns):
        roles = {m.role for m in turn}
        role_summary = turn[0].role if len(roles) == 1 else "mixed"

        parts: list[str] = []
        for msg in turn:
            label = "User" if msg.role == "user" else "Assistant" if msg.role == "assistant" else msg.role.capitalize()
            parts.append(f"{label}:\n{msg.content.strip()}")

        text = "\n\n".join(p for p in parts if p)
        text = normalize_chunk_text(text)

        if text:
            positions = [m.position for m in turn]
            msg_range = f"{min(positions)}-{max(positions)}" if positions else str(i)
            raw_chunks.append(_RawChunk(
                text=text,
                role_summary=role_summary,
                message_range=msg_range,
                turn_index=i,
            ))

    return raw_chunks


def _merge_short_chunks(chunks: list[_RawChunk]) -> list[_RawChunk]:
    """合并过短的 chunk（< CHUNK_MIN_MERGE 字符）与相邻 chunk。"""
    if not chunks:
        return []

    merged: list[_RawChunk] = []
    for chunk in chunks:
        if len(chunk.text) < CHUNK_MIN_MERGE and merged:
            prev = merged[-1]
            new_role = prev.role_summary if prev.role_summary == chunk.role_summary else "mixed"
            merged[-1] = _RawChunk(
                text=f"{prev.text}\n\n{chunk.text}".strip(),
                role_summary=new_role,
                message_range=f"{prev.message_range},{chunk.message_range}",
                turn_index=prev.turn_index,
            )
        else:
            merged.append(chunk)

    return merged


def _split_long_chunks(raw_chunks: list[_RawChunk], doc: Document) -> list[Chunk]:
    """将超过 CHUNK_MAX_SIZE 的块用滑动窗口再切分，组装最终 Chunk。"""
    result: list[Chunk] = []
    idx = 0

    for rc in raw_chunks:
        parts = _semantic_window_split(rc.text) if len(rc.text) > CHUNK_MAX_SIZE else [rc.text]

        for part in parts:
            prefix = build_chunk_prefix(doc.platform, doc.title, doc.created_at)
            full_text = f"{prefix}\n{part.strip()}"
            chunk_id = f"{doc.doc_id}_{idx}"

            result.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source_path=doc.path,
                platform=doc.platform,
                title=doc.title,
                message_range=rc.message_range,
                role_summary=rc.role_summary,
                text=full_text,
                char_count=len(full_text),
                created_at=doc.created_at,
                url=doc.url,
                tags=doc.tags[:],
                text_hash=text_hash(full_text),
            ))
            idx += 1

    return result


def _semantic_window_split(text: str) -> list[str]:
    """
    滑动窗口切分（用于长轮次）。
    尝试在句子/段落边界切分。
    """
    if len(text) <= CHUNK_TARGET_SIZE:
        return [text]

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_TARGET_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""],
        )
        parts = splitter.split_text(text)
        return [p.strip() for p in parts if p.strip()]
    except ImportError:
        pass

    step = max(1, CHUNK_TARGET_SIZE - CHUNK_OVERLAP)
    parts = []
    for i in range(0, len(text), step):
        segment = text[i:i + CHUNK_TARGET_SIZE].strip()
        if segment:
            parts.append(segment)
    return parts


class _RawChunk:
    """内部中间结构，最终会转换为 Chunk。"""
    __slots__ = ("text", "role_summary", "message_range", "turn_index")

    def __init__(self, text: str, role_summary: str, message_range: str, turn_index: int):
        self.text = text
        self.role_summary = role_summary
        self.message_range = message_range
        self.turn_index = turn_index

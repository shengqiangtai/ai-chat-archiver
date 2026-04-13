"""ChromaDB 向量存储 — 可抽象接口便于替换为 FAISS / Qdrant。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb

from app.core.config import CHROMA_PATH, EMBEDDING_DIM
from app.models.schemas import Chunk, RetrievalHit

logger = logging.getLogger("archiver.vectorstore")

COLLECTION_NAME = "ai_chats"


class ChromaStore:
    """封装 ChromaDB 的读写操作，接口层可替换。"""

    def __init__(self, persist_path: Path | None = None):
        path = persist_path or CHROMA_PATH
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIM},
        )

    def reset(self) -> None:
        """删除并重建 collection。"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIM},
        )
        logger.info("向量库已重置")

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """
        批量写入/更新 Chunk（外部传入 embedding）。
        以 chunk_id 为主键，支持增量更新。
        """
        if not chunks or not embeddings:
            return 0

        batch_size = 100
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_emb = embeddings[i:i + batch_size]

            self.collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_emb,
                documents=[c.text for c in batch_chunks],
                metadatas=[
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "platform": c.platform,
                        "title": c.title,
                        "role_summary": c.role_summary,
                        "message_range": c.message_range,
                        "char_count": c.char_count,
                        "created_at": c.created_at,
                        "url": c.url or "",
                        "source_path": c.source_path,
                        "text_hash": c.text_hash,
                    }
                    for c in batch_chunks
                ],
            )
            total += len(batch_chunks)

        return total

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 15,
        filter_platform: Optional[str] = None,
        filter_doc_id: Optional[str] = None,
    ) -> list[RetrievalHit]:
        """向量相似度检索，返回 RetrievalHit 列表。"""
        where = {}
        if filter_platform:
            where["platform"] = filter_platform
        if filter_doc_id:
            where["doc_id"] = filter_doc_id

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            where=where if where else None,
            include=["metadatas", "documents", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[RetrievalHit] = []
        for idx, chunk_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            doc = docs[idx] if idx < len(docs) else ""
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - distance))

            hits.append(RetrievalHit(
                chunk_id=str(chunk_id),
                doc_id=str(meta.get("doc_id") or ""),
                score=score,
                rerank_score=None,
                platform=str(meta.get("platform") or "Unknown"),
                title=str(meta.get("title") or "Untitled"),
                excerpt=str(doc or ""),
                path=str(meta.get("source_path") or ""),
                created_at=str(meta.get("created_at") or ""),
                url=str(meta.get("url") or "") or None,
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除某个文档的所有 Chunk。"""
        found = self.collection.get(where={"doc_id": doc_id}, include=[])
        ids = found.get("ids") or []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def get_stats(self) -> dict:
        """返回各平台的 chunk 数量统计。"""
        data = self.collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        by_platform: dict[str, int] = {}
        for meta in metas:
            platform = str(meta.get("platform") or "Unknown")
            by_platform[platform] = by_platform.get(platform, 0) + 1
        return {"total_chunks": len(metas), "by_platform": by_platform}

    def collection_count(self) -> int:
        return int(self.collection.count())


_store_instance: ChromaStore | None = None


def get_store() -> ChromaStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaStore()
    return _store_instance

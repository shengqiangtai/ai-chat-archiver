"""查询缓存 — 两层缓存策略。

Layer 1: query → retrieval result cache
Layer 2: query + selected sources → final answer cache

使用 SQLite 实现本地文件缓存。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from app.core.config import CACHE_DIR, CACHE_TTL_SECONDS
from app.utils.hashing import text_hash

logger = logging.getLogger("archiver.cache")


class QueryCache:
    """基于 SQLite 的查询缓存。"""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "query_cache.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS answer_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()

    def get_retrieval(self, query: str) -> Optional[list[dict]]:
        """获取检索结果缓存。"""
        key = text_hash(query.strip().lower())
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT result, created_at FROM retrieval_cache WHERE query_hash = ?", (key,)
            ).fetchone()
        if not row:
            return None
        if time.time() - row[1] > CACHE_TTL_SECONDS:
            self._delete_retrieval(key)
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def set_retrieval(self, query: str, results: list[dict]) -> None:
        """设置检索结果缓存。"""
        key = text_hash(query.strip().lower())
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO retrieval_cache(query_hash, query, result, created_at) VALUES (?,?,?,?)",
                (key, query, json.dumps(results, ensure_ascii=False), time.time()),
            )
            conn.commit()

    def get_answer(self, query: str, source_ids: list[str]) -> Optional[dict]:
        """获取最终回答缓存。"""
        cache_key = text_hash(f"{query.strip().lower()}|{'|'.join(sorted(source_ids))}")
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT result, created_at FROM answer_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if not row:
            return None
        if time.time() - row[1] > CACHE_TTL_SECONDS:
            self._delete_answer(cache_key)
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def set_answer(self, query: str, source_ids: list[str], result: dict) -> None:
        """设置最终回答缓存。"""
        cache_key = text_hash(f"{query.strip().lower()}|{'|'.join(sorted(source_ids))}")
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO answer_cache(cache_key, query, result, created_at) VALUES (?,?,?,?)",
                (cache_key, query, json.dumps(result, ensure_ascii=False), time.time()),
            )
            conn.commit()

    def clear_all(self) -> None:
        """清除所有缓存。"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM retrieval_cache")
            conn.execute("DELETE FROM answer_cache")
            conn.commit()
        logger.info("缓存已清除")

    def _delete_retrieval(self, key: str) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM retrieval_cache WHERE query_hash = ?", (key,))
            conn.commit()

    def _delete_answer(self, key: str) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM answer_cache WHERE cache_key = ?", (key,))
            conn.commit()


_cache_instance: QueryCache | None = None


def get_cache() -> QueryCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache()
    return _cache_instance

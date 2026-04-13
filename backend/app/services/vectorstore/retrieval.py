"""检索模块 — 封装完整的向量检索 + rerank 流程。"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.core.config import RERANK_TOP_N, RETRIEVAL_SCORE_THRESHOLD, RETRIEVAL_TOP_K
from app.models.schemas import RetrievalHit
from app.services.embedding.embedder import get_embedder
from app.services.vectorstore.chroma_store import get_store

logger = logging.getLogger("archiver.retrieval")


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    top_n: int = RERANK_TOP_N,
    platform_filter: Optional[str] = None,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    use_rerank: bool = True,
) -> list[RetrievalHit]:
    """
    完整检索链路：
    1. query 预处理
    2. embedding
    3. vector search top_k
    4. rerank top_n（可选）
    5. 过滤低分
    6. 去重（同一 doc 最多 2 个）
    """
    user_query = preprocess_query(query)
    if not user_query:
        return []

    t0 = time.time()

    embedder = get_embedder()
    query_embedding = embedder.encode_query(user_query)
    t_embed = time.time() - t0

    store = get_store()
    candidates = store.query(
        query_embedding=query_embedding,
        top_k=max(top_k, RETRIEVAL_TOP_K),
        filter_platform=platform_filter,
    )
    t_search = time.time() - t0 - t_embed

    if use_rerank and candidates:
        try:
            from app.services.rerank.reranker import get_reranker
            reranker = get_reranker()
            candidates = reranker.rerank(user_query, candidates, top_n=top_n)
            t_rerank = time.time() - t0 - t_embed - t_search
            logger.info("Rerank 耗时: %.2fs", t_rerank)
        except Exception as e:
            logger.warning("Rerank 失败，跳过: %s", e)

    filtered: list[RetrievalHit] = []
    per_doc_count: dict[str, int] = {}
    final_limit = max(1, top_n)

    for hit in candidates:
        effective_score = hit.rerank_score if hit.rerank_score is not None else hit.score
        if effective_score < score_threshold:
            continue
        used = per_doc_count.get(hit.doc_id, 0)
        if used >= 2:
            continue
        per_doc_count[hit.doc_id] = used + 1
        filtered.append(hit)
        if len(filtered) >= final_limit:
            break

    total_time = time.time() - t0
    logger.info(
        "检索完成: query=%r, 候选=%d, 最终=%d, embed=%.2fs, search=%.2fs, total=%.2fs",
        user_query[:50], len(candidates), len(filtered), t_embed, t_search, total_time,
    )
    return filtered


def preprocess_query(query: str) -> str:
    """
    轻量 query 预处理：
    - trim
    - 去掉无意义停用前缀
    - 规则改写模糊词
    """
    q = (query or "").strip()
    if not q:
        return ""

    prefixes_to_remove = [
        "请问", "帮我", "请帮我", "我想知道", "我想问",
        "你能告诉我", "能不能帮我", "帮忙", "麻烦",
    ]
    for prefix in prefixes_to_remove:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    replacements = {
        "这次": "", "之前": "", "上次": "", "那个聊天": "",
        "那次对话": "", "那个对话": "",
    }
    for old, new in replacements.items():
        q = q.replace(old, new)

    return q.strip()

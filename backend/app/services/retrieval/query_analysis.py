"""Heuristic query analysis for retrieval routing."""

from __future__ import annotations

import re

from app.models.schemas import QueryAnalysis

_PATH_RE = re.compile(
    r"(?:~|/|\$[A-Z_][A-Z0-9_]*)(?:/[A-Za-z0-9_.-]+)+/?|"
    r"\b[A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml|toml|sql|sh)\b"
)

_RELATION_PATTERNS = (
    "depends on",
    "depend on",
    "related to",
    "connect",
    "connected to",
    "uses",
    "used by",
    "which module",
    "which file",
    "依赖",
    "关联",
    "关系",
    "连接",
    "引用",
    "谁用",
    "谁依赖",
)

_FOLLOW_UP_PATTERNS = (
    "那",
    "它",
    "这个",
    "那个",
    "上次",
    "之前",
    "这次",
    "刚才",
    "why",
    "what about",
    "how about",
    "why does",
    "为什么",
    "怎么",
    "会不会",
    "会不会是",
)


def analyze_query(query: str) -> QueryAnalysis:
    text = (query or "").strip()
    if not text:
        return QueryAnalysis(
            query_type="general",
            enable_rewrite=False,
            enable_rerank=True,
            enable_graph=True,
            reasons=["empty query"],
        )

    reasons: list[str] = []
    if _PATH_RE.search(text):
        reasons.append("matched symbolic path pattern")
        return QueryAnalysis(
            query_type="symbolic",
            enable_rewrite=False,
            enable_rerank=False,
            enable_graph=False,
            reasons=reasons,
        )

    if _has_pattern(text, _FOLLOW_UP_PATTERNS):
        reasons.append("matched follow-up/context markers")
        return QueryAnalysis(
            query_type="follow_up",
            enable_rewrite=True,
            enable_rerank=True,
            enable_graph=True,
            reasons=reasons,
        )

    if _has_pattern(text, _RELATION_PATTERNS):
        reasons.append("matched relation markers")
        return QueryAnalysis(
            query_type="relation",
            enable_rewrite=False,
            enable_rerank=True,
            enable_graph=True,
            reasons=reasons,
        )

    return QueryAnalysis(
        query_type="general",
        enable_rewrite=False,
        enable_rerank=True,
        enable_graph=True,
        reasons=["default general query"],
    )


def _has_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(pattern in text or pattern in lowered for pattern in patterns)

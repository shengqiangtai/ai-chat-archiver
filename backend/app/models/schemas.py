"""Pydantic 模型定义 — 请求/响应 + 内部数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

ScoreValue = Optional[float]


# ═══════════════════════════════════════════════════════════════════════════
# 基础聊天归档
# ═══════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str
    content: str
    time: Optional[str] = None


class SaveRequest(BaseModel):
    platform: str
    model: Optional[str] = None
    title: str
    url: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    messages: List[Message] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    platform: Optional[str] = None
    limit: int = 20


# ═══════════════════════════════════════════════════════════════════════════
# 文档结构（内部使用 dataclass）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    doc_id: str
    platform: str
    title: str
    created_at: str
    updated_at: str
    path: str
    url: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    raw_markdown: str = ""
    model_name: Optional[str] = None
    file_hash: str = ""
    modified_time: float = 0.0


@dataclass
class ParsedMessage:
    role: str
    content: str
    position: int
    timestamp: Optional[str] = None
    section_title: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    platform: str
    title: str
    message_range: str
    role_summary: str
    text: str
    char_count: int
    created_at: str
    url: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    model_name: Optional[str] = None
    turn_index: int = 0
    chunk_index: int = 0
    text_hash: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# 检索结果
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalHit:
    chunk_id: str
    doc_id: str
    score: float
    rerank_score: ScoreValue
    platform: str
    title: str
    excerpt: str
    path: str
    created_at: str
    url: Optional[str] = None
    keyword_score: ScoreValue = None
    fused_score: ScoreValue = None
    entity_score: ScoreValue = None
    role_summary: str = ""
    message_range: str = ""
    model_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)
    turn_index: int = 0
    chunk_index: int = 0


@dataclass
class QueryAnalysis:
    query_type: str
    enable_rewrite: bool
    enable_rerank: bool
    enable_graph: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class EntityMention:
    entity_id: str
    name: str
    norm_name: str
    entity_type: str
    chunk_id: str
    doc_id: str
    turn_index: int
    mention_text: str


# ═══════════════════════════════════════════════════════════════════════════
# 知识库 API
# ═══════════════════════════════════════════════════════════════════════════

class KbSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    platform_filter: Optional[str] = None
    model_filter: Optional[str] = None
    tag_filter: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    retrieval_mode: str = "mix"
    rerank_mode: str = "auto"
    score_threshold: float = 0.30
    rewrite_query: bool = True
    include_debug: bool = False


class RetrievalDebug(BaseModel):
    query_analysis: Optional[QueryAnalysis] = None
    analysis_scope: Optional[str] = None
    graph_routed: bool = False
    graph_hit_count: int = 0
    graph_hits: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class KbSearchResponse(BaseModel):
    query: str
    hits: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    debug: Optional[RetrievalDebug] = None


class QARequest(BaseModel):
    query: str
    mode: str = "concise"
    top_k: int = 15
    top_n: int = 5
    platform_filter: Optional[str] = None
    model_filter: Optional[str] = None
    tag_filter: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    retrieval_mode: str = "mix"
    rerank_mode: str = "auto"
    rewrite_query: bool = True


class QAResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    debug: Optional[RetrievalDebug] = None


class OllamaModelUpdateRequest(BaseModel):
    model: str


# ═══════════════════════════════════════════════════════════════════════════
# 来源引用
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SourceRef:
    source_id: str
    chunk_id: str
    platform: str
    title: str
    path: str
    score: float
    rerank_score: ScoreValue = None
    url: Optional[str] = None
    excerpt: str = ""
    message_range: str = ""
    turn_index: int = 0


@dataclass
class Citation:
    source_id: str
    reason: str


@dataclass
class AnswerResult:
    answer: str
    citations: List[Citation]
    uncertainty: Optional[str]
    sources: List[SourceRef]
    debug: Dict[str, Any] = field(default_factory=dict)

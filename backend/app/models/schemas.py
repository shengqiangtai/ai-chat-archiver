"""Pydantic 模型定义 — 请求/响应 + 内部数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field


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
    tags: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)


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
    text_hash: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# 检索结果
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalHit:
    chunk_id: str
    doc_id: str
    score: float
    rerank_score: Optional[float]
    platform: str
    title: str
    excerpt: str
    path: str
    created_at: str
    url: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# 知识库 API
# ═══════════════════════════════════════════════════════════════════════════

class KbSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    platform_filter: Optional[str] = None
    score_threshold: float = 0.30


class KbSearchResponse(BaseModel):
    query: str
    hits: list[dict] = Field(default_factory=list)
    total: int = 0


class QARequest(BaseModel):
    query: str
    mode: str = "concise"
    top_k: int = 15
    top_n: int = 5


class QAResponse(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    sources: list[dict] = Field(default_factory=list)
    debug: Optional[dict] = None


class OllamaModelUpdateRequest(BaseModel):
    model: str


# ═══════════════════════════════════════════════════════════════════════════
# 来源引用
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SourceRef:
    source_id: str
    platform: str
    title: str
    path: str
    score: float
    rerank_score: Optional[float] = None
    url: Optional[str] = None
    excerpt: str = ""


@dataclass
class Citation:
    source_id: str
    reason: str


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    uncertainty: Optional[str]
    sources: list[SourceRef]
    debug: dict[str, Any] = field(default_factory=dict)

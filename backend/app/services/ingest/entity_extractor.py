"""轻量实体抽取器 — 面向本地聊天知识库的低成本实体索引。"""

from __future__ import annotations

import re
from collections import Counter

from app.models.schemas import Chunk, EntityMention
from app.services.graph.relation_extractor import extract_relations
from app.utils.hashing import text_hash

_GENERIC_TOKENS = {
    "user", "assistant", "system", "source", "query", "instruct", "json", "http", "https",
    "true", "false", "null", "python",  # python 保留为技术实体，后面单独放行
}

_TECH_ALLOWLIST = {
    "python", "fastapi", "chromadb", "sqlite", "qwen", "ollama", "lm studio", "chatgpt",
    "claude", "gemini", "deepseek", "poe", "rag", "rerank", "embedding", "transformers",
    "codex", "codex cli", "skill", "skill-installer", "skill-creator", "plugin",
}

_FILE_RE = re.compile(r"\b[A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml|toml|sql|sh)\b")
_PATH_RE = re.compile(r"(?:~|/|\$[A-Z_][A-Z0-9_]*)(?:/[A-Za-z0-9_.-]+)+/?")
_COMMAND_RE = re.compile(r"(?:\$)?[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b")
_MODEL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_.:/-]{2,}(?:-[A-Za-z0-9_.:/-]+)+\b")
_CAMEL_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[A-Z]{2,}[A-Za-z0-9_-]*)\b")
_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\b")
_MIXED_PHRASE_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,})){1,2}\b")
_CJK_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,10}(?:模型|向量库|检索|索引|缓存|工作流|问答|嵌入|重排|引用|对话)")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:/-]{2,48}")


def normalize_entity_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", (name or "").strip())
    normalized = normalized.strip("`'\"[](){}<>.,;:!?")
    normalized = normalized.lstrip("$")
    return normalized.lower()


def extract_entities_from_text(text: str) -> list[tuple[str, str]]:
    if not text:
        return []

    candidates: list[tuple[str, str]] = []

    for match in _PATH_RE.findall(text):
        candidates.append((match, "path"))
        tail = match.rstrip("/").split("/")[-1]
        if "." in tail:
            candidates.append((tail, "file"))
    for match in _FILE_RE.findall(text):
        candidates.append((match, "file"))
    for match in _COMMAND_RE.findall(text):
        candidates.append((match, "command"))
    for match in _MODEL_RE.findall(text):
        candidates.append((match, "model"))
    for match in _PHRASE_RE.findall(text):
        candidates.append((match, "phrase"))
    for match in _MIXED_PHRASE_RE.findall(text):
        candidates.append((match, "phrase"))
    for match in _CAMEL_RE.findall(text):
        candidates.append((match, "tech"))
    for match in _CJK_TERM_RE.findall(text):
        candidates.append((match, "topic"))

    # 最后补一轮 token 扫描，覆盖 python/rag/sqlite 这类全小写技术词
    for token in _TOKEN_RE.findall(text):
        norm = normalize_entity_name(token)
        if norm in _TECH_ALLOWLIST:
            candidates.append((token, "tech"))

    deduped: dict[str, tuple[str, str]] = {}
    for raw, entity_type in candidates:
        norm = normalize_entity_name(raw)
        if not norm or len(norm) < 2:
            continue
        if norm in _GENERIC_TOKENS and norm not in _TECH_ALLOWLIST:
            continue
        if norm.isdigit():
            continue
        if norm not in deduped:
            deduped[norm] = (raw.strip(), entity_type)

    return list(deduped.values())


def extract_entities_from_chunk(chunk: Chunk) -> list[EntityMention]:
    mentions, _ = extract_graph_metadata_from_chunk(chunk)
    return mentions


def extract_entities_from_chunks(chunks: list[Chunk]) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    for chunk in chunks:
        chunk_mentions, chunk_relations = extract_graph_metadata_from_chunk(chunk)
        setattr(chunk, "graph_relations", chunk_relations)
        mentions.extend(chunk_mentions)
    return mentions


def extract_graph_metadata_from_chunk(chunk: Chunk) -> tuple[list[EntityMention], list[dict[str, str]]]:
    mentions: list[EntityMention] = []
    seen: set[str] = set()
    raw_entities = extract_entities_from_text(chunk.text)

    for tag in chunk.tags:
        if tag.strip():
            raw_entities.append((tag.strip(), "tag"))
    if chunk.model_name:
        raw_entities.append((chunk.model_name, "model"))
    if chunk.platform:
        raw_entities.append((chunk.platform, "platform"))

    for name, entity_type in raw_entities:
        norm_name = normalize_entity_name(name)
        if not norm_name:
            continue
        key = f"{chunk.chunk_id}|{norm_name}"
        if key in seen:
            continue
        seen.add(key)

        mentions.append(
            EntityMention(
                entity_id=text_hash(f"entity:{norm_name}")[:16],
                name=name,
                norm_name=norm_name,
                entity_type=entity_type,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                turn_index=chunk.turn_index,
                mention_text=name,
            )
        )

    relations = extract_relations(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        entity_names=[mention.name for mention in mentions],
    )
    return mentions, relations


def extract_query_entities(query: str, limit: int = 8) -> list[str]:
    entities = [normalize_entity_name(name) for name, _ in extract_entities_from_text(query)]
    counts = Counter(e for e in entities if e)
    ordered = [name for name, _ in counts.most_common(limit)]
    if not ordered:
        # fallback: 用词面 token 做弱实体命中
        ordered = [normalize_entity_name(token) for token in _TOKEN_RE.findall(query)[:limit]]
    return [e for e in ordered if e]

"""引用解析与验证模块。

负责：
- 从 LLM 输出中提取引用
- 验证引用是否与来源匹配
- 标记低可信回答
"""

from __future__ import annotations

import json
import re
from typing import Optional

from app.models.schemas import AnswerResult, Citation, RetrievalHit, SourceRef


def parse_llm_output(raw: str, hits: list[RetrievalHit]) -> AnswerResult:
    """
    解析 LLM 输出，支持 JSON 格式和纯文本格式的容错解析。
    """
    sources = _build_sources(hits)

    parsed = _try_parse_json(raw)
    if parsed:
        answer = str(parsed.get("answer") or raw)
        citations = [
            Citation(source_id=str(c.get("source_id", "")), reason=str(c.get("reason", "")))
            for c in (parsed.get("citations") or [])
        ]
        uncertainty = parsed.get("uncertainty")
        if uncertainty and str(uncertainty).lower() in ("null", "none", ""):
            uncertainty = None
    else:
        answer = raw.strip()
        citations = _extract_citations_from_text(answer)
        uncertainty = None

    result = AnswerResult(
        answer=answer,
        citations=citations,
        uncertainty=str(uncertainty) if uncertainty else None,
        sources=sources,
    )

    _validate_citations(result, hits)
    return result


def _try_parse_json(text: str) -> Optional[dict]:
    """尝试从文本中提取 JSON。"""
    text = text.strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _extract_citations_from_text(text: str) -> list[Citation]:
    """从纯文本回答中提取 [Source X] 引用。"""
    pattern = re.compile(r'\[Source\s*(\d+)\]')
    found = pattern.findall(text)
    seen = set()
    citations = []
    for source_id in found:
        if source_id not in seen:
            seen.add(source_id)
            citations.append(Citation(source_id=source_id, reason="文中引用"))
    return citations


def _build_sources(hits: list[RetrievalHit]) -> list[SourceRef]:
    """将检索结果转为 SourceRef 列表。"""
    sources: list[SourceRef] = []
    seen_docs: set[str] = set()

    for idx, hit in enumerate(hits, start=1):
        key = hit.doc_id or f"{hit.platform}:{hit.title}"
        if key in seen_docs:
            continue
        seen_docs.add(key)
        sources.append(SourceRef(
            source_id=str(idx),
            platform=hit.platform,
            title=hit.title,
            path=hit.path,
            score=hit.score,
            rerank_score=hit.rerank_score,
            url=hit.url,
            excerpt=hit.excerpt[:200] if hit.excerpt else "",
        ))

    return sources


def _validate_citations(result: AnswerResult, hits: list[RetrievalHit]) -> None:
    """
    验证引用：
    - 如果引用的 source_id 不存在于 sources 中，标记 warning
    - 如果回答中没有任何引用，标记为低可信
    """
    valid_ids = {s.source_id for s in result.sources}

    invalid_citations = []
    for c in result.citations:
        if c.source_id not in valid_ids:
            invalid_citations.append(c.source_id)

    warnings = []
    if invalid_citations:
        warnings.append(f"引用了不存在的来源: {', '.join(invalid_citations)}")

    if not result.citations and hits:
        warnings.append("回答中未引用任何来源，可信度较低")

    if warnings:
        if result.uncertainty:
            result.uncertainty += "; " + "; ".join(warnings)
        else:
            result.uncertainty = "; ".join(warnings)

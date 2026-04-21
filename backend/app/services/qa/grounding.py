"""Lightweight grounding checks for answer support."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from app.models.schemas import RetrievalHit


@dataclass
class GroundingResult:
    supported: bool
    should_downgrade: bool
    support_rate: float
    message: str
    unsupported_claims: list[str] = field(default_factory=list)


def evaluate_grounding(*, answer: str, hits: list[RetrievalHit]) -> GroundingResult:
    claims = _split_claims(answer)
    if not hits:
        return GroundingResult(
            supported=False,
            should_downgrade=True,
            support_rate=0.0,
            message="no_supporting_hits",
            unsupported_claims=claims,
        )

    if not claims:
        return GroundingResult(
            supported=True,
            should_downgrade=False,
            support_rate=1.0,
            message="no_claims_detected",
        )

    support_bags = [_support_tokens(hit.excerpt or "") for hit in hits if hit.excerpt]
    unsupported: list[str] = []
    for claim in claims:
        claim_tokens = _support_tokens(claim)
        if len(claim_tokens) < 2:
            continue
        if not any(_claim_supported(claim_tokens, source_tokens) for source_tokens in support_bags):
            unsupported.append(claim)

    supported_claims = max(0, len(claims) - len(unsupported))
    support_rate = supported_claims / len(claims) if claims else 1.0
    supported = len(unsupported) == 0
    return GroundingResult(
        supported=supported,
        should_downgrade=not supported,
        support_rate=support_rate,
        message="ok" if supported else "weak_support",
        unsupported_claims=unsupported,
    )


def _split_claims(text: str) -> list[str]:
    cleaned = re.sub(r"\[Source\s*\d+\]", " ", text or "")
    parts = re.split(r"[。！？!?;\n]+", cleaned)
    claims: list[str] = []
    for part in parts:
        claim = re.sub(r"\s+", " ", part).strip(" ，。；;:-")
        if len(claim) >= 8:
            claims.append(claim)
    return claims


def _support_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_./:-]{2,}|[\u4e00-\u9fff]{2,8}", text or "")
    stopwords = {
        "用户", "助手", "来源", "回答", "内容", "相关", "问题", "这个", "那个", "一个", "可以",
        "然后", "以及", "因为", "所以", "如果", "就是", "进行", "用于", "中的", "时候",
        "source", "answer", "content",
    }
    normalized: set[str] = set()
    for token in tokens:
        norm = token.lower().strip()
        if not norm or norm.isdigit() or norm in stopwords:
            continue
        normalized.add(norm)
    return normalized


def _claim_supported(claim_tokens: set[str], source_tokens: set[str]) -> bool:
    overlap = claim_tokens & source_tokens
    if len(overlap) >= 2:
        return True

    for token in claim_tokens:
        if token in source_tokens and re.search(r"[./:_-]|\d", token):
            return True
    return False

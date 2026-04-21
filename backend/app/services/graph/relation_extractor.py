"""Lightweight relation extraction for graph metadata groundwork."""

from __future__ import annotations

import re

_RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdepends on\b", re.IGNORECASE), "depends_on"),
    (re.compile(r"\brequires\b", re.IGNORECASE), "depends_on"),
    (re.compile(r"\bimports?\b", re.IGNORECASE), "depends_on"),
    (re.compile(r"\bused for\b", re.IGNORECASE), "used_for"),
    (re.compile(r"\buses\b", re.IGNORECASE), "uses"),
]


def _entity_positions(text: str, entity_names: list[str]) -> list[tuple[int, str]]:
    lowered = text.lower()
    positions: list[tuple[int, str]] = []
    seen: set[str] = set()

    for entity_name in entity_names:
        name = (entity_name or "").strip()
        if not name:
            continue
        norm = name.lower()
        if norm in seen:
            continue
        idx = lowered.find(norm)
        if idx < 0:
            continue
        seen.add(norm)
        positions.append((idx, name))

    positions.sort(key=lambda item: item[0])
    return positions


def _pick_relation_entities(
    *,
    entity_positions: list[tuple[int, str]],
    relation_start: int,
    relation_end: int,
) -> tuple[str, str] | None:
    source = None
    target = None

    for idx, name in entity_positions:
        if idx < relation_start:
            source = name
        elif idx >= relation_end:
            target = name
            break

    if source and target and source.lower() != target.lower():
        return source, target
    return None


def extract_relations(*, chunk_id: str, text: str, entity_names: list[str]) -> list[dict[str, str]]:
    """Extract lightweight binary relations from a chunk of text."""
    if not text or not entity_names:
        return []

    entity_positions = _entity_positions(text, entity_names)
    if len(entity_positions) < 2:
        return []

    relations: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for pattern, relation_type in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            picked = _pick_relation_entities(
                entity_positions=entity_positions,
                relation_start=match.start(),
                relation_end=match.end(),
            )
            if not picked:
                continue
            source_entity, target_entity = picked
            key = (relation_type, source_entity.lower(), target_entity.lower())
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "chunk_id": chunk_id,
                    "relation_type": relation_type,
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                }
            )

    return relations

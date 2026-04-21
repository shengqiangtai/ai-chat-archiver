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
_TOKEN_CHARS = r"A-Za-z0-9_./:-"


def _entity_occurrences(text: str, entity_names: list[str]) -> list[tuple[int, int, str]]:
    occurrences: list[tuple[int, int, str]] = []
    seen: set[str] = set()

    for entity_name in entity_names:
        name = (entity_name or "").strip()
        if not name:
            continue
        norm = name.lower()
        if norm in seen:
            continue
        seen.add(norm)

        pattern = re.compile(
            rf"(?<![{_TOKEN_CHARS}]){re.escape(name)}(?![{_TOKEN_CHARS}])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            occurrences.append((match.start(), match.end(), name))

    occurrences.sort(key=lambda item: (item[0], item[1], item[2].lower()))
    return occurrences


def _pick_relation_entities(
    *,
    entity_occurrences: list[tuple[int, int, str]],
    relation_start: int,
    relation_end: int,
) -> tuple[str, str] | None:
    before = [occ for occ in entity_occurrences if occ[1] <= relation_start]
    after = [occ for occ in entity_occurrences if occ[0] >= relation_end]

    if not before or not after:
        return None

    source = max(before, key=lambda item: (item[1], item[0]))
    target = min(after, key=lambda item: (item[0], item[1]))

    if source[2].lower() != target[2].lower():
        return source[2], target[2]
    return None


def extract_relations(*, chunk_id: str, text: str, entity_names: list[str]) -> list[dict[str, str]]:
    """Extract lightweight binary relations from a chunk of text."""
    if not text or not entity_names:
        return []

    entity_occurrences = _entity_occurrences(text, entity_names)
    if len(entity_occurrences) < 2:
        return []

    relations: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for pattern, relation_type in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            picked = _pick_relation_entities(
                entity_occurrences=entity_occurrences,
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

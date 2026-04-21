from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.sqlite import Database
from app.models.schemas import Chunk
from app.services.ingest import entity_extractor
from app.services.graph.relation_extractor import extract_relations


def test_extract_relations_finds_dependency_pattern() -> None:
    relations = extract_relations(
        chunk_id="chunk-1",
        text="retrieval.py depends on reranker.py for final ordering.",
        entity_names=["retrieval.py", "reranker.py"],
    )

    assert relations == [
        {
            "chunk_id": "chunk-1",
            "relation_type": "depends_on",
            "source_entity": "retrieval.py",
            "target_entity": "reranker.py",
        }
    ]


def test_graph_metadata_tables_and_indexes_are_created(tmp_path) -> None:
    db = Database(tmp_path / "graph.db")

    with db._conn() as conn:
        rows = conn.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE name IN (
                'kb_graph_relations',
                'idx_kb_graph_relations_chunk',
                'idx_kb_graph_relations_source',
                'idx_kb_graph_relations_target'
            )
            """
        ).fetchall()

    names = {str(row["name"]) for row in rows}

    assert "kb_graph_relations" in names
    assert "idx_kb_graph_relations_chunk" in names
    assert "idx_kb_graph_relations_source" in names
    assert "idx_kb_graph_relations_target" in names


def test_extract_entities_from_chunks_persists_graph_relations(monkeypatch) -> None:
    calls: list[tuple[list[dict[str, str]], str]] = []

    class FakeDB:
        def upsert_graph_relations(self, relations, created_at: str = "") -> int:
            calls.append((list(relations), created_at))
            return len(relations)

    monkeypatch.setattr(entity_extractor, "get_db", lambda: FakeDB())

    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_path="source.md",
        platform="ChatGPT",
        title="Example",
        message_range="0-1",
        role_summary="mixed",
        text="retrieval.py depends on reranker.py for final ordering.",
        char_count=56,
        created_at="2026-04-21",
    )

    mentions = entity_extractor.extract_entities_from_chunks([chunk])

    assert calls == [
        (
            [
                {
                    "chunk_id": "chunk-1",
                    "relation_type": "depends_on",
                    "source_entity": "retrieval.py",
                    "target_entity": "reranker.py",
                }
            ],
            "2026-04-21",
        )
    ]
    assert any(mention.name == "retrieval.py" for mention in mentions)
    assert any(mention.name == "reranker.py" for mention in mentions)

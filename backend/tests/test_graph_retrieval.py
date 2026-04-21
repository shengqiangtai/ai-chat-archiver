from __future__ import annotations

import types
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from app.db.sqlite import Database
from app.models.schemas import Chunk
from app.services.ingest import entity_extractor
from app.services.graph.relation_extractor import extract_relations
from app.services.qa import pipeline as pipeline_module
from app.services.vectorstore import retrieval as retrieval_module


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


def test_extract_relations_uses_later_mentions_for_repeated_entities() -> None:
    relations = extract_relations(
        chunk_id="chunk-2",
        text=(
            "alpha.py is discussed first. beta.py is discussed next. "
            "Later beta.py depends on alpha.py for orchestration."
        ),
        entity_names=["alpha.py", "beta.py"],
    )

    assert relations == [
        {
            "chunk_id": "chunk-2",
            "relation_type": "depends_on",
            "source_entity": "beta.py",
            "target_entity": "alpha.py",
        }
    ]


def test_extract_relations_does_not_match_entity_substrings() -> None:
    relations = extract_relations(
        chunk_id="chunk-3",
        text="reranker.py depends on retrieval.py for final ordering.",
        entity_names=["rerank", "retrieval.py"],
    )

    assert relations == []


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


def test_extract_entities_from_chunks_does_not_persist_graph_relations(monkeypatch) -> None:
    def _fail_get_db():
        raise AssertionError("extract_entities_from_chunks should stay pure")

    monkeypatch.setattr(entity_extractor, "get_db", _fail_get_db, raising=False)

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

    assert any(mention.name == "retrieval.py" for mention in mentions)
    assert any(mention.name == "reranker.py" for mention in mentions)


def test_pipeline_persists_graph_relations_explicitly(monkeypatch) -> None:
    calls: list[tuple[list[dict[str, str]], str]] = []

    class FakeDB:
        def upsert_graph_relations(self, relations, created_at: str = "") -> int:
            calls.append((list(relations), created_at))
            return len(relations)

    chunk = Chunk(
        chunk_id="chunk-4",
        doc_id="doc-4",
        source_path="source.md",
        platform="ChatGPT",
        title="Example",
        message_range="0-1",
        role_summary="mixed",
        text="retrieval.py depends on reranker.py for final ordering.",
        char_count=56,
        created_at="2026-04-21",
    )

    result = pipeline_module._persist_graph_metadata(FakeDB(), [chunk], "2026-04-21")

    assert result == 1
    assert calls == [
        (
            [
                {
                    "chunk_id": "chunk-4",
                    "relation_type": "depends_on",
                    "source_entity": "retrieval.py",
                    "target_entity": "reranker.py",
                }
            ],
            "2026-04-21",
        )
    ]


def test_relation_query_graph_hit_can_outrank_weaker_dense_hit(monkeypatch) -> None:
    class DummyEmbedder:
        def encode_query(self, query: str) -> list[float]:
            return [0.0]

    class DummyStore:
        def query(self, *args, **kwargs) -> list[retrieval_module.RetrievalHit]:
            return [
                retrieval_module.RetrievalHit(
                    chunk_id="dense-weak",
                    doc_id="doc-dense",
                    score=0.01,
                    rerank_score=None,
                    platform="ChatGPT",
                    title="Weak Dense Hit",
                    excerpt="weak dense content",
                    path="dense.md",
                    created_at="2026-04-21",
                )
            ]

    class DummyDB:
        def search_kb_chunks(self, *args, **kwargs) -> list[dict]:
            return []

        def search_entities(self, *args, **kwargs) -> list[dict]:
            return []

        def get_related_entities(self, *args, **kwargs) -> list[dict]:
            return []

        def search_entity_chunks(self, *args, **kwargs) -> list[dict]:
            return []

    class DummyCache:
        def get_retrieval(self, *args, **kwargs):
            return None

        def set_retrieval(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr(retrieval_module, "get_embedder", lambda: DummyEmbedder())
    monkeypatch.setattr(retrieval_module, "get_store", lambda: DummyStore())
    monkeypatch.setattr(retrieval_module, "get_db", lambda: DummyDB())
    monkeypatch.setattr(retrieval_module, "get_cache", lambda: DummyCache())
    monkeypatch.setattr(
        retrieval_module,
        "retrieve_graph_candidates",
        lambda *args, **kwargs: [
            retrieval_module.RetrievalHit(
                chunk_id="graph-strong",
                doc_id="doc-graph",
                score=0.0,
                rerank_score=None,
                platform="ChatGPT",
                title="Graph Hit",
                excerpt="retrieval depends on reranker",
                path="graph.md",
                created_at="2026-04-21",
                entity_score=9.0,
                entity_names=["retrieval.py", "reranker.py"],
            )
        ],
    )

    result = retrieval_module.retrieve_debug(
        query="Which module depends on the reranker?",
        use_cache=False,
        retrieval_mode="mix",
        use_rerank=False,
        score_threshold=0.0,
        expand_neighbors=False,
        top_k=3,
        top_n=3,
    )

    assert [hit["chunk_id"] for hit in result["hits"][:2]] == ["graph-strong", "dense-weak"]
    assert result["debug"]["graph_routed"] is True
    assert result["debug"]["graph_hit_count"] == 1


def test_graph_routing_is_skipped_when_query_analysis_disables_it(monkeypatch) -> None:
    class DummyEmbedder:
        def encode_query(self, query: str) -> list[float]:
            return [0.0]

    class DummyStore:
        def query(self, *args, **kwargs) -> list[dict]:
            return []

    class DummyDB:
        def search_kb_chunks(self, *args, **kwargs) -> list[dict]:
            return []

        def search_entities(self, *args, **kwargs) -> list[dict]:
            raise AssertionError("entity lookup should be skipped")

        def get_related_entities(self, *args, **kwargs) -> list[dict]:
            raise AssertionError("entity lookup should be skipped")

        def search_entity_chunks(self, *args, **kwargs) -> list[dict]:
            raise AssertionError("entity lookup should be skipped")

    class DummyCache:
        def get_retrieval(self, *args, **kwargs):
            return None

        def set_retrieval(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr(retrieval_module, "get_embedder", lambda: DummyEmbedder())
    monkeypatch.setattr(retrieval_module, "get_store", lambda: DummyStore())
    monkeypatch.setattr(retrieval_module, "get_db", lambda: DummyDB())
    monkeypatch.setattr(retrieval_module, "get_cache", lambda: DummyCache())

    def _unexpected_graph_call(*args, **kwargs):
        raise AssertionError("graph retrieval should be skipped")

    monkeypatch.setattr(retrieval_module, "retrieve_graph_candidates", _unexpected_graph_call)

    result = retrieval_module.retrieve_debug(
        query="backend/app/services/qa/pipeline.py",
        use_cache=False,
        retrieval_mode="mix",
        use_rerank=False,
        expand_neighbors=False,
        top_k=3,
        top_n=3,
    )

    assert result["debug"]["query_analysis"]["enable_graph"] is False
    assert result["debug"]["graph_routed"] is False
    assert result["debug"]["graph_hit_count"] == 0
    assert result["debug"]["graph_hits"] == []


def test_debug_exposes_graph_routing_state_and_hits(monkeypatch) -> None:
    class DummyEmbedder:
        def encode_query(self, query: str) -> list[float]:
            return [0.0]

    class DummyStore:
        def query(self, *args, **kwargs) -> list[dict]:
            return []

    class DummyDB:
        def search_kb_chunks(self, *args, **kwargs) -> list[dict]:
            return []

        def search_entities(self, *args, **kwargs) -> list[dict]:
            return []

        def get_related_entities(self, *args, **kwargs) -> list[dict]:
            return []

        def search_entity_chunks(self, *args, **kwargs) -> list[dict]:
            return []

    class DummyCache:
        def get_retrieval(self, *args, **kwargs):
            return None

        def set_retrieval(self, *args, **kwargs) -> None:
            return None

    graph_hits = [
        retrieval_module.RetrievalHit(
            chunk_id="graph-1",
            doc_id="doc-1",
            score=0.0,
            rerank_score=None,
            platform="ChatGPT",
            title="Graph 1",
            excerpt="first graph hit",
            path="graph-1.md",
            created_at="2026-04-21",
            entity_score=3.0,
            entity_names=["alpha", "beta"],
        ),
        retrieval_module.RetrievalHit(
            chunk_id="graph-2",
            doc_id="doc-2",
            score=0.0,
            rerank_score=None,
            platform="ChatGPT",
            title="Graph 2",
            excerpt="second graph hit",
            path="graph-2.md",
            created_at="2026-04-21",
            entity_score=2.0,
            entity_names=["beta", "gamma"],
        ),
    ]

    monkeypatch.setattr(retrieval_module, "get_embedder", lambda: DummyEmbedder())
    monkeypatch.setattr(retrieval_module, "get_store", lambda: DummyStore())
    monkeypatch.setattr(retrieval_module, "get_db", lambda: DummyDB())
    monkeypatch.setattr(retrieval_module, "get_cache", lambda: DummyCache())
    monkeypatch.setattr(retrieval_module, "retrieve_graph_candidates", lambda *args, **kwargs: graph_hits)

    result = retrieval_module.retrieve_debug(
        query="Which module depends on beta?",
        use_cache=False,
        retrieval_mode="mix",
        use_rerank=False,
        expand_neighbors=False,
        top_k=3,
        top_n=3,
    )

    assert result["debug"]["graph_routed"] is True
    assert result["debug"]["graph_hit_count"] == 2
    assert [hit["chunk_id"] for hit in result["debug"]["graph_hits"]] == ["graph-1", "graph-2"]
    assert result["debug"]["final_hits"][0]["chunk_id"] == "graph-1"

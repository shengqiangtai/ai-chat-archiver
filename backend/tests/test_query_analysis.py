from __future__ import annotations

import types
import sys
from pathlib import Path
from typing import Any, get_args, get_origin

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models.schemas import KbSearchResponse, QAResponse, RetrievalDebug
from app.services.retrieval.query_analysis import analyze_query

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from app.services.vectorstore import retrieval as retrieval_module


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = get_args(annotation)
    if type(None) in args:
        for arg in args:
            if arg is not type(None):
                return arg
    return annotation


def test_symbolic_path_query_skips_graph_and_rewrite() -> None:
    analysis = analyze_query("backend/app/services/qa/pipeline.py")

    assert analysis.query_type == "symbolic"
    assert analysis.enable_rewrite is False
    assert analysis.enable_graph is False


def test_relation_query_enables_graph_and_rerank() -> None:
    analysis = analyze_query("Which module depends on the reranker?")

    assert analysis.query_type == "relation"
    assert analysis.enable_graph is True
    assert analysis.enable_rerank is True


def test_follow_up_query_enables_rewrite() -> None:
    analysis = analyze_query("那它为什么会超时？")

    assert analysis.enable_rewrite is True


def test_response_debug_state_is_typed() -> None:
    assert "debug" in KbSearchResponse.model_fields
    assert _unwrap_optional(KbSearchResponse.model_fields["debug"].annotation) is RetrievalDebug
    assert _unwrap_optional(QAResponse.model_fields["debug"].annotation) is RetrievalDebug


def test_symbolic_query_suppresses_entity_branch_in_mix_mode(monkeypatch) -> None:
    class DummyEmbedder:
        def encode_query(self, query: str) -> list[float]:
            return [0.0]

    class DummyStore:
        def query(self, *args: Any, **kwargs: Any) -> list[dict]:
            return []

    class DummyDB:
        def search_kb_chunks(self, *args: Any, **kwargs: Any) -> list[dict]:
            return []

        def search_entities(self, *args: Any, **kwargs: Any) -> list[dict]:
            raise AssertionError("entity lookup should be suppressed for symbolic queries")

        def get_related_entities(self, *args: Any, **kwargs: Any) -> list[dict]:
            raise AssertionError("related-entity lookup should be suppressed for symbolic queries")

        def search_entity_chunks(self, *args: Any, **kwargs: Any) -> list[dict]:
            raise AssertionError("entity chunk lookup should be suppressed for symbolic queries")

    class DummyCache:
        def get_retrieval(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_retrieval(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(retrieval_module, "get_embedder", lambda: DummyEmbedder())
    monkeypatch.setattr(retrieval_module, "get_store", lambda: DummyStore())
    monkeypatch.setattr(retrieval_module, "get_db", lambda: DummyDB())
    monkeypatch.setattr(retrieval_module, "get_cache", lambda: DummyCache())

    result = retrieval_module.retrieve_debug(
        query="backend/app/services/qa/pipeline.py",
        use_cache=False,
        retrieval_mode="mix",
        use_rerank=True,
        expand_neighbors=False,
        top_k=3,
        top_n=3,
    )

    assert result["hits"] == []
    assert result["debug"]["entity_count"] == 0
    assert result["debug"]["query_analysis"] is not None
    assert result["debug"]["query_analysis"]["query_type"] == "symbolic"
    assert result["debug"]["analysis_scope"] == "retrieval_query"


def test_cache_hit_debug_uses_cold_path_rerank_semantics(monkeypatch) -> None:
    cached_hits = [
        {
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "score": 0.9,
            "rerank_score": None,
            "platform": "ChatGPT",
            "title": "Example",
            "excerpt": "content",
            "path": "backend/app/services/qa/pipeline.py",
            "created_at": "2024-01-01",
            "url": None,
            "keyword_score": None,
            "fused_score": None,
            "entity_score": None,
            "role_summary": "",
            "message_range": "",
            "model_name": None,
            "tags": [],
            "entity_names": [],
            "turn_index": 0,
            "chunk_index": 0,
        }
    ]

    class DummyCache:
        def get_retrieval(self, *args: Any, **kwargs: Any) -> list[dict] | None:
            return cached_hits

        def set_retrieval(self, *args: Any, **kwargs: Any) -> None:
            return None

    class DummyEmbedder:
        def encode_query(self, query: str) -> list[float]:
            return [0.0]

    class DummyStore:
        def query(self, *args: Any, **kwargs: Any) -> list[dict]:
            return []

    class DummyDB:
        def search_kb_chunks(self, *args: Any, **kwargs: Any) -> list[dict]:
            return []

        def search_entities(self, *args: Any, **kwargs: Any) -> list[dict]:
            raise AssertionError("cache hit should prevent cold-path entity lookup")

        def get_related_entities(self, *args: Any, **kwargs: Any) -> list[dict]:
            raise AssertionError("cache hit should prevent cold-path entity lookup")

        def search_entity_chunks(self, *args: Any, **kwargs: Any) -> list[dict]:
            raise AssertionError("cache hit should prevent cold-path entity lookup")

    monkeypatch.setattr(retrieval_module, "get_cache", lambda: DummyCache())
    monkeypatch.setattr(retrieval_module, "get_embedder", lambda: DummyEmbedder())
    monkeypatch.setattr(retrieval_module, "get_store", lambda: DummyStore())
    monkeypatch.setattr(retrieval_module, "get_db", lambda: DummyDB())

    result = retrieval_module.retrieve_debug(
        query="backend/app/services/qa/pipeline.py",
        use_cache=True,
        retrieval_mode="mix",
        use_rerank=True,
        expand_neighbors=False,
        top_k=3,
        top_n=3,
    )

    assert result["debug"]["cache_hit"] is True
    assert result["debug"]["rerank_effective_mode"] == "off"
    assert result["debug"]["rerank_status"] == "skipped"
    assert result["debug"]["query_analysis"] is not None
    assert result["debug"]["query_analysis"]["query_type"] == "symbolic"
    assert result["debug"]["analysis_scope"] == "retrieval_query"


def test_graph_enabled_query_still_runs_entity_branch_without_neighbor_expansion(monkeypatch) -> None:
    class DummyEmbedder:
        def encode_query(self, query: str) -> list[float]:
            return [0.0]

    class DummyStore:
        def query(self, *args: Any, **kwargs: Any) -> list[dict]:
            return []

    class DummyDB:
        def search_kb_chunks(self, *args: Any, **kwargs: Any) -> list[dict]:
            return []

        def search_entities(self, *args: Any, **kwargs: Any) -> list[dict]:
            return [{"entity_id": "entity-1", "norm_name": "reranker"}]

        def get_related_entities(self, *args: Any, **kwargs: Any) -> list[dict]:
            return []

        def search_entity_chunks(self, *args: Any, **kwargs: Any) -> list[dict]:
            return [
                {
                    "chunk_id": "chunk-entity-1",
                    "doc_id": "doc-entity-1",
                    "score": 0.8,
                    "rerank_score": None,
                    "platform": "ChatGPT",
                    "title": "Entity Example",
                    "excerpt": "content",
                    "path": "backend/app/services/rerank/reranker.py",
                    "created_at": "2024-01-01",
                    "url": None,
                    "keyword_score": None,
                    "fused_score": None,
                    "entity_score": 0.8,
                    "role_summary": "",
                    "message_range": "",
                    "model_name": None,
                    "tags": [],
                    "entity_names": ["reranker"],
                    "turn_index": 0,
                    "chunk_index": 0,
                }
            ]

    class DummyCache:
        def get_retrieval(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_retrieval(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(retrieval_module, "get_embedder", lambda: DummyEmbedder())
    monkeypatch.setattr(retrieval_module, "get_store", lambda: DummyStore())
    monkeypatch.setattr(retrieval_module, "get_db", lambda: DummyDB())
    monkeypatch.setattr(retrieval_module, "get_cache", lambda: DummyCache())

    result = retrieval_module.retrieve_debug(
        query="Which module depends on the reranker?",
        use_cache=False,
        retrieval_mode="mix",
        use_rerank=True,
        expand_neighbors=False,
        top_k=3,
        top_n=3,
    )

    assert result["debug"]["query_analysis"] is not None
    assert result["debug"]["query_analysis"]["query_type"] == "relation"
    assert result["debug"]["analysis_scope"] == "retrieval_query"
    assert result["debug"]["entity_count"] == 1
    assert result["debug"]["candidate_count"] == 1

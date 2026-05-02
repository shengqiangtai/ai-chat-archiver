from __future__ import annotations

import types
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from app.models.schemas import RetrievalHit
from app.services.qa.grounding import evaluate_grounding
from app.services.qa import pipeline as pipeline_module


def test_grounding_marks_answer_supported_when_cited_text_overlaps() -> None:
    hits = [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=0.8,
            rerank_score=None,
            platform="ChatGPT",
            title="Title",
            excerpt="Reranker timeout falls back to original ordering.",
            path="AI-Chats/test.md",
            created_at="2026-01-01",
        )
    ]

    result = evaluate_grounding(
        answer="The reranker timeout falls back to original ordering.",
        hits=hits,
    )

    assert result.supported is True
    assert result.should_downgrade is False


def test_grounding_requests_downgrade_for_unsupported_claim() -> None:
    result = evaluate_grounding(
        answer="The system uses a graph database in production.",
        hits=[],
    )

    assert result.supported is False
    assert result.should_downgrade is True


def test_qa_answer_downgrades_unsupported_generation(monkeypatch) -> None:
    hits = [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=0.9,
            rerank_score=None,
            platform="ChatGPT",
            title="Title",
            excerpt="Reranker timeout falls back to original ordering.",
            path="AI-Chats/test.md",
            created_at="2026-01-01",
        )
    ]

    async def _rewrite_query(query: str, enable_llm: bool = True):
        return types.SimpleNamespace(rewritten_query=None, applied=False, strategy="off")

    class DummyCache:
        def get_answer(self, *args, **kwargs):
            return None

        def set_answer(self, *args, **kwargs):
            return None

    class DummyGenerator:
        async def generate(self, *args, **kwargs):
            return "The system uses a graph database in production."

    monkeypatch.setattr(pipeline_module, "rewrite_query", _rewrite_query)
    monkeypatch.setattr(pipeline_module, "retrieve", lambda **kwargs: hits)
    monkeypatch.setattr(pipeline_module, "get_cache", lambda: DummyCache())
    monkeypatch.setattr(pipeline_module, "get_generator", lambda: DummyGenerator())
    monkeypatch.setattr(pipeline_module, "unload_generator", lambda: None)
    async def _to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr(pipeline_module.asyncio, "to_thread", _to_thread, raising=False)

    result = pipeline_module.asyncio.run(
        pipeline_module.qa_answer(
            query="How does reranker timeout work?",
            include_debug=True,
        )
    )

    assert result.uncertainty == "现有来源对结论支撑不足，以下答案已降级为保守表述。"
    assert "以下是检索到的相关片段" in result.answer
    assert result.debug["grounding"]["should_downgrade"] is True


def test_qa_answer_skips_answer_cache_when_instruction_context_is_present(monkeypatch) -> None:
    hits = [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=0.9,
            rerank_score=None,
            platform="ChatGPT",
            title="Title",
            excerpt="The answer is based on a cached-looking source.",
            path="AI-Chats/test.md",
            created_at="2026-01-01",
        )
    ]

    async def _rewrite_query(query: str, enable_llm: bool = True):
        return types.SimpleNamespace(rewritten_query=None, applied=False, strategy="off")

    class DummyCache:
        get_answer_called = False
        set_answer_called = False

        def get_answer(self, *args, **kwargs):
            self.get_answer_called = True
            return {"answer": "cached answer", "citations": [], "sources": []}

        def set_answer(self, *args, **kwargs):
            self.set_answer_called = True

    class DummyGenerator:
        async def generate(self, *args, **kwargs):
            return "fresh answer"

    cache = DummyCache()

    monkeypatch.setattr(pipeline_module, "rewrite_query", _rewrite_query)
    monkeypatch.setattr(pipeline_module, "retrieve", lambda **kwargs: hits)
    monkeypatch.setattr(pipeline_module, "get_cache", lambda: cache)
    monkeypatch.setattr(pipeline_module, "get_generator", lambda: DummyGenerator())
    monkeypatch.setattr(pipeline_module, "unload_generator", lambda: None)

    async def _to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(pipeline_module.asyncio, "to_thread", _to_thread, raising=False)

    result = pipeline_module.asyncio.run(
        pipeline_module.qa_answer(
            query="Use a custom instruction",
            instruction_context="system: answer freshly",
        )
    )

    assert result.answer == "fresh answer"
    assert cache.get_answer_called is False
    assert cache.set_answer_called is False


def test_qa_answer_stream_can_raise_generation_errors(monkeypatch) -> None:
    hits = [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=0.9,
            rerank_score=None,
            platform="ChatGPT",
            title="Title",
            excerpt="The stream has enough support.",
            path="AI-Chats/test.md",
            created_at="2026-01-01",
        )
    ]

    async def _rewrite_query(query: str, enable_llm: bool = True):
        return types.SimpleNamespace(rewritten_query=None, applied=False, strategy="off")

    class FailingGenerator:
        async def generate_stream(self, *args, **kwargs):
            raise RuntimeError("stream failed")
            yield

    monkeypatch.setattr(pipeline_module, "rewrite_query", _rewrite_query)
    monkeypatch.setattr(pipeline_module, "retrieve", lambda **kwargs: hits)
    monkeypatch.setattr(pipeline_module, "get_generator", lambda: FailingGenerator())
    monkeypatch.setattr(pipeline_module, "unload_generator", lambda: None)

    async def _to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(pipeline_module.asyncio, "to_thread", _to_thread, raising=False)

    async def _consume():
        chunks = []
        async for chunk in pipeline_module.qa_answer_stream(
            query="stream failure",
            raise_generation_errors=True,
        ):
            chunks.append(chunk)
        return chunks

    try:
        pipeline_module.asyncio.run(_consume())
    except RuntimeError as err:
        assert str(err) == "stream failed"
    else:
        raise AssertionError("expected stream failure to be raised")

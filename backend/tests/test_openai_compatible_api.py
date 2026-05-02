from __future__ import annotations

import json
import importlib
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))


@dataclass
class DummyAnswer:
    answer: str
    citations: list
    uncertainty: str | None
    sources: list
    debug: dict


@pytest.fixture(autouse=True)
def isolated_runtime_config(monkeypatch, tmp_path):
    env_snapshot = {key: os.environ.get(key) for key in ("ARCHIVER_STORAGE_ROOT",)}
    monkeypatch.setenv("ARCHIVER_STORAGE_ROOT", str(tmp_path / "storage"))

    import app.core.config as config

    importlib.reload(config)
    if "app.services.llm.generator" in sys.modules:
        importlib.reload(sys.modules["app.services.llm.generator"])
    if "app.api.routes_docs" in sys.modules:
        importlib.reload(sys.modules["app.api.routes_docs"])
    yield
    for key, value in env_snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    importlib.reload(config)
    if "app.services.llm.generator" in sys.modules:
        importlib.reload(sys.modules["app.services.llm.generator"])
    if "app.api.routes_docs" in sys.modules:
        importlib.reload(sys.modules["app.api.routes_docs"])


def test_v1_models_returns_openai_shape() -> None:
    from app.main import app

    client = TestClient(app)
    response = client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "ai-chat-archiver-rag"
    assert data["data"][0]["object"] == "model"


def test_v1_chat_completions_non_streaming(monkeypatch) -> None:
    from app.api import routes_openai_compatible as route_module
    from app.main import app

    async def fake_qa_answer(**kwargs):
        assert kwargs["query"] == "What is indexed?"
        assert kwargs["instruction_context"] == (
            "system: Be concise.\n\n"
            "developer: Prefer cited answers."
        )
        assert kwargs["rewrite_query_enabled"] is False
        assert kwargs["raise_generation_errors"] is True
        return DummyAnswer(
            answer="Indexed answer",
            citations=[],
            uncertainty=None,
            sources=[],
            debug={},
        )

    monkeypatch.setattr(route_module, "qa_answer", fake_qa_answer)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ai-chat-archiver-rag",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "developer", "content": "Prefer cited answers."},
                {"role": "user", "content": "What is indexed?"},
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "ai-chat-archiver-rag"
    assert data["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Indexed answer",
    }
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_v1_chat_completions_non_streaming_uses_server_error(monkeypatch) -> None:
    from app.api import routes_openai_compatible as route_module
    from app.main import app

    async def fake_qa_answer(**kwargs):
        assert kwargs["query"] == "Trigger error"
        assert kwargs["rewrite_query_enabled"] is False
        assert kwargs["raise_generation_errors"] is True
        raise RuntimeError("generate failed")

    monkeypatch.setattr(route_module, "qa_answer", fake_qa_answer)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ai-chat-archiver-rag",
            "messages": [{"role": "user", "content": "Trigger error"}],
            "stream": False,
        },
    )

    assert response.status_code == 500
    data = response.json()
    assert data["error"]["type"] == "server_error"
    assert data["error"]["message"] == "generate failed"


def test_v1_chat_completions_requires_user_message() -> None:
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ai-chat-archiver-rag",
            "messages": [{"role": "system", "content": "Only system."}],
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "invalid_request_error"
    assert "user message" in data["error"]["message"].lower()


def test_v1_chat_completions_streaming(monkeypatch) -> None:
    from app.api import routes_openai_compatible as route_module
    from app.main import app

    async def fake_stream(**kwargs):
        assert kwargs["query"] == "Stream this"
        assert kwargs["instruction_context"] is None
        assert kwargs["rewrite_query_enabled"] is False
        assert kwargs["raise_generation_errors"] is True
        yield "Hel"
        yield "lo"

    monkeypatch.setattr(route_module, "qa_answer_stream", fake_stream)

    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "ai-chat-archiver-rag",
            "messages": [{"role": "user", "content": "Stream this"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        body = response.read().decode("utf-8")

    assert "data: [DONE]" in body
    chunks = [
        line.removeprefix("data: ")
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    decoded = [json.loads(chunk) for chunk in chunks]
    assert decoded[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert decoded[1]["choices"][0]["delta"] == {"content": "Hel"}
    assert decoded[2]["choices"][0]["delta"] == {"content": "lo"}
    assert decoded[-1]["choices"][0]["finish_reason"] == "stop"


def test_v1_chat_completions_streaming_hides_sources_marker(monkeypatch) -> None:
    from app.api import routes_openai_compatible as route_module
    from app.main import app

    async def fake_stream(**kwargs):
        assert kwargs["query"] == "Hide metadata"
        assert kwargs["instruction_context"] is None
        assert kwargs["rewrite_query_enabled"] is False
        assert kwargs["raise_generation_errors"] is True
        yield "Answer"
        yield "\n\n[SOURCES_JSON][]"

    monkeypatch.setattr(route_module, "qa_answer_stream", fake_stream)

    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "ai-chat-archiver-rag",
            "messages": [{"role": "user", "content": "Hide metadata"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        body = response.read().decode("utf-8")

    assert "Answer" in body
    assert "[SOURCES_JSON]" not in body


def test_v1_chat_completions_streaming_uses_error_payload(monkeypatch) -> None:
    from app.api import routes_openai_compatible as route_module
    from app.main import app

    async def fake_stream(**kwargs):
        assert kwargs["query"] == "Trigger error"
        assert kwargs["rewrite_query_enabled"] is False
        assert kwargs["raise_generation_errors"] is True
        raise RuntimeError("stream failed")
        yield

    monkeypatch.setattr(route_module, "qa_answer_stream", fake_stream)

    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "ai-chat-archiver-rag",
            "messages": [{"role": "user", "content": "Trigger error"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        body = response.read().decode("utf-8")

    chunks = [
        line.removeprefix("data: ")
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    decoded = [json.loads(chunk) for chunk in chunks]
    assert decoded[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert decoded[1]["error"]["type"] == "server_error"
    assert decoded[1]["error"]["message"] == "stream failed"
    assert "Error: stream failed" not in body
    assert "data: [DONE]" in body


def test_llm_status_includes_openai_compatible(monkeypatch) -> None:
    from app.api import routes_docs as route_module
    from app.main import app

    class FakeOpenAICompatibleGenerator:
        def __init__(self):
            self.model = "status-chat"

        async def is_available(self):
            return True

        async def list_models(self):
            return ["status-chat", "other-chat"]

    monkeypatch.setattr(
        route_module,
        "OpenAICompatibleGenerator",
        FakeOpenAICompatibleGenerator,
    )

    client = TestClient(app)
    response = client.get("/api/kb/llm/status")

    assert response.status_code == 200
    data = response.json()
    assert "openai_compatible" in data
    assert data["openai_compatible"]["available"] is True
    assert data["openai_compatible"]["models"] == ["status-chat", "other-chat"]
    assert data["openai_compatible"]["current_model"] == "status-chat"
    assert data["openai_compatible"]["base_url"]


def test_switch_backend_accepts_openai_compatible() -> None:
    from app.core.config import get_current_openai_compatible_model
    from app.services.llm.generator import get_generator
    from app.main import app

    provider = get_generator()
    old_generator = provider.get_openai_compatible()
    assert old_generator.model != "external-chat"

    client = TestClient(app)
    response = client.put(
        "/api/kb/llm/backend",
        json={"backend": "openai_compatible", "model": "external-chat"},
    )

    assert response.status_code == 200
    assert response.json()["current_backend"] == "openai_compatible"
    assert get_current_openai_compatible_model() == "external-chat"
    assert provider.get_openai_compatible() is not old_generator
    assert provider.get_openai_compatible().model == "external-chat"

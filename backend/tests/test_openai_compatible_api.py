from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

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

# OpenAI Compatible API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider-neutral OpenAI-compatible generation backend and expose `/v1/models` plus `/v1/chat/completions` for standard LLM clients.

**Architecture:** Keep the existing `/api/kb/*` APIs untouched. Add `openai_compatible` as a fourth `GeneratorProvider` backend for outbound model calls, then add a small FastAPI router that wraps the existing RAG QA pipeline in OpenAI-compatible response shapes. API keys stay environment-only.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, httpx, pytest, existing AI Chat Archiver backend modules.

---

## File Structure

- Modify `backend/app/core/config.py`: add OpenAI-compatible environment defaults, runtime config fields, getters, and setters. Do not persist API keys.
- Modify `backend/app/models/schemas.py`: add Pydantic request models for OpenAI-compatible chat messages and chat completion requests.
- Modify `backend/app/services/llm/generator.py`: add `OpenAICompatibleGenerator` and wire it into `GeneratorProvider`.
- Create `backend/app/api/routes_openai_compatible.py`: expose `/v1/models` and `/v1/chat/completions`, including SSE streaming.
- Modify `backend/app/main.py`: register the new router before static-file fallback.
- Modify `backend/app/api/routes_docs.py`: include `openai_compatible` in backend switching and status data.
- Create `backend/tests/test_openai_compatible_generator.py`: unit tests for outbound provider behavior.
- Create `backend/tests/test_openai_compatible_api.py`: API tests with mocked RAG pipeline.
- Modify `README.md`: document environment variables and client setup.
- Modify `docs/API.md`: document `/v1` endpoints.

---

### Task 1: Runtime Configuration

**Files:**
- Modify: `backend/app/core/config.py`
- Test: `backend/tests/test_openai_compatible_generator.py`

- [ ] **Step 1: Write failing config tests**

Create `backend/tests/test_openai_compatible_generator.py` with these initial tests:

```python
from __future__ import annotations

import importlib
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def test_openai_compatible_config_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARCHIVER_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.example.com/v1/")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "example-chat")
    monkeypatch.setenv("OPENAI_COMPAT_PROVIDER_NAME", "Example")

    import app.core.config as config

    config = importlib.reload(config)

    assert config.OPENAI_COMPAT_BASE_URL == "https://api.example.com/v1/"
    assert config.OPENAI_COMPAT_MODEL == "example-chat"
    assert config.OPENAI_COMPAT_PROVIDER_NAME == "Example"
    assert config.get_current_openai_compatible_base_url() == "https://api.example.com/v1/"
    assert config.get_current_openai_compatible_model() == "example-chat"
    assert config.get_current_openai_compatible_provider_name() == "Example"
    assert "OPENAI_COMPAT_API_KEY" not in config.load_runtime_config()


def test_openai_compatible_runtime_config_does_not_store_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARCHIVER_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "secret-key")

    import app.core.config as config

    config = importlib.reload(config)
    config.set_current_openai_compatible_config(
        base_url="https://api.changed.com/v1",
        model="changed-chat",
        provider_name="Changed",
    )

    data = config.load_runtime_config()
    assert data["openai_compatible_base_url"] == "https://api.changed.com/v1"
    assert data["openai_compatible_model"] == "changed-chat"
    assert data["openai_compatible_provider_name"] == "Changed"
    assert "secret-key" not in config.RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")
    assert "api_key" not in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
python -m pytest tests/test_openai_compatible_generator.py -q
```

Expected: FAIL because `OPENAI_COMPAT_BASE_URL` and the getter/setter functions do not exist.

- [ ] **Step 3: Implement config fields and helpers**

In `backend/app/core/config.py`, add after the LM Studio config block:

```python
# ── 通用 OpenAI 兼容 API ─────────────────────────────────────────────────
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:1234/v1")
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "")
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "ai-chat-archiver-rag")
OPENAI_COMPAT_PROVIDER_NAME = os.getenv("OPENAI_COMPAT_PROVIDER_NAME", "OpenAI Compatible")
```

Update `_default_runtime()`:

```python
def _default_runtime() -> dict[str, Any]:
    return {
        "last_index_time": None,
        "ollama_model": OLLAMA_MODEL,
        "lmstudio_model": LMSTUDIO_MODEL,
        "openai_compatible_base_url": OPENAI_COMPAT_BASE_URL,
        "openai_compatible_model": OPENAI_COMPAT_MODEL,
        "openai_compatible_provider_name": OPENAI_COMPAT_PROVIDER_NAME,
        "generator_backend": "lmstudio",
    }
```

Add helpers near the existing model getters:

```python
def get_current_openai_compatible_base_url() -> str:
    return str(load_runtime_config().get("openai_compatible_base_url") or OPENAI_COMPAT_BASE_URL)


def get_current_openai_compatible_model() -> str:
    return str(load_runtime_config().get("openai_compatible_model") or OPENAI_COMPAT_MODEL)


def get_current_openai_compatible_provider_name() -> str:
    return str(
        load_runtime_config().get("openai_compatible_provider_name")
        or OPENAI_COMPAT_PROVIDER_NAME
    )


def set_current_openai_compatible_config(
    base_url: str | None = None,
    model: str | None = None,
    provider_name: str | None = None,
) -> None:
    data = load_runtime_config()
    if base_url is not None:
        data["openai_compatible_base_url"] = base_url.strip()
    if model is not None:
        data["openai_compatible_model"] = model.strip()
    if provider_name is not None:
        data["openai_compatible_provider_name"] = provider_name.strip()
    data.pop("openai_compatible_api_key", None)
    data.pop("api_key", None)
    save_runtime_config(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd backend
python -m pytest tests/test_openai_compatible_generator.py -q
```

Expected: PASS for the two config tests.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/config.py backend/tests/test_openai_compatible_generator.py
git commit -m "feat: add openai compatible runtime config"
```

---

### Task 2: Outbound OpenAI-Compatible Generator

**Files:**
- Modify: `backend/app/services/llm/generator.py`
- Modify: `backend/tests/test_openai_compatible_generator.py`

- [ ] **Step 1: Add failing generator tests**

Append these tests to `backend/tests/test_openai_compatible_generator.py`:

```python
import json

import httpx
import pytest


@pytest.mark.asyncio
async def test_openai_compatible_generator_sends_bearer_auth(monkeypatch) -> None:
    from app.services.llm.generator import OpenAICompatibleGenerator

    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "answer from provider"}}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    generator = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1/",
        api_key="secret-key",
        model="example-chat",
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )

    answer = await generator.generate("Question", max_tokens=123, system_prompt="System")

    assert answer == "answer from provider"
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["payload"] == {
        "model": "example-chat",
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question"},
        ],
        "max_tokens": 123,
        "temperature": 0.3,
        "stream": False,
    }


@pytest.mark.asyncio
async def test_openai_compatible_generator_streams_delta_content() -> None:
    from app.services.llm.generator import OpenAICompatibleGenerator

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        body = (
            'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    generator = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1",
        api_key="",
        model="example-chat",
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )

    chunks = [
        chunk
        async for chunk in generator.generate_stream(
            "Question",
            max_tokens=50,
            system_prompt=None,
        )
    ]

    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_openai_compatible_generator_availability_uses_models_endpoint() -> None:
    from app.services.llm.generator import OpenAICompatibleGenerator

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.example.com/v1/models"
        return httpx.Response(200, json={"data": [{"id": "example-chat"}]})

    transport = httpx.MockTransport(handler)
    generator = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1",
        api_key="secret-key",
        model="example-chat",
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )

    assert await generator.is_available() is True
    assert await generator.list_models() == ["example-chat"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
python -m pytest tests/test_openai_compatible_generator.py -q
```

Expected: FAIL because `OpenAICompatibleGenerator` is not defined.

- [ ] **Step 3: Implement `OpenAICompatibleGenerator`**

In `backend/app/services/llm/generator.py`, import the new config values:

```python
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_BASE_URL,
    OPENAI_COMPAT_MODEL,
    get_current_openai_compatible_base_url,
    get_current_openai_compatible_model,
```

Add this class after `LMStudioGenerator`:

```python
class OpenAICompatibleGenerator:
    """Provider-neutral OpenAI-compatible chat completions client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        client_factory=None,
    ) -> None:
        self.base_url = (base_url or get_current_openai_compatible_base_url()).rstrip("/")
        self.api_key = OPENAI_COMPAT_API_KEY if api_key is None else api_key
        self.model = model or get_current_openai_compatible_model() or OPENAI_COMPAT_MODEL
        self._client_factory = client_factory or httpx.AsyncClient

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _messages(self, prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        max_tokens: int = CONCISE_MAX_TOKENS,
        system_prompt: str | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": self._messages(prompt, system_prompt),
            "max_tokens": max_tokens,
            "temperature": GENERATION_TEMPERATURE,
            "stream": False,
        }
        async with self._client_factory(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content") or "").strip()

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = CONCISE_MAX_TOKENS,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "messages": self._messages(prompt, system_prompt),
            "max_tokens": max_tokens,
            "temperature": GENERATION_TEMPERATURE,
            "stream": True,
        }
        async with self._client_factory(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    text = (line or "").strip()
                    if not text or not text.startswith("data:"):
                        continue
                    chunk_str = text[5:].strip()
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield str(content)

    async def is_available(self) -> bool:
        try:
            async with self._client_factory(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/models", headers=self._headers())
                return resp.status_code < 500
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        async with self._client_factory(timeout=20.0) as client:
            resp = await client.get(f"{self.base_url}/models", headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        models = data.get("data") or []
        return [str(m.get("id", "")) for m in models if isinstance(m, dict) and m.get("id")]
```

- [ ] **Step 4: Wire provider into `GeneratorProvider`**

In `GeneratorProvider.__init__`, add:

```python
self._openai_compatible: OpenAICompatibleGenerator | None = None
```

Add:

```python
def get_openai_compatible(self) -> OpenAICompatibleGenerator:
    if self._openai_compatible is None:
        self._openai_compatible = OpenAICompatibleGenerator()
    return self._openai_compatible
```

In `generate`, add before the LM Studio branch:

```python
if backend == "openai_compatible":
    compat = self.get_openai_compatible()
    try:
        return await compat.generate(prompt, max_tokens, system_prompt=system_prompt)
    except Exception as e:
        logger.warning("OpenAI 兼容 API 生成失败: %s", e)
```

In `generate_stream`, add before the LM Studio branch:

```python
if backend == "openai_compatible":
    compat = self.get_openai_compatible()
    try:
        async for token in compat.generate_stream(prompt, max_tokens, system_prompt=system_prompt):
            yield token
        return
    except Exception as e:
        logger.warning("OpenAI 兼容 API 流式生成失败: %s", e)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd backend
python -m pytest tests/test_openai_compatible_generator.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/llm/generator.py backend/tests/test_openai_compatible_generator.py
git commit -m "feat: add openai compatible generator"
```

---

### Task 3: Public `/v1` Request Models and Router

**Files:**
- Modify: `backend/app/models/schemas.py`
- Create: `backend/app/api/routes_openai_compatible.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_openai_compatible_api.py`

- [ ] **Step 1: Write failing API tests**

Create `backend/tests/test_openai_compatible_api.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
python -m pytest tests/test_openai_compatible_api.py -q
```

Expected: FAIL because the router and schemas do not exist.

- [ ] **Step 3: Add request schemas**

In `backend/app/models/schemas.py`, add near `QARequest`:

```python
class OpenAIChatMessage(BaseModel):
    role: str
    content: str | List[Dict[str, Any]]
    name: Optional[str] = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str = "ai-chat-archiver-rag"
    messages: List[OpenAIChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
```

- [ ] **Step 4: Create router implementation**

Create `backend/app/api/routes_openai_compatible.py`:

```python
"""OpenAI-compatible API surface for standard chat clients."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.schemas import OpenAIChatCompletionRequest
from app.services.qa.pipeline import qa_answer, qa_answer_stream

router = APIRouter(tags=["openai-compatible"])

DEFAULT_MODEL_ID = "ai-chat-archiver-rag"


def _now() -> int:
    return int(time.time())


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return ""


def _extract_query(data: OpenAIChatCompletionRequest) -> str | None:
    for message in reversed(data.messages):
        if message.role == "user":
            text = _content_to_text(message.content)
            if text:
                return text
    return None


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/v1/models")
def list_models():
    created = _now()
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_ID,
                "object": "model",
                "created": created,
                "owned_by": "ai-chat-archiver",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(data: OpenAIChatCompletionRequest):
    if not data.messages:
        return _error("messages must contain at least one user message")

    query = _extract_query(data)
    if not query:
        return _error("messages must contain at least one non-empty user message")

    model = data.model or DEFAULT_MODEL_ID
    completion_id = _completion_id()
    created = _now()

    if data.stream:
        return StreamingResponse(
            _chat_completion_stream(data, query, completion_id, created, model),
            media_type="text/event-stream",
        )

    result = await qa_answer(query=query)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(),
    }


async def _chat_completion_stream(
    data: OpenAIChatCompletionRequest,
    query: str,
    completion_id: str,
    created: int,
    model: str,
):
    del data
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    try:
        async for token in qa_answer_stream(query=query):
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": token}, "finish_reason": None}
                    ],
                }
            )
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
    except Exception as err:
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"Error: {err}"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    yield "data: [DONE]\n\n"
```

- [ ] **Step 5: Register router**

In `backend/app/main.py`, import:

```python
from app.api.routes_openai_compatible import router as openai_compatible_router
```

Register before static file handling:

```python
app.include_router(openai_compatible_router)
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
cd backend
python -m pytest tests/test_openai_compatible_api.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/schemas.py backend/app/api/routes_openai_compatible.py backend/app/main.py backend/tests/test_openai_compatible_api.py
git commit -m "feat: expose openai compatible chat api"
```

---

### Task 4: Backend Status and Switching API

**Files:**
- Modify: `backend/app/api/routes_docs.py`
- Test: `backend/tests/test_openai_compatible_api.py`

- [ ] **Step 1: Add failing backend-management tests**

Append to `backend/tests/test_openai_compatible_api.py`:

```python
def test_llm_status_includes_openai_compatible(monkeypatch) -> None:
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/kb/llm/status")

    assert response.status_code == 200
    data = response.json()
    assert "openai_compatible" in data
    assert data["openai_compatible"]["current_model"]
    assert data["openai_compatible"]["base_url"]


def test_switch_backend_accepts_openai_compatible() -> None:
    from app.main import app

    client = TestClient(app)
    response = client.put(
        "/api/kb/llm/backend",
        json={"backend": "openai_compatible", "model": "external-chat"},
    )

    assert response.status_code == 200
    assert response.json()["current_backend"] == "openai_compatible"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd backend
python -m pytest tests/test_openai_compatible_api.py::test_llm_status_includes_openai_compatible tests/test_openai_compatible_api.py::test_switch_backend_accepts_openai_compatible -q
```

Expected: FAIL because `routes_docs.py` only accepts `lmstudio`, `ollama`, and `transformers`.

- [ ] **Step 3: Update LLM management routes**

In `backend/app/api/routes_docs.py`, import:

```python
    get_current_openai_compatible_base_url,
    get_current_openai_compatible_model,
    get_current_openai_compatible_provider_name,
    set_current_openai_compatible_config,
```

Update `BackendSwitchRequest` comment:

```python
backend: str  # "lmstudio" | "ollama" | "transformers" | "openai_compatible"
```

In `api_llm_status`, add:

```python
"openai_compatible": {
    "available": bool(get_current_openai_compatible_base_url()),
    "models": [get_current_openai_compatible_model()],
    "current_model": get_current_openai_compatible_model(),
    "base_url": get_current_openai_compatible_base_url(),
    "provider_name": get_current_openai_compatible_provider_name(),
},
```

In `api_switch_backend`, update validation:

```python
if backend not in ("lmstudio", "ollama", "transformers", "openai_compatible"):
    raise HTTPException(
        status_code=400,
        detail="backend 必须为 lmstudio / ollama / transformers / openai_compatible",
    )
```

Update model handling:

```python
elif backend == "openai_compatible":
    set_current_openai_compatible_config(model=data.model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd backend
python -m pytest tests/test_openai_compatible_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes_docs.py backend/tests/test_openai_compatible_api.py
git commit -m "feat: manage openai compatible backend"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/API.md`

- [ ] **Step 1: Update README LLM backend table**

In `README.md`, update the LLM backend table to include:

```markdown
| `OpenAI Compatible` | DeepSeek、OpenRouter、SiliconFlow、vLLM、自建兼容服务等 | 使用云端或外部兼容服务 |
```

- [ ] **Step 2: Add README environment example**

Add an "OpenAI Compatible Providers" subsection near the LLM backend section:

```markdown
### OpenAI Compatible Providers

项目也可以调用任意 OpenAI 兼容的 `/v1/chat/completions` 服务。

```bash
export OPENAI_COMPAT_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_COMPAT_API_KEY="your-api-key"
export OPENAI_COMPAT_MODEL="deepseek-chat"
export OPENAI_COMPAT_PROVIDER_NAME="DeepSeek"
```

启动后切换生成后端：

```bash
curl -X PUT http://localhost:8765/api/kb/llm/backend \
  -H "Content-Type: application/json" \
  -d '{"backend":"openai_compatible","model":"deepseek-chat"}'
```

API key 只从环境变量读取，不会写入 `AI-Chats/config.json`。
```

- [ ] **Step 3: Add README client setup**

Add:

```markdown
### OpenAI-Compatible Local RAG API

标准 LLM 客户端可以把本项目作为 OpenAI 兼容服务使用：

- Base URL: `http://127.0.0.1:8765/v1`
- API Key: 任意非空字符串
- Model: `ai-chat-archiver-rag`

常用端点：

- `GET /v1/models`
- `POST /v1/chat/completions`

这些端点会调用本项目的知识库 RAG pipeline，而不是裸模型对话。
```

- [ ] **Step 4: Update docs/API.md**

Append:

```markdown
## OpenAI-Compatible API

Base URL for compatible clients: `http://127.0.0.1:8765/v1`

### Models

- `GET /v1/models`

Returns:

```json
{
  "object": "list",
  "data": [
    {
      "id": "ai-chat-archiver-rag",
      "object": "model",
      "created": 1777651200,
      "owned_by": "ai-chat-archiver"
    }
  ]
}
```

### Chat Completions

- `POST /v1/chat/completions`

Example:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-chat-archiver-rag",
    "messages": [
      {"role": "user", "content": "总结最近关于 rerank 的记录"}
    ],
    "stream": false
  }'
```

Streaming:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-chat-archiver-rag",
    "messages": [
      {"role": "user", "content": "总结最近关于 rerank 的记录"}
    ],
    "stream": true
  }'
```

Unsupported OpenAI fields such as tools, vision, and strict JSON schema output are ignored in the first compatible version.
```

- [ ] **Step 5: Commit**

```bash
git add README.md docs/API.md
git commit -m "docs: document openai compatible api"
```

---

### Task 6: Full Verification

**Files:**
- Verify only

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
cd backend
python -m pytest tests/test_openai_compatible_generator.py tests/test_openai_compatible_api.py -q
```

Expected: PASS.

- [ ] **Step 2: Run existing lightweight backend tests**

Run:

```bash
cd backend
python -m pytest tests/test_query_analysis.py tests/test_grounding.py tests/test_fusion.py -q
```

Expected: PASS.

- [ ] **Step 3: Check no API key was written**

Run:

```bash
rg -n "OPENAI_COMPAT_API_KEY|secret-key|your-api-key" AI-Chats/config.json backend/app docs README.md
```

Expected: only documentation/example code and config constant names appear; no real secret value is present in `AI-Chats/config.json`.

- [ ] **Step 4: Check current Git diff**

Run:

```bash
git status --short --branch
git log --oneline -6
```

Expected: task commits are on top of the current branch. Existing unrelated user changes may remain unstaged; do not revert them.

- [ ] **Step 5: Final commit if verification required doc/test adjustments**

If Task 6 required fixes, commit them:

```bash
git add <changed-files>
git commit -m "test: verify openai compatible api"
```

If no files changed, do not create an empty commit.

---

## Self-Review

- Spec coverage: Tasks 1-2 cover outbound external provider calls and API-key handling. Tasks 3-4 cover public `/v1/models`, `/v1/chat/completions`, streaming, invalid request handling, and backend management. Task 5 covers README and API docs. Task 6 covers verification.
- Placeholder scan: The plan contains no unresolved markers or unspecified implementation steps.
- Type consistency: The plan consistently uses `OpenAIChatMessage`, `OpenAIChatCompletionRequest`, `OpenAICompatibleGenerator`, and `openai_compatible`.

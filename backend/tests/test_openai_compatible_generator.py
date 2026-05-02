from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(autouse=True)
def reload_config_after_test():
    env_snapshot = {
        key: os.environ.get(key)
        for key in (
            "ARCHIVER_STORAGE_ROOT",
            "OPENAI_COMPAT_API_KEY",
            "OPENAI_COMPAT_BASE_URL",
            "OPENAI_COMPAT_MODEL",
            "OPENAI_COMPAT_PROVIDER_NAME",
        )
    }
    yield
    for key, value in env_snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    if "app.core.config" in sys.modules:
        importlib.reload(sys.modules["app.core.config"])


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


def test_openai_compatible_runtime_config_does_not_store_api_key(
    monkeypatch, tmp_path
) -> None:
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


def test_save_runtime_config_scrubs_api_keys(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARCHIVER_STORAGE_ROOT", str(tmp_path / "storage"))

    import app.core.config as config

    config = importlib.reload(config)
    config.save_runtime_config(
        {
            "api_key": "legacy-secret",
            "openai_compatible_api_key": "openai-secret",
            "OPENAI_COMPAT_API_KEY": "env-style-secret",
            "openai_compatible_base_url": "https://api.saved.com/v1",
        }
    )

    data = config.load_runtime_config()
    saved_text = config.RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")

    assert data["openai_compatible_base_url"] == "https://api.saved.com/v1"
    assert "api_key" not in data
    assert "openai_compatible_api_key" not in data
    assert "OPENAI_COMPAT_API_KEY" not in data
    assert "legacy-secret" not in saved_text
    assert "openai-secret" not in saved_text
    assert "env-style-secret" not in saved_text


def test_load_runtime_config_rewrites_stale_api_keys(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARCHIVER_STORAGE_ROOT", str(tmp_path / "storage"))

    import app.core.config as config

    config = importlib.reload(config)
    config.RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RUNTIME_CONFIG_PATH.write_text(
        json.dumps(
            {
                "api_key": "stale-legacy-secret",
                "openai_compatible_api_key": "stale-openai-secret",
                "OPENAI_COMPAT_API_KEY": "stale-env-style-secret",
                "openai_compatible_model": "stale-model",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    data = config.load_runtime_config()
    saved_text = config.RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")

    assert data["openai_compatible_model"] == "stale-model"
    assert "api_key" not in data
    assert "openai_compatible_api_key" not in data
    assert "OPENAI_COMPAT_API_KEY" not in data
    assert "stale-legacy-secret" not in saved_text
    assert "stale-openai-secret" not in saved_text
    assert "stale-env-style-secret" not in saved_text


def test_openai_compatible_generator_sends_bearer_auth() -> None:
    asyncio.run(_openai_compatible_generator_sends_bearer_auth())


async def _openai_compatible_generator_sends_bearer_auth() -> None:
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


def test_openai_compatible_generator_streams_delta_content() -> None:
    asyncio.run(_openai_compatible_generator_streams_delta_content())


async def _openai_compatible_generator_streams_delta_content() -> None:
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


def test_openai_compatible_generator_rejects_empty_choices() -> None:
    asyncio.run(_openai_compatible_generator_rejects_empty_choices())


async def _openai_compatible_generator_rejects_empty_choices() -> None:
    from app.services.llm.generator import OpenAICompatibleGenerator

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(handler)
    generator = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1",
        api_key="",
        model="example-chat",
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )

    with pytest.raises(RuntimeError, match="空 choices"):
        await generator.generate("Question")


def test_openai_compatible_generator_rejects_empty_stream() -> None:
    asyncio.run(_openai_compatible_generator_rejects_empty_stream())


async def _openai_compatible_generator_rejects_empty_stream() -> None:
    from app.services.llm.generator import OpenAICompatibleGenerator

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    transport = httpx.MockTransport(handler)
    generator = OpenAICompatibleGenerator(
        base_url="https://api.example.com/v1",
        api_key="",
        model="example-chat",
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )

    with pytest.raises(RuntimeError, match="流式响应为空"):
        async for _ in generator.generate_stream("Question"):
            pass


def test_openai_compatible_generator_availability_uses_models_endpoint() -> None:
    asyncio.run(_openai_compatible_generator_availability_uses_models_endpoint())


async def _openai_compatible_generator_availability_uses_models_endpoint() -> None:
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


def test_openai_compatible_generator_availability_rejects_client_errors() -> None:
    asyncio.run(_openai_compatible_generator_availability_rejects_client_errors())


async def _openai_compatible_generator_availability_rejects_client_errors() -> None:
    from app.services.llm.generator import OpenAICompatibleGenerator

    for status_code in (401, 404):
        async def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "https://api.example.com/v1/models"
            return httpx.Response(status_code)

        transport = httpx.MockTransport(handler)
        generator = OpenAICompatibleGenerator(
            base_url="https://api.example.com/v1",
            api_key="secret-key",
            model="example-chat",
            client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
        )

        assert await generator.is_available() is False


def test_generator_provider_dispatches_openai_compatible_generate(monkeypatch) -> None:
    asyncio.run(_generator_provider_dispatches_openai_compatible_generate(monkeypatch))


async def _generator_provider_dispatches_openai_compatible_generate(monkeypatch) -> None:
    import app.services.llm.generator as generator_module

    class FakeOpenAICompatible:
        async def generate(self, prompt, max_tokens, system_prompt=None):
            assert prompt == "Question"
            assert max_tokens == generator_module.CONCISE_MAX_TOKENS
            assert system_prompt == "System"
            return "provider answer"

    monkeypatch.setattr(generator_module, "get_generator_backend", lambda: "openai_compatible")

    provider = generator_module.GeneratorProvider()
    provider._openai_compatible = FakeOpenAICompatible()

    assert await provider.generate("Question", system_prompt="System") == "provider answer"


def test_generator_provider_dispatches_openai_compatible_stream(monkeypatch) -> None:
    asyncio.run(_generator_provider_dispatches_openai_compatible_stream(monkeypatch))


async def _generator_provider_dispatches_openai_compatible_stream(monkeypatch) -> None:
    import app.services.llm.generator as generator_module

    class FakeOpenAICompatible:
        async def generate_stream(self, prompt, max_tokens, system_prompt=None):
            assert prompt == "Question"
            assert max_tokens == generator_module.CONCISE_MAX_TOKENS
            assert system_prompt == "System"
            yield "provider "
            yield "stream"

    monkeypatch.setattr(generator_module, "get_generator_backend", lambda: "openai_compatible")

    provider = generator_module.GeneratorProvider()
    provider._openai_compatible = FakeOpenAICompatible()

    chunks = [
        chunk
        async for chunk in provider.generate_stream("Question", system_prompt="System")
    ]

    assert chunks == ["provider ", "stream"]


def test_generator_provider_does_not_fallback_after_partial_openai_stream(
    monkeypatch,
) -> None:
    asyncio.run(
        _generator_provider_does_not_fallback_after_partial_openai_stream(monkeypatch)
    )


async def _generator_provider_does_not_fallback_after_partial_openai_stream(
    monkeypatch,
) -> None:
    import app.services.llm.generator as generator_module

    class FakeOpenAICompatible:
        async def generate_stream(self, prompt, max_tokens, system_prompt=None):
            del prompt, max_tokens, system_prompt
            yield "partial "
            raise RuntimeError("remote stream failed")

    class FakeTransformers:
        is_available = True

        def generate(self, prompt, max_tokens):
            del prompt, max_tokens
            return "fallback answer"

    monkeypatch.setattr(generator_module, "get_generator_backend", lambda: "openai_compatible")

    provider = generator_module.GeneratorProvider()
    provider._openai_compatible = FakeOpenAICompatible()
    provider._transformers = FakeTransformers()

    chunks: list[str] = []
    with pytest.raises(RuntimeError, match="remote stream failed"):
        async for chunk in provider.generate_stream("Question"):
            chunks.append(chunk)

    assert chunks == ["partial "]


def test_generator_provider_can_raise_openai_generate_errors(monkeypatch) -> None:
    asyncio.run(_generator_provider_can_raise_openai_generate_errors(monkeypatch))


async def _generator_provider_can_raise_openai_generate_errors(monkeypatch) -> None:
    import app.services.llm.generator as generator_module

    class FakeOpenAICompatible:
        async def generate(self, prompt, max_tokens, system_prompt=None):
            del prompt, max_tokens, system_prompt
            raise RuntimeError("remote generate failed")

    class FakeTransformers:
        is_available = True

        def generate(self, prompt, max_tokens):
            del prompt, max_tokens
            return "fallback answer"

    monkeypatch.setattr(generator_module, "get_generator_backend", lambda: "openai_compatible")

    provider = generator_module.GeneratorProvider()
    provider._openai_compatible = FakeOpenAICompatible()
    provider._transformers = FakeTransformers()

    with pytest.raises(RuntimeError, match="remote generate failed"):
        await provider.generate("Question", raise_backend_errors=True)


def test_generator_provider_can_raise_openai_stream_errors_before_tokens(
    monkeypatch,
) -> None:
    asyncio.run(
        _generator_provider_can_raise_openai_stream_errors_before_tokens(monkeypatch)
    )


async def _generator_provider_can_raise_openai_stream_errors_before_tokens(
    monkeypatch,
) -> None:
    import app.services.llm.generator as generator_module

    class FakeOpenAICompatible:
        async def generate_stream(self, prompt, max_tokens, system_prompt=None):
            del prompt, max_tokens, system_prompt
            raise RuntimeError("remote stream failed")
            yield

    class FakeTransformers:
        is_available = True

        def generate(self, prompt, max_tokens):
            del prompt, max_tokens
            return "fallback answer"

    monkeypatch.setattr(generator_module, "get_generator_backend", lambda: "openai_compatible")

    provider = generator_module.GeneratorProvider()
    provider._openai_compatible = FakeOpenAICompatible()
    provider._transformers = FakeTransformers()

    with pytest.raises(RuntimeError, match="remote stream failed"):
        async for _ in provider.generate_stream("Question", raise_backend_errors=True):
            pass


def test_generator_provider_strict_lmstudio_generate_errors(monkeypatch) -> None:
    asyncio.run(_generator_provider_strict_lmstudio_generate_errors(monkeypatch))


async def _generator_provider_strict_lmstudio_generate_errors(monkeypatch) -> None:
    import app.services.llm.generator as generator_module

    class FakeLMStudio:
        async def is_available(self):
            return False

    class FakeTransformers:
        is_available = True

        def generate(self, prompt, max_tokens):
            del prompt, max_tokens
            return "fallback answer"

    monkeypatch.setattr(generator_module, "get_generator_backend", lambda: "lmstudio")

    provider = generator_module.GeneratorProvider()
    provider._lmstudio = FakeLMStudio()
    provider._transformers = FakeTransformers()

    with pytest.raises(RuntimeError, match="LM Studio 不可用"):
        await provider.generate("Question", raise_backend_errors=True)

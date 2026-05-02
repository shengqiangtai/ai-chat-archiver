from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

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
            "openai_compatible_base_url": "https://api.saved.com/v1",
        }
    )

    data = config.load_runtime_config()
    saved_text = config.RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")

    assert data["openai_compatible_base_url"] == "https://api.saved.com/v1"
    assert "api_key" not in data
    assert "openai_compatible_api_key" not in data
    assert "legacy-secret" not in saved_text
    assert "openai-secret" not in saved_text


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
    assert "stale-legacy-secret" not in saved_text
    assert "stale-openai-secret" not in saved_text

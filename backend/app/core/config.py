"""集中配置模块 — 所有路径、模型、参数均可通过环境变量或 config.json 覆盖。"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

_cfg_logger = logging.getLogger("archiver.config")

# ── 基础路径 ──────────────────────────────────────────────────────────────
# backend/ 目录（config.py 在 backend/app/core/ 下，向上三层）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 整个仓库根目录（backend/ 的上一级），models/ 和 AI-Chats/ 目录放在这里
REPO_ROOT = PROJECT_ROOT.parent

# AI-Chats 默认存储在项目根目录 ai-chat-archiver/AI-Chats/，可通过环境变量覆盖
STORAGE_ROOT = Path(os.getenv("ARCHIVER_STORAGE_ROOT", str(REPO_ROOT / "AI-Chats")))

PORT = int(os.getenv("ARCHIVER_PORT", "8765"))
ALLOWED_ORIGINS: list[str] = ["*"]
DB_PATH = STORAGE_ROOT / "index.db"

# ── 模型路径（优先使用项目本地 models/ 目录，回退到 HuggingFace Hub）─────
# models/ 放在仓库根目录（ai-chat-archiver/models/），不在 backend/ 内
MODELS_DIR = REPO_ROOT / "models"


def _resolve_model_path(
    env_var: str,
    local_subdir: str,
    hub_repo: str,
    silent_fallback: bool = False,
) -> str:
    """
    模型路径解析优先级：
    1. 环境变量显式指定
    2. 项目本地 models/<local_subdir>（由 download_models.py 下载）
    3. HuggingFace Hub repo_id（兜底，会触发联网下载）

    silent_fallback=True：本地不存在时静默返回 hub_repo，不打印 WARNING。
    用于由外部服务（如 LM Studio）管理的模型，无需在本地存放。
    """
    # 优先级 1：环境变量
    env_val = os.getenv(env_var, "").strip()
    if env_val:
        p = Path(env_val)
        if p.exists():
            return str(p)
        _cfg_logger.warning("[config] 环境变量 %s=%s 指向的路径不存在，忽略", env_var, env_val)

    # 优先级 2：项目本地 models/ 目录
    local_path = MODELS_DIR / local_subdir
    if local_path.exists() and (local_path / "config.json").exists():
        _cfg_logger.debug("[config] 使用本地模型: %s", local_path)
        return str(local_path)

    # 优先级 3：HuggingFace Hub（兜底）
    if not silent_fallback:
        _cfg_logger.warning(
            "[config] 本地模型 %s 不存在，将使用 HuggingFace Hub: %s\n"
            "  请运行: cd backend && HF_ENDPOINT=https://hf-mirror.com python download_models.py",
            local_path,
            hub_repo,
        )
    return hub_repo


EMBEDDING_MODEL = _resolve_model_path(
    "EMBEDDING_MODEL",
    "Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-0.6B",
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

RERANKER_MODEL = _resolve_model_path(
    "RERANKER_MODEL",
    "Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-0.6B",
)

GENERATOR_MODEL = _resolve_model_path(
    "GENERATOR_MODEL",
    "Qwen3.5-0.8B",
    "Qwen/Qwen3.5-0.8B",
    silent_fallback=True,  # 默认后端为 LM Studio，无需本地模型文件
)

# 如果三个模型都已解析为本地绝对路径，全局开启离线模式，
# 防止 transformers / huggingface_hub 在任何环节尝试联网。
_all_local = all(
    Path(m).exists()
    for m in (EMBEDDING_MODEL, RERANKER_MODEL, GENERATOR_MODEL)
)
if _all_local:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _cfg_logger.debug("[config] 所有模型均在本地，已启用全局离线模式")

# ── 向量数据库 ────────────────────────────────────────────────────────────
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma")))

# ── 硬件感知：M1/M2 8GB 时自动降低负载 ──────────────────────────────────
def _is_low_memory() -> bool:
    """检测是否为低内存环境（< 12 GB），用于自动调整默认参数。"""
    try:
        import subprocess, re
        out = subprocess.check_output(["sysctl", "hw.memsize"], text=True)
        m = re.search(r"(\d+)", out)
        if m:
            return int(m.group(1)) < 12 * 1024 ** 3
    except Exception:
        pass
    return False


_LOW_MEM = os.getenv("LOW_MEMORY_MODE", "").lower() not in ("0", "false", "no") and _is_low_memory()
if _LOW_MEM:
    _cfg_logger.info("[config] 检测到低内存环境（<12 GB），已启用保守参数预设")

# ── 切块参数 ──────────────────────────────────────────────────────────────
CHUNK_TARGET_SIZE = int(os.getenv("CHUNK_TARGET_SIZE", "500" if _LOW_MEM else "700"))
CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE", "1000" if _LOW_MEM else "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80" if _LOW_MEM else "100"))
CHUNK_MIN_MERGE = int(os.getenv("CHUNK_MIN_MERGE", "80" if _LOW_MEM else "100"))

# ── 检索参数 ──────────────────────────────────────────────────────────────
# 低内存：top_k 10→减少 embedding 次数；rerank_top_n 4→减少 reranker 压力
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10" if _LOW_MEM else "15"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4" if _LOW_MEM else "6"))
DEFAULT_RERANK_MODE = str(os.getenv("RERANK_MODE", "auto")).strip().lower() or "auto"
if DEFAULT_RERANK_MODE not in {"auto", "off", "on"}:
    DEFAULT_RERANK_MODE = "auto"
RERANK_CANDIDATE_LIMIT = int(os.getenv("RERANK_CANDIDATE_LIMIT", "4" if _LOW_MEM else "6"))
RERANK_TIMEOUT_MS = int(os.getenv("RERANK_TIMEOUT_MS", "8000" if _LOW_MEM else "12000"))
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.30"))
# 低内存：缩短 context 窗口，减少 tokenize 开销
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "2000" if _LOW_MEM else "2800"))

# ── 生成参数 ──────────────────────────────────────────────────────────────
# 低内存：concise 模式 max_tokens 降低，减少生成时间和内存峰值
CONCISE_MAX_TOKENS = int(os.getenv("CONCISE_MAX_TOKENS", "200" if _LOW_MEM else "300"))
DETAILED_MAX_TOKENS = int(os.getenv("DETAILED_MAX_TOKENS", "400" if _LOW_MEM else "700"))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.3"))

# 低内存模式：生成完毕后自动卸载生成模型（释放 ~1.8GB），下次问答再重新加载
# 代价是首次生成有 ~10s 加载延迟，但避免内存耗尽导致整体卡死
UNLOAD_GENERATOR_AFTER_INFERENCE = (
    os.getenv("UNLOAD_GENERATOR_AFTER_INFERENCE", "1" if _LOW_MEM else "0").lower()
    not in ("0", "false", "no")
)

# ── LM Studio（默认推荐后端） ─────────────────────────────────────────────
# LM Studio 提供 OpenAI 兼容 API，默认 http://localhost:1234/v1
# 在 LM Studio 中加载模型后，后端通过此地址调用
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "")  # 空字符串 = 使用 LM Studio 当前加载的模型

# ── Ollama（备用后端） ────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# ── 缓存 ──────────────────────────────────────────────────────────────────
CACHE_DIR = DATA_DIR / "cache"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

# ── 运行时配置（可动态修改的部分持久化到 config.json） ─────────────────
RUNTIME_CONFIG_PATH = STORAGE_ROOT / "config.json"


def _default_runtime() -> dict[str, Any]:
    return {
        "last_index_time": None,
        "ollama_model": OLLAMA_MODEL,
        "lmstudio_model": LMSTUDIO_MODEL,
        "generator_backend": "lmstudio",  # 默认使用 LM Studio
    }


def load_runtime_config() -> dict[str, Any]:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    if not RUNTIME_CONFIG_PATH.exists():
        data = _default_runtime()
        save_runtime_config(data)
        return data
    try:
        data = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        data = _default_runtime()
        save_runtime_config(data)
        return data
    merged = _default_runtime()
    merged.update(data)
    return merged


def save_runtime_config(data: dict[str, Any]) -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    merged = _default_runtime()
    merged.update(data or {})
    RUNTIME_CONFIG_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_current_ollama_model() -> str:
    return str(load_runtime_config().get("ollama_model") or OLLAMA_MODEL)


def set_current_ollama_model(model_name: str) -> None:
    data = load_runtime_config()
    data["ollama_model"] = model_name
    save_runtime_config(data)


def get_current_lmstudio_model() -> str:
    return str(load_runtime_config().get("lmstudio_model") or LMSTUDIO_MODEL)


def set_current_lmstudio_model(model_name: str) -> None:
    data = load_runtime_config()
    data["lmstudio_model"] = model_name
    save_runtime_config(data)


def get_generator_backend() -> str:
    return str(load_runtime_config().get("generator_backend") or "lmstudio")


def set_generator_backend(backend: str) -> None:
    data = load_runtime_config()
    data["generator_backend"] = backend
    save_runtime_config(data)


def set_last_index_time(ts_iso: str) -> None:
    data = load_runtime_config()
    data["last_index_time"] = ts_iso
    save_runtime_config(data)

"""Embedding 模块 — 封装 Qwen3-Embedding-0.6B。

支持：
- 单例模式（模型只加载一次）
- 文档侧 / 查询侧区分编码（Instruction-aware）
- 批量编码
- 自动设备检测（CUDA / MPS / CPU）
- 完全离线加载，不联网检查更新

M1/M2 Mac 优化：
- 优先 MPS 加速
- float16 减少约 50% 内存
- 索引时 batch_size=8 避免 MPS 内存溢出
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import EMBEDDING_MODEL

logger = logging.getLogger("archiver.embedding")

QUERY_INSTRUCTION = (
    "Instruct: 从用户的AI对话记录知识库中，检索与以下问题最相关的对话片段\nQuery: "
)


def detect_device() -> str:
    """检测可用的计算设备。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Embedder:
    """Embedding 模型封装，单例模式。"""

    _instance: Embedder | None = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = detect_device()
        self._infer_lock = Lock()
        is_local = Path(EMBEDDING_MODEL).exists()
        logger.info("加载 Embedding 模型: %s (device=%s, local=%s)", EMBEDDING_MODEL, self.device, is_local)

        # 本地路径时强制离线，避免任何网络请求
        if is_local:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        # M1 MPS: 用 model_kwargs 指定 float16 减少内存
        model_kwargs: dict[str, Any] = {}
        if self.device in ("mps", "cuda"):
            try:
                import torch
                model_kwargs["torch_dtype"] = torch.float16
            except ImportError:
                pass

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            device=self.device,
            model_kwargs=model_kwargs if model_kwargs else None,
            **({"local_files_only": True} if is_local else {}),
        )
        logger.info("Embedding 模型加载完成 (device=%s)", self.device)

    @classmethod
    def get(cls) -> Embedder:
        """单例获取。"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _to_list(self, vecs: Any) -> list[list[float]]:
        if hasattr(vecs, "tolist"):
            return vecs.tolist()
        return [list(v) for v in vecs]

    def encode_docs(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """
        文档侧批量编码（索引 chunk 时使用）。
        batch_size=8 适合 M1 8GB，避免 MPS 内存溢出。
        """
        if not texts:
            return []
        all_vecs = []
        with self._infer_lock:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_vecs.extend(self._to_list(vecs))
                # 每批次后释放 MPS 缓存
                if self.device == "mps":
                    _flush_mps()
        return all_vecs

    def encode_query(self, query: str) -> list[float]:
        """
        查询侧编码（用户提问时使用）。
        自动拼接 QUERY_INSTRUCTION 前缀后再编码。
        """
        text = QUERY_INSTRUCTION + (query or "")
        with self._infer_lock:
            vector = self.model.encode(
                [text],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        rows = self._to_list(vector)
        return rows[0] if rows else []

    def get_dimension(self) -> int:
        """返回嵌入向量维度。"""
        with self._infer_lock:
            dummy = self.model.encode(["test"], normalize_embeddings=True, convert_to_numpy=True)
        return len(dummy[0]) if len(dummy) > 0 else 0


def _flush_mps() -> None:
    """释放 MPS 缓存，减少显存压力。"""
    try:
        import torch
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def get_embedder() -> Embedder:
    return Embedder.get()

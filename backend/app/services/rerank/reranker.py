"""Reranker 模块 — 封装 Qwen3-Reranker-0.6B。

对检索初筛结果进行精排，提升检索质量。
使用 cross-encoder 方式为 (query, document) 对打分。
完全离线加载，不联网检查更新。

M1/M2 Mac 优化：
- 优先 MPS 加速
- float16 减少内存约 50%
- 延迟加载：实例化时不加载模型，首次 rerank() 时才加载
- 单次处理 pair 数量上限，防止 MPS 内存溢出
- 加载失败时降级为向量分数排序（不卡死）
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from app.core.config import RERANKER_MODEL, RERANK_TOP_N
from app.models.schemas import RetrievalHit

logger = logging.getLogger("archiver.rerank")

# 单次 rerank 的最大 pair 数，防止 M1 MPS OOM
_MAX_RERANK_BATCH = 8


class Reranker:
    """Reranker 模型封装，单例 + 延迟加载。"""

    _instance: Reranker | None = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._loaded = False
        self._load_lock = Lock()
        self._infer_lock = Lock()
        # 延迟加载：不在 __init__ 里调用 _load_model()

    def _load_model(self) -> None:
        """按需加载（首次 rerank 时触发）。"""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                device = self._detect_device()
                is_local = Path(RERANKER_MODEL).exists()
                logger.info("加载 Reranker 模型: %s (device=%s, local=%s)", RERANKER_MODEL, device, is_local)

                # 本地路径时强制离线
                if is_local:
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

                extra = {"local_files_only": True} if is_local else {}
                dtype = torch.float16 if device != "cpu" else torch.float32

                self.tokenizer = AutoTokenizer.from_pretrained(
                    RERANKER_MODEL, trust_remote_code=True, **extra
                )
                if self.tokenizer.pad_token_id is None:
                    fallback_pad = self.tokenizer.eos_token or self.tokenizer.unk_token
                    if fallback_pad is not None:
                        self.tokenizer.pad_token = fallback_pad

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    RERANKER_MODEL,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    **extra,
                )
                if getattr(self.model.config, "pad_token_id", None) is None:
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id
                self.device = device
                if device != "cpu":
                    self.model = self.model.to(device)
                self.model.eval()
                self._loaded = True
                logger.info("Reranker 模型加载完成 (dtype=%s, device=%s)", dtype, device)
            except Exception as e:
                logger.error("Reranker 模型加载失败: %s，将使用向量分数排序", e)
                self.model = None
                self.tokenizer = None
                self._loaded = False

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def get(cls) -> Reranker:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_n: int = RERANK_TOP_N,
        timeout_ms: int | None = None,
    ) -> tuple[list[RetrievalHit], dict]:
        """
        对检索结果重排。
        - 如果模型可用：使用 cross-encoder 打分
        - 如果模型不可用/加载失败：直接用向量 similarity 排序（不卡死）
        返回按分数降序排列的 top_n 结果。
        """
        started_at = time.monotonic()
        meta = {
            "attempted": bool(hits),
            "applied": False,
            "fallback": False,
            "timed_out": False,
            "reason": "no_hits",
            "elapsed_ms": 0,
            "scored_candidates": 0,
        }
        if not hits:
            return [], meta

        top_n = max(1, top_n)
        original_hits = sorted(hits, key=lambda h: h.score or 0.0, reverse=True)

        # 按需加载
        self._load_model()

        # 模型不可用时降级为向量分数
        if self.model is None or self.tokenizer is None:
            logger.warning("Reranker 不可用，使用向量相似度排序")
            meta.update(
                {
                    "fallback": True,
                    "reason": "model_unavailable",
                    "elapsed_ms": round((time.monotonic() - started_at) * 1000, 1),
                    "scored_candidates": 0,
                }
            )
            return original_hits[:top_n], meta

        try:
            import torch

            prefix = "Instruct: 给定一个搜索查询，检索与查询相关的对话片段\n"
            deadline = (
                time.monotonic() + max(timeout_ms, 1) / 1000.0
                if timeout_ms is not None and timeout_ms > 0
                else None
            )

            # 分批处理，防止 MPS OOM；同时串行化推理，避免本地并发导致模型崩溃
            all_scores: list[float] = []
            with self._infer_lock:
                for i in range(0, len(hits), _MAX_RERANK_BATCH):
                    if deadline is not None and time.monotonic() >= deadline:
                        logger.warning("Rerank 超时，回退到原始检索排序")
                        meta.update(
                            {
                                "fallback": True,
                                "timed_out": True,
                                "reason": "timeout",
                                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 1),
                                "scored_candidates": len(all_scores),
                            }
                        )
                        return original_hits[:top_n], meta

                    batch_hits = hits[i : i + _MAX_RERANK_BATCH]
                    pairs = []
                    for hit in batch_hits:
                        text = hit.excerpt.strip()[:400]  # 截短减少内存
                        pairs.append([prefix + query, text])

                    inputs = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    if self.device != "cpu":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits.float()
                        # Qwen3-Reranker 输出 shape 可能是 [batch, 2]（二分类）或 [batch, 1] / [batch]
                        # 二分类时取 positive 类（[:,1]）的 logit 作为相关性分数
                        if logits.dim() == 2 and logits.shape[-1] == 2:
                            scores = logits[:, 1]
                        elif logits.dim() == 2 and logits.shape[-1] == 1:
                            scores = logits.squeeze(-1)
                        elif logits.dim() == 2:
                            scores = logits[:, 0]
                        else:
                            scores = logits
                        if scores.dim() == 0:
                            scores = scores.unsqueeze(0)
                        batch_scores = scores.cpu().tolist()

                    all_scores.extend(
                        batch_scores if isinstance(batch_scores, list) else [batch_scores]
                    )

                    # 每批次释放 MPS 缓存
                    del inputs, outputs, scores
                    if self.device == "mps":
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass

                    if deadline is not None and time.monotonic() >= deadline and i + _MAX_RERANK_BATCH < len(hits):
                        logger.warning("Rerank 超时，回退到原始检索排序")
                        meta.update(
                            {
                                "fallback": True,
                                "timed_out": True,
                                "reason": "timeout",
                                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 1),
                                "scored_candidates": len(all_scores),
                            }
                        )
                        return original_hits[:top_n], meta

            for i, hit in enumerate(hits):
                hit.rerank_score = float(all_scores[i]) if i < len(all_scores) else 0.0

            hits.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
            meta.update(
                {
                    "applied": True,
                    "reason": "ok",
                    "elapsed_ms": round((time.monotonic() - started_at) * 1000, 1),
                    "scored_candidates": len(all_scores),
                }
            )
            return hits[:top_n], meta

        except Exception as e:
            logger.error("Rerank 执行失败: %s，降级为向量分数排序", e)
            meta.update(
                {
                    "fallback": True,
                    "reason": f"error:{type(e).__name__}",
                    "elapsed_ms": round((time.monotonic() - started_at) * 1000, 1),
                    "scored_candidates": 0,
                }
            )
            return original_hits[:top_n], meta

    @property
    def is_available(self) -> bool:
        return Path(RERANKER_MODEL).exists() or (self.model is not None)


_reranker_instance: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker.get()
    return _reranker_instance

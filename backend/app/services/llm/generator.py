"""LLM 生成模块 — 支持 LM Studio / Ollama / transformers 三后端。

后端优先级（默认）：
1. LM Studio（OpenAI 兼容 API，推荐 Mac 用户使用）
2. Ollama（备用）
3. transformers 本地推理（最后兜底）

LM Studio 优势（相比 transformers 直接推理）：
- 内存管理由 LM Studio 负责，不占 Python 进程内存
- 支持 Metal GPU 加速，推理速度更快
- 支持 GGUF 量化模型，内存占用更低
- 有 GUI 可直观管理模型
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

import httpx

from app.core.config import (
    CONCISE_MAX_TOKENS,
    DETAILED_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GENERATOR_MODEL,
    LMSTUDIO_BASE_URL,
    OLLAMA_BASE_URL,
    get_current_lmstudio_model,
    get_current_ollama_model,
    get_generator_backend,
)

logger = logging.getLogger("archiver.llm")


# ═══════════════════════════════════════════════════════════════════════════
# LM Studio 后端（OpenAI 兼容 API）— 推荐
# ═══════════════════════════════════════════════════════════════════════════

class LMStudioGenerator:
    """通过 LM Studio 的 OpenAI 兼容 API 生成文本。

    LM Studio 默认运行在 http://localhost:1234/v1，
    提供 /chat/completions 和 /models 等端点。
    """

    def __init__(self, base_url: str = LMSTUDIO_BASE_URL, model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or get_current_lmstudio_model() or None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = CONCISE_MAX_TOKENS,
        system_prompt: str | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": GENERATION_TEMPERATURE,
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                return str(choices[0].get("message", {}).get("content", "")).strip()
            return ""

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = CONCISE_MAX_TOKENS,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": GENERATION_TEMPERATURE,
            "stream": True,
        }
        if self.model:
            payload["model"] = self.model

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
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
                        yield content

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/models")
                return resp.status_code < 500
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/models")
                resp.raise_for_status()
                data = resp.json()
                models = data.get("data") or []
                return [str(m.get("id", "")) for m in models if m.get("id")]
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════
# Ollama 后端（备用）
# ═══════════════════════════════════════════════════════════════════════════

class OllamaGenerator:
    """Ollama 后端生成器。"""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or get_current_ollama_model()

    async def generate(self, prompt: str, max_tokens: int = CONCISE_MAX_TOKENS) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": GENERATION_TEMPERATURE,
            },
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return str(data.get("response") or "")

    async def generate_stream(self, prompt: str, max_tokens: int = CONCISE_MAX_TOKENS) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": GENERATION_TEMPERATURE,
            },
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    text = (line or "").strip()
                    if not text:
                        continue
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("response")
                    if token:
                        yield token
                    if obj.get("done"):
                        break

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.base_url)
                return response.status_code < 500
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            models = data.get("models") or []
            return [str(m.get("name")) for m in models if isinstance(m, dict) and m.get("name")]


# ═══════════════════════════════════════════════════════════════════════════
# transformers 本地推理（兜底，内存占用大）
# ═══════════════════════════════════════════════════════════════════════════

class TransformersGenerator:
    """使用 transformers 库进行本地推理。延迟加载模式。"""

    _instance: TransformersGenerator | None = None

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._loaded = False

    def _load_model(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.device = _detect_device()
            is_local = Path(GENERATOR_MODEL).exists()
            logger.info("加载生成模型: %s (device=%s)", GENERATOR_MODEL, self.device)

            if is_local:
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            extra = {"local_files_only": True} if is_local else {}
            dtype = torch.float16 if self.device != "cpu" else torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(
                GENERATOR_MODEL, trust_remote_code=True, **extra
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                GENERATOR_MODEL,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **extra,
            )
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("生成模型加载完成 (device=%s)", self.device)
        except Exception as e:
            logger.error("生成模型加载失败: %s", e)
            self.model = None
            self.tokenizer = None

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        import gc
        gc.collect()

    @classmethod
    def get(cls) -> TransformersGenerator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def generate(self, prompt: str, max_tokens: int = CONCISE_MAX_TOKENS) -> str:
        self._load_model()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("生成模型未加载")

        import torch
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536,
        )
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 512),
                temperature=GENERATION_TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                top_k=20,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @property
    def is_available(self) -> bool:
        return Path(GENERATOR_MODEL).exists() or (self.model is not None)


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


# ═══════════════════════════════════════════════════════════════════════════
# 统一生成器提供层
# ═══════════════════════════════════════════════════════════════════════════

class GeneratorProvider:
    """生成器统一提供层。

    后端选择优先级：lmstudio → ollama → transformers
    通过 config.json 中的 generator_backend 字段切换。
    """

    def __init__(self) -> None:
        self._lmstudio: LMStudioGenerator | None = None
        self._ollama: OllamaGenerator | None = None
        self._transformers: TransformersGenerator | None = None

    def get_lmstudio(self) -> LMStudioGenerator:
        if self._lmstudio is None:
            self._lmstudio = LMStudioGenerator()
        return self._lmstudio

    def get_ollama(self) -> OllamaGenerator:
        if self._ollama is None:
            self._ollama = OllamaGenerator()
        return self._ollama

    def get_transformers(self) -> TransformersGenerator:
        if self._transformers is None:
            self._transformers = TransformersGenerator.get()
        return self._transformers

    async def generate(
        self,
        prompt: str,
        mode: str = "concise",
        system_prompt: str | None = None,
    ) -> str:
        """根据配置选择后端生成文本。

        Args:
            prompt: 用户侧 prompt（对 LM Studio 作为 user message）
            mode: concise / detailed
            system_prompt: 可选 system prompt（LM Studio chat completions 会用到）
        """
        max_tokens = CONCISE_MAX_TOKENS if mode == "concise" else DETAILED_MAX_TOKENS
        backend = get_generator_backend()

        # 1. LM Studio（使用 chat completions，分离 system/user）
        if backend == "lmstudio":
            lm = self.get_lmstudio()
            if await lm.is_available():
                try:
                    return await lm.generate(prompt, max_tokens, system_prompt=system_prompt)
                except Exception as e:
                    logger.warning("LM Studio 生成失败: %s", e)
            else:
                logger.warning("LM Studio 不可用，尝试其他后端")

        # 2. Ollama
        if backend in ("ollama", "lmstudio"):
            ollama = self.get_ollama()
            if await ollama.is_available():
                try:
                    # Ollama /api/generate 只接受拼接 prompt
                    full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    return await ollama.generate(full, max_tokens)
                except Exception as e:
                    logger.warning("Ollama 生成失败: %s", e)

        # 3. transformers 兜底
        gen = self.get_transformers()
        if gen.is_available:
            import asyncio
            full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            return await asyncio.to_thread(gen.generate, full, max_tokens)

        raise RuntimeError("所有生成后端均不可用。请确认 LM Studio 已启动并加载了模型。")

    async def generate_stream(
        self,
        prompt: str,
        mode: str = "concise",
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成。LM Studio 和 Ollama 支持真流式。"""
        max_tokens = CONCISE_MAX_TOKENS if mode == "concise" else DETAILED_MAX_TOKENS
        backend = get_generator_backend()

        # 1. LM Studio 真流式
        if backend == "lmstudio":
            lm = self.get_lmstudio()
            if await lm.is_available():
                try:
                    async for token in lm.generate_stream(prompt, max_tokens, system_prompt=system_prompt):
                        yield token
                    return
                except Exception as e:
                    logger.warning("LM Studio 流式生成失败: %s", e)

        # 2. Ollama 真流式
        if backend in ("ollama", "lmstudio"):
            ollama = self.get_ollama()
            if await ollama.is_available():
                try:
                    full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    async for token in ollama.generate_stream(full, max_tokens):
                        yield token
                    return
                except Exception as e:
                    logger.warning("Ollama 流式生成失败: %s", e)

        # 3. transformers 模拟流式
        gen = self.get_transformers()
        if gen.is_available:
            import asyncio
            full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            text = await asyncio.to_thread(gen.generate, full, max_tokens)
            for word in text.split(" "):
                yield word + " "
                await asyncio.sleep(0.01)
            return

        yield "所有生成后端均不可用。请确认 LM Studio 已启动并加载了模型。"


_provider: GeneratorProvider | None = None


def get_generator() -> GeneratorProvider:
    global _provider
    if _provider is None:
        _provider = GeneratorProvider()
    return _provider


def unload_generator() -> None:
    """主动卸载 transformers 生成模型以释放内存。"""
    gen = TransformersGenerator.get()
    gen.unload()

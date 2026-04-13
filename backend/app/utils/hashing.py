"""哈希工具 — 文件级去重 & chunk 级去重。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """计算文件内容的哈希值（用于文件级去重和增量索引）。"""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def text_hash(text: str, algorithm: str = "sha256") -> str:
    """计算文本的哈希值（用于 chunk 级去重）。"""
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()


def short_id(text: str, length: int = 16) -> str:
    """生成短 ID（基于 sha1 前 N 位）。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]

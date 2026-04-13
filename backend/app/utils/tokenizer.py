"""轻量级 token 估算工具 — 用于控制上下文长度。"""

from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """
    粗略估算 token 数量。
    中文按每 1.5 字符约 1 token 估算，英文按空格分词后约 1.3 token/word。
    """
    if not text:
        return 0

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    non_chinese = re.sub(r"[\u4e00-\u9fff]", "", text)
    words = len(non_chinese.split())

    return int(chinese_chars / 1.5 + words * 1.3)


def trim_to_token_budget(text: str, max_tokens: int) -> str:
    """将文本修剪到 token 预算内。"""
    if estimate_tokens(text) <= max_tokens:
        return text

    ratio = max_tokens / max(estimate_tokens(text), 1)
    target_len = int(len(text) * ratio * 0.9)
    return text[:target_len]

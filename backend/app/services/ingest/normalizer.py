"""文本标准化 — 对切块前的文本做清洗和格式统一。"""

from __future__ import annotations

from app.utils.text_clean import clean_text, normalize_markdown_headings


def normalize_chunk_text(text: str) -> str:
    """
    对 chunk 文本做标准化处理：
    1. 清洗无意义 UI 文案
    2. 标准化 Markdown 标题
    3. 去除多余空行
    """
    result = clean_text(text)
    result = normalize_markdown_headings(result)
    return result.strip()


def build_chunk_prefix(platform: str, title: str, created_at: str) -> str:
    """为 chunk 内容添加来源前缀，帮助检索和生成时识别来源。"""
    return f"[来源: {platform} | {title} | {created_at}]"

"""文本清洗工具 — 适配聊天语料的清洗策略。"""

from __future__ import annotations

import re


_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)
_UI_NOISE = re.compile(
    r"^(复制|Copy|编辑|Edit|分享|Share|点赞|Like|收藏|Save|重新生成|Regenerate|"
    r"Continue generating|Stop generating|Try again|🔄|👍|👎|📋|⬆️|⬇️)$",
    re.MULTILINE | re.IGNORECASE,
)
_DECORATIVE_LINE = re.compile(r"^[─━═\-=*~·•]{4,}$", re.MULTILINE)
_HEADING_NORMALIZE = re.compile(r"^(#{1,6})\s*", re.MULTILINE)


def clean_text(text: str) -> str:
    """综合清洗：去除无意义内容，但保留对问答有用的角色标记和代码块。"""
    if not text:
        return ""

    result = _UI_NOISE.sub("", text)
    result = _DECORATIVE_LINE.sub("", result)
    result = _TRAILING_SPACES.sub("", result)
    result = _MULTI_BLANK.sub("\n\n", result)

    return result.strip()


def normalize_markdown_headings(text: str) -> str:
    """标准化 Markdown 标题格式：确保 # 后有一个空格。"""
    def _fix(m: re.Match) -> str:
        hashes = m.group(1)
        return f"{hashes} "
    return _HEADING_NORMALIZE.sub(_fix, text)


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """安全截断文本，尽量在句子或段落边界截断。"""
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    for sep in ["\n\n", "\n", "。", ".", "！", "!", "？", "?", " "]:
        pos = cut.rfind(sep)
        if pos > max_chars * 0.5:
            return cut[:pos] + suffix
    return cut + suffix


def extract_code_blocks(text: str) -> list[str]:
    """提取文本中的代码块。"""
    pattern = re.compile(r"```[\s\S]*?```", re.DOTALL)
    return pattern.findall(text)


def remove_role_markers_for_display(text: str) -> str:
    """移除角色标记用于显示（不影响检索用途）。"""
    result = re.sub(r"^(User:|Assistant:|System:)\s*", "", text, flags=re.MULTILINE)
    return result.strip()

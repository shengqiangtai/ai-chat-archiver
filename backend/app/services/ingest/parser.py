"""Markdown 解析器 — 将 chat.md 解析为结构化消息列表。

不做暴力字符切割，而是先解析出消息块级别结构，
保留 role / content / position / section_title 信息。
"""

from __future__ import annotations

import re
from typing import Optional

from app.models.schemas import ParsedMessage

_HEADING_RE = re.compile(r"^##\s+(.+)$")


def parse_chat_markdown(text: str) -> list[ParsedMessage]:
    """
    解析 chat.md 为消息列表。
    识别 ## 标题中的角色标记（👤 User / 🤖 Assistant / System）。
    """
    if not text:
        return []

    messages: list[ParsedMessage] = []
    current_role: Optional[str] = None
    current_title: Optional[str] = None
    buffer: list[str] = []
    position = 0

    for line in text.splitlines():
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            if current_role and buffer:
                content = "\n".join(buffer).strip()
                if content:
                    messages.append(ParsedMessage(
                        role=current_role,
                        content=content,
                        position=position,
                        section_title=current_title,
                    ))
                    position += 1

            heading_text = heading_match.group(1).strip()
            current_role = _detect_role(heading_text)
            current_title = heading_text
            buffer = []
            continue

        if current_role:
            if line.strip() == "---":
                continue
            buffer.append(line)

    if current_role and buffer:
        content = "\n".join(buffer).strip()
        if content:
            messages.append(ParsedMessage(
                role=current_role,
                content=content,
                position=position,
                section_title=current_title,
            ))

    return messages


def _detect_role(heading: str) -> str:
    """从标题文本推断角色。"""
    text = heading.lower()
    if "user" in text or "human" in text or "👤" in heading:
        return "user"
    if "assistant" in text or "ai" in text or "bot" in text or "🤖" in heading:
        return "assistant"
    if "system" in text or "🧩" in heading:
        return "system"
    return "mixed"


def group_into_turns(messages: list[ParsedMessage]) -> list[list[ParsedMessage]]:
    """
    将消息按问答轮次分组。
    一个 turn = 一个 user 消息 + 后续的 assistant 回复。
    """
    if not messages:
        return []

    turns: list[list[ParsedMessage]] = []
    current: list[ParsedMessage] = []

    for msg in messages:
        if msg.role == "user":
            if current:
                turns.append(current)
                current = []
            current.append(msg)
        else:
            if not current:
                current = [msg]
            else:
                current.append(msg)

    if current:
        turns.append(current)

    return turns

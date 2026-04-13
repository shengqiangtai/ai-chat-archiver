"""Prompt 构建器 — 为小模型 / LM Studio / Ollama 设计的严格约束 Prompt。

支持两种输出格式：
- 纯文本拼接（transformers / Ollama /api/generate 使用）
- system + user 分离（LM Studio chat completions 使用）
"""

from __future__ import annotations

from app.core.config import MAX_CONTEXT_LENGTH
from app.models.schemas import RetrievalHit

SYSTEM_PROMPT = """你是一个本地知识库问答助手。
你只能依据提供的检索片段回答。
如果检索片段不足以支持结论，请明确说"根据当前知识库内容，无法确定"。
回答时尽量简洁，并在相关句子后附上 [Source X]。
不要编造不存在的信息，不要引用未提供的来源。"""

OUTPUT_FORMAT_INSTRUCTION = """
请按以下 JSON 格式输出：
{"answer": "你的回答", "citations": [{"source_id": "1", "reason": "引用原因"}], "uncertainty": "不确定的部分，无则为null"}
如果无法生成有效JSON，直接用纯文本回答也可以。"""


def build_context(hits: list[RetrievalHit], max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """将检索结果拼接为上下文字符串，控制总长度。"""
    parts: list[str] = []
    total = 0

    for idx, hit in enumerate(hits, start=1):
        body = hit.excerpt.strip()
        if not body:
            continue

        remain = max_length - total
        if remain <= 0:
            break
        if len(body) > remain:
            body = body[:remain]

        source_header = f"[Source {idx}]\nplatform: {hit.platform}\ntitle: {hit.title}\npath: {hit.path}"
        entry = f"{source_header}\nexcerpt: {body}"
        parts.append(entry)
        total += len(body)

        if total >= max_length:
            break

    return "\n\n".join(parts)


def build_user_prompt(
    question: str,
    hits: list[RetrievalHit],
    mode: str = "concise",
) -> str:
    """构建 user 部分的 prompt（不含 system prompt）。"""
    contexts = build_context(hits)
    length_hint = "150~300字" if mode == "concise" else "300~700字"

    return f"""用户问题：
{question}

检索片段：
{contexts}

输出要求：
1. 简要回答（控制在{length_hint}内）
2. 依据来源列表
3. 如果有不确定性，单独说明
{OUTPUT_FORMAT_INSTRUCTION}"""


def build_qa_prompt(
    question: str,
    hits: list[RetrievalHit],
    mode: str = "concise",
) -> str:
    """构建完整拼接版 prompt（transformers / Ollama 使用）。"""
    user_part = build_user_prompt(question, hits, mode)
    return f"{SYSTEM_PROMPT}\n\n{user_part}"


def get_system_prompt() -> str:
    """获取 system prompt（LM Studio chat completions 使用）。"""
    return SYSTEM_PROMPT


def build_fallback_answer(hits: list[RetrievalHit]) -> str:
    """当 LLM 不可用时，生成纯检索结果摘要。"""
    if not hits:
        return "知识库中未找到相关记录。"

    lines = ["以下是检索到的相关片段："]
    for i, hit in enumerate(hits[:5], start=1):
        short = hit.excerpt.strip().replace("\n", " ")
        if len(short) > 150:
            short = short[:150] + "..."
        score = hit.rerank_score if hit.rerank_score is not None else hit.score
        lines.append(f"\n[Source {i}] {hit.platform} · {hit.title}（相关度: {score:.2f}）\n{short}")

    return "\n".join(lines)

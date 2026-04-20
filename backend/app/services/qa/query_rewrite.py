"""查询改写模块 — 处理指代、时间引用和冗余口语化提问。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.services.llm.generator import get_generator

logger = logging.getLogger("archiver.qa.query_rewrite")

_AMBIGUOUS_MARKERS = (
    "上次", "之前", "这次", "那个", "那次", "当时", "里面", "提到的", "说的", "聊的", "相关的对话",
)

_PREFIXES = (
    "请问", "帮我", "请帮我", "我想知道", "我想问", "你能告诉我", "能不能帮我", "帮忙", "麻烦",
    "帮我找一下", "帮我回忆一下", "帮我看看", "帮我总结一下",
)

_GENERIC_PHRASES = (
    "的对话内容", "的聊天内容", "那次对话", "那个对话", "那个聊天", "相关内容", "相关的内容",
    "之前讨论过的", "之前聊过的", "上次说过的", "帮我找一下", "帮我总结一下",
)

_REWRITE_SYSTEM_PROMPT = """你是知识库检索前的查询改写器。
你的任务是把用户口语化、带指代或时间引用的问题，改写成适合检索的独立查询。
只保留用户真正想找的主题、实体、技术名词和约束条件。
不要编造新信息，不要扩写答案。
输出 JSON：{"rewritten_query":"..."}"""


@dataclass
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    applied: bool
    strategy: str


def needs_query_rewrite(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    return any(marker in q for marker in _AMBIGUOUS_MARKERS)


def heuristic_rewrite(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""

    for prefix in _PREFIXES:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    for phrase in _GENERIC_PHRASES:
        q = q.replace(phrase, " ")

    q = re.sub(r"(我之前|我上次|我这次|之前|上次|这次|那个|那次|当时|里面|其中)", " ", q)
    q = re.sub(r"(关于)\s+", "", q)
    q = re.sub(r"[，。！？、]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q or (query or "").strip()


def _parse_llm_rewrite(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    try:
        data = json.loads(text)
        return str(data.get("rewritten_query") or "").strip()
    except Exception:
        pass

    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            return str(data.get("rewritten_query") or "").strip()
        except Exception:
            pass

    return text.splitlines()[0].strip()


def _is_valid_rewrite(original: str, rewritten: str) -> bool:
    if not rewritten:
        return False
    if len(rewritten) > max(120, len(original) * 2):
        return False
    return True


async def rewrite_query(query: str, enable_llm: bool = True) -> QueryRewriteResult:
    original = (query or "").strip()
    heuristic = heuristic_rewrite(original)

    if not original:
        return QueryRewriteResult("", "", False, "identity")

    if not needs_query_rewrite(original):
        return QueryRewriteResult(
            original_query=original,
            rewritten_query=heuristic or original,
            applied=(heuristic or original) != original,
            strategy="rule_based" if (heuristic or original) != original else "identity",
        )

    if enable_llm:
        prompt = f"""用户原问题：
{original}

请把它改写成适合在聊天知识库中检索的独立查询，尽量保留主题词、专有名词、技术名词和范围约束。
如果原问题已经足够明确，就原样返回。
"""
        try:
            raw = await get_generator().generate(
                prompt,
                mode="concise",
                system_prompt=_REWRITE_SYSTEM_PROMPT,
            )
            llm_query = heuristic_rewrite(_parse_llm_rewrite(raw))
            if _is_valid_rewrite(original, llm_query):
                return QueryRewriteResult(
                    original_query=original,
                    rewritten_query=llm_query,
                    applied=llm_query != original,
                    strategy="llm",
                )
        except Exception as err:
            logger.warning("LLM query rewrite 失败，降级为规则改写: %s", err)

    final_query = heuristic or original
    return QueryRewriteResult(
        original_query=original,
        rewritten_query=final_query,
        applied=final_query != original,
        strategy="rule_based" if final_query != original else "identity",
    )

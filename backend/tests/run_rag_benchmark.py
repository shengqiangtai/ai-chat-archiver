"""本地 RAG 检索基准脚本。

用途：
- 对固定问题集批量调用 `/api/kb/search`
- 对比 `vector / hybrid / entity / mix / mix+rerank`
- 输出每种模式的 Recall@5、HitRate@5、Recall@10、MRR@10 和平均耗时

默认基准集基于当前仓库里真实归档过的一条聊天：
`Codex Skills 安装指南`
后续应逐步替换为覆盖更多真实聊天的评测集。
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.evaluation.models import BenchmarkCase
from app.services.evaluation.reporting import build_evaluation_summary
from app.services.evaluation.runner import evaluate_retrieval_case
from app.services.qa.query_rewrite import rewrite_query
from app.services.vectorstore.retrieval import retrieve_debug


DEFAULT_FIXTURE = ROOT / "backend" / "tests" / "fixtures" / "rag_benchmark.json"


@dataclass
class ModeConfig:
    name: str
    retrieval_mode: str
    rerank_mode: str


MODES = [
    ModeConfig(name="vector", retrieval_mode="vector", rerank_mode="off"),
    ModeConfig(name="hybrid", retrieval_mode="hybrid", rerank_mode="off"),
    ModeConfig(name="entity", retrieval_mode="entity", rerank_mode="off"),
    ModeConfig(name="mix", retrieval_mode="mix", rerank_mode="off"),
    ModeConfig(name="mix_rerank", retrieval_mode="mix", rerank_mode="on"),
]


def load_cases(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("benchmark 文件必须是数组")
    cases: list[BenchmarkCase] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cases.append(
            BenchmarkCase(
                id=str(item["id"]),
                question=str(item["question"]),
                expected_chunk_ids=[str(chunk_id) for chunk_id in item.get("expected_chunk_ids") or []],
                question_type=str(item["question_type"]),
                difficulty=str(item["difficulty"]),
                source_type=str(item["source_type"]),
                requires_relation_reasoning=bool(item.get("requires_relation_reasoning", False)),
                requires_context_resolution=bool(item.get("requires_context_resolution", False)),
                notes=(str(item["notes"]) if item.get("notes") is not None else None),
            )
        )
    return cases


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_local_search(payload: dict[str, Any]) -> dict[str, Any]:
    rewrite = asyncio.run(
        rewrite_query(
            str(payload.get("query") or ""),
            enable_llm=bool(payload.get("rewrite_query", True)),
        )
    )
    query_for_retrieval = rewrite.rewritten_query or str(payload.get("query") or "")
    result = retrieve_debug(
        query=query_for_retrieval,
        top_k=int(payload.get("top_k") or 5),
        top_n=int(payload.get("top_k") or 5),
        platform_filter=payload.get("platform_filter"),
        model_filter=payload.get("model_filter"),
        tag_filter=payload.get("tag_filter"),
        date_from=payload.get("date_from"),
        date_to=payload.get("date_to"),
        score_threshold=float(payload.get("score_threshold") or 0.30),
        use_rerank=str(payload.get("rerank_mode") or "auto") != "off",
        retrieval_mode=str(payload.get("retrieval_mode") or "mix"),
        expand_neighbors=False,
        neighbor_turn_window=1,
        use_cache=True,
        rerank_mode=str(payload.get("rerank_mode") or "auto"),
    )
    return {
        "hits": result["hits"],
        "total": len(result["hits"]),
        "debug": {
            **(result["debug"] or {}),
            "original_query": payload.get("query"),
            "rewritten_query": rewrite.rewritten_query,
            "rewrite_applied": rewrite.applied,
            "rewrite_strategy": rewrite.strategy,
        },
    }


def run_mode(
    *,
    transport: str,
    base_url: str,
    mode: ModeConfig,
    cases: list[BenchmarkCase],
    top_k: int,
) -> dict[str, Any]:
    case_results = []
    errors: list[str] = []

    for case in cases:
        payload = {
            "query": case.question,
            "top_k": top_k,
            "retrieval_mode": mode.retrieval_mode,
            "rerank_mode": mode.rerank_mode,
            "include_debug": True,
            "rewrite_query": True,
        }
        try:
            started_at = time.monotonic()
            if transport == "http":
                response = post_json(f"{base_url.rstrip('/')}/api/kb/search", payload)
            else:
                response = run_local_search(payload)
            ranked_chunk_ids = [hit.get("chunk_id") for hit in (response.get("hits") or []) if isinstance(hit, dict)]
            elapsed_seconds = ((response.get("debug") or {}).get("total_time"))
            if elapsed_seconds is None:
                elapsed_seconds = round(time.monotonic() - started_at, 3)
            result = evaluate_retrieval_case(
                case=case,
                ranked_chunk_ids=ranked_chunk_ids,
                mode=mode.name,
                elapsed_seconds=float(elapsed_seconds) if elapsed_seconds is not None else None,
            )
            case_results.append(result)
        except (HTTPError, URLError, TimeoutError, ValueError) as err:
            errors.append(f"{case.id}: {err}")

    summary = build_evaluation_summary(mode=mode.name, cases=case_results)
    summary_dict = asdict(summary)
    summary_dict.update(
        {
            "retrieval_mode": mode.retrieval_mode,
            "rerank_mode": mode.rerank_mode,
            "total_cases": len(cases),
            "evaluated_cases": len(case_results),
            "errors": errors,
        }
    )
    return summary_dict


def format_summary(results: list[dict[str, Any]]) -> str:
    lines = [
        "mode         recall@5  hit@5    recall@10  mrr@10   avg-seconds  evaluated/errors",
        "-----------  --------  -------  ---------  -------  -----------  ----------------",
    ]
    for item in results:
        lines.append(
            f"{item['mode']:<11}  "
            f"{item['recall_at_5']:<8.3f}  "
            f"{item['hit_rate_at_5']:<7.3f}  "
            f"{item['recall_at_10']:<9.3f}  "
            f"{item['mrr_at_10']:<7.3f}  "
            f"{(item['avg_elapsed_seconds'] if item['avg_elapsed_seconds'] is not None else '-'):>11}  "
            f"{item['evaluated_cases']}/{len(item['errors'])}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行本地 RAG 检索 benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="后端地址")
    parser.add_argument("--transport", choices=["local", "http"], default="local", help="local=直接调用后端检索链路，http=通过 API 请求")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="benchmark 用例文件")
    parser.add_argument("--top-k", type=int, default=5, help="每次检索返回的候选数")
    parser.add_argument("--case-limit", type=int, default=0, help="仅跑前 N 个用例，0 表示全部")
    parser.add_argument("--mode", action="append", dest="modes", help="只跑指定 mode，可重复传入")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = parser.parse_args()

    fixture_path = args.fixture.resolve()
    if not fixture_path.exists():
        print(f"benchmark 文件不存在: {fixture_path}", file=sys.stderr)
        return 1

    cases = load_cases(fixture_path)
    if args.case_limit and args.case_limit > 0:
        cases = cases[: args.case_limit]
    if not cases:
        print("没有可执行的 benchmark 用例", file=sys.stderr)
        return 1

    selected_modes = MODES
    if args.modes:
        allowed = set(args.modes)
        selected_modes = [mode for mode in MODES if mode.name in allowed]
        if not selected_modes:
            print(f"没有匹配的 mode: {sorted(allowed)}", file=sys.stderr)
            return 1

    results: list[dict[str, Any]] = []
    for mode in selected_modes:
        results.append(
            run_mode(
                transport=args.transport,
                base_url=args.base_url,
                mode=mode,
                cases=cases,
                top_k=max(1, args.top_k),
            )
        )

    if args.json:
        print(
            json.dumps(
                {
                    "fixture": str(fixture_path),
                    "transport": args.transport,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"fixture: {fixture_path}")
        print(f"transport: {args.transport}")
        print(f"cases: {len(cases)}")
        print(format_summary(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

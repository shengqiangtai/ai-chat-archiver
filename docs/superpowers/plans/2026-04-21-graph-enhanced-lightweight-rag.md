# Graph-Enhanced Lightweight RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight local-first RAG system with a mature hybrid retrieval backbone, automatic graph-gated dual-level retrieval, offline evaluation, and citation/grounding-aware answering.

**Architecture:** Keep the current FastAPI + SQLite + Chroma + React stack, but separate the control plane (`query_analysis`, `fusion`, `graph retrieval`, `grounding`, `evaluation`) into focused modules. The default retrieval path stays stable and production-like, while graph-enhanced retrieval remains a gated augmentation that is benchmarked independently.

**Tech Stack:** Python, FastAPI, SQLite, Chroma, Pydantic, pytest, React, TypeScript, Vite

---

## File Structure

**Create:**
- `backend/app/services/evaluation/__init__.py`
- `backend/app/services/evaluation/models.py`
- `backend/app/services/evaluation/metrics.py`
- `backend/app/services/evaluation/runner.py`
- `backend/app/services/evaluation/reporting.py`
- `backend/app/services/retrieval/__init__.py`
- `backend/app/services/retrieval/query_analysis.py`
- `backend/app/services/retrieval/fusion.py`
- `backend/app/services/graph/__init__.py`
- `backend/app/services/graph/relation_extractor.py`
- `backend/app/services/graph/retrieval.py`
- `backend/app/services/qa/grounding.py`
- `backend/tests/test_evaluation_runner.py`
- `backend/tests/test_query_analysis.py`
- `backend/tests/test_fusion.py`
- `backend/tests/test_graph_retrieval.py`
- `backend/tests/test_grounding.py`
- `backend/tests/test_reporting.py`

**Modify:**
- `backend/app/models/schemas.py`
- `backend/app/db/sqlite.py`
- `backend/app/services/ingest/entity_extractor.py`
- `backend/app/services/vectorstore/retrieval.py`
- `backend/app/services/qa/pipeline.py`
- `backend/app/api/routes_search.py`
- `backend/app/api/routes_qa.py`
- `backend/tests/run_rag_benchmark.py`
- `backend/tests/fixtures/rag_benchmark.json`
- `frontend/src/types/index.ts`
- `frontend/src/api/client.ts`
- `frontend/src/pages/KnowledgeBaseQA.tsx`
- `frontend/src/components/AnswerCard.tsx`
- `frontend/src/components/SourcePreview.tsx`
- `docs/EVALUATION.md`
- `docs/ARCHITECTURE.md`

### Task 1: Build the Evaluation Backbone

**Files:**
- Create: `backend/app/services/evaluation/__init__.py`
- Create: `backend/app/services/evaluation/models.py`
- Create: `backend/app/services/evaluation/metrics.py`
- Create: `backend/app/services/evaluation/runner.py`
- Create: `backend/app/services/evaluation/reporting.py`
- Create: `backend/tests/test_evaluation_runner.py`
- Modify: `backend/tests/run_rag_benchmark.py`
- Modify: `backend/tests/fixtures/rag_benchmark.json`
- Modify: `docs/EVALUATION.md`

- [ ] **Step 1: Write the failing evaluation tests**

```python
# backend/tests/test_evaluation_runner.py
from app.services.evaluation.metrics import compute_recall_at_k, compute_mrr
from app.services.evaluation.models import BenchmarkCase, RetrievalEvalInput
from app.services.evaluation.runner import evaluate_retrieval_case


def test_compute_recall_at_k_hits_when_expected_chunk_is_present() -> None:
    assert compute_recall_at_k(
        expected_chunk_ids=["chunk-b"],
        ranked_chunk_ids=["chunk-a", "chunk-b", "chunk-c"],
        k=2,
    ) == 1.0


def test_compute_mrr_uses_first_relevant_rank() -> None:
    assert compute_mrr(
        expected_chunk_ids=["chunk-b"],
        ranked_chunk_ids=["chunk-a", "chunk-b", "chunk-c"],
        k=10,
    ) == 0.5


def test_evaluate_retrieval_case_keeps_relation_flags() -> None:
    case = BenchmarkCase(
        id="case-1",
        question="Which module uses the reranker?",
        expected_chunk_ids=["chunk-b"],
        question_type="relation",
        difficulty="medium",
        source_type="chat",
        requires_relation_reasoning=True,
        requires_context_resolution=False,
    )
    result = evaluate_retrieval_case(
        case=case,
        retrieval_input=RetrievalEvalInput(
            mode="hybrid_graph",
            ranked_chunk_ids=["chunk-a", "chunk-b", "chunk-c"],
            elapsed_seconds=0.42,
        ),
    )
    assert result.recall_at_5 == 1.0
    assert result.mrr_at_10 == 0.5
    assert result.requires_relation_reasoning is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && pytest tests/test_evaluation_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.evaluation'`

- [ ] **Step 3: Write the minimal evaluation modules and benchmark schema**

```python
# backend/app/services/evaluation/models.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkCase:
    id: str
    question: str
    expected_chunk_ids: list[str]
    question_type: str
    difficulty: str
    source_type: str
    requires_relation_reasoning: bool
    requires_context_resolution: bool
    notes: str | None = None


@dataclass
class RetrievalEvalInput:
    mode: str
    ranked_chunk_ids: list[str]
    elapsed_seconds: float
    graph_routed: bool = False


@dataclass
class RetrievalEvalResult:
    case_id: str
    mode: str
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    hit_rate_at_5: float
    graph_routed: bool
    requires_relation_reasoning: bool
    elapsed_seconds: float


@dataclass
class EvaluationSummary:
    mode: str
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    hit_rate_at_5: float
    graph_route_rate: float
    relation_win_rate: float
    avg_elapsed_seconds: float
    cases: list[RetrievalEvalResult] = field(default_factory=list)
```

```python
# backend/app/services/evaluation/metrics.py
from __future__ import annotations


def compute_recall_at_k(*, expected_chunk_ids: list[str], ranked_chunk_ids: list[str], k: int) -> float:
    expected = set(expected_chunk_ids)
    ranked = set(ranked_chunk_ids[:k])
    return 1.0 if expected & ranked else 0.0


def compute_hit_rate_at_k(*, expected_chunk_ids: list[str], ranked_chunk_ids: list[str], k: int) -> float:
    return compute_recall_at_k(expected_chunk_ids=expected_chunk_ids, ranked_chunk_ids=ranked_chunk_ids, k=k)


def compute_mrr(*, expected_chunk_ids: list[str], ranked_chunk_ids: list[str], k: int) -> float:
    expected = set(expected_chunk_ids)
    for index, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        if chunk_id in expected:
            return 1.0 / index
    return 0.0
```

```python
# backend/app/services/evaluation/runner.py
from __future__ import annotations

from app.services.evaluation.metrics import compute_hit_rate_at_k, compute_mrr, compute_recall_at_k
from app.services.evaluation.models import BenchmarkCase, RetrievalEvalInput, RetrievalEvalResult


def evaluate_retrieval_case(*, case: BenchmarkCase, retrieval_input: RetrievalEvalInput) -> RetrievalEvalResult:
    ranked = retrieval_input.ranked_chunk_ids
    return RetrievalEvalResult(
        case_id=case.id,
        mode=retrieval_input.mode,
        recall_at_5=compute_recall_at_k(expected_chunk_ids=case.expected_chunk_ids, ranked_chunk_ids=ranked, k=5),
        recall_at_10=compute_recall_at_k(expected_chunk_ids=case.expected_chunk_ids, ranked_chunk_ids=ranked, k=10),
        mrr_at_10=compute_mrr(expected_chunk_ids=case.expected_chunk_ids, ranked_chunk_ids=ranked, k=10),
        hit_rate_at_5=compute_hit_rate_at_k(expected_chunk_ids=case.expected_chunk_ids, ranked_chunk_ids=ranked, k=5),
        graph_routed=retrieval_input.graph_routed,
        requires_relation_reasoning=case.requires_relation_reasoning,
        elapsed_seconds=retrieval_input.elapsed_seconds,
    )
```

```python
# backend/tests/fixtures/rag_benchmark.json
[
  {
    "id": "kb-001",
    "question": "Codex skills should be loaded from which directory?",
    "expected_chunk_ids": ["fixture-chunk-skills"],
    "question_type": "exact_lookup",
    "difficulty": "easy",
    "source_type": "chat",
    "requires_relation_reasoning": false,
    "requires_context_resolution": false
  }
]
```

- [ ] **Step 4: Run the evaluation tests again**

Run: `cd backend && pytest tests/test_evaluation_runner.py -q`
Expected: PASS

- [ ] **Step 5: Commit the evaluation backbone**

```bash
git add backend/app/services/evaluation backend/tests/test_evaluation_runner.py backend/tests/run_rag_benchmark.py backend/tests/fixtures/rag_benchmark.json docs/EVALUATION.md
git commit -m "feat: add typed rag evaluation backbone"
```

### Task 2: Add Query Analysis and Routing Decisions

**Files:**
- Create: `backend/app/services/retrieval/__init__.py`
- Create: `backend/app/services/retrieval/query_analysis.py`
- Create: `backend/tests/test_query_analysis.py`
- Modify: `backend/app/models/schemas.py`
- Modify: `backend/app/services/vectorstore/retrieval.py`
- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Write the failing routing tests**

```python
# backend/tests/test_query_analysis.py
from app.services.retrieval.query_analysis import analyze_query


def test_exact_symbolic_query_skips_graph_and_rewrite() -> None:
    result = analyze_query("backend/app/services/qa/pipeline.py")
    assert result.query_type == "symbolic"
    assert result.enable_rewrite is False
    assert result.enable_graph is False


def test_relation_query_enables_graph() -> None:
    result = analyze_query("Which module depends on the reranker?")
    assert result.query_type == "relation"
    assert result.enable_graph is True
    assert result.enable_rerank is True


def test_follow_up_query_enables_rewrite() -> None:
    result = analyze_query("那它为什么会超时？")
    assert result.enable_rewrite is True
```

- [ ] **Step 2: Run the routing tests to verify they fail**

Run: `cd backend && pytest tests/test_query_analysis.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.retrieval'`

- [ ] **Step 3: Implement query analysis and extend debug typing**

```python
# backend/app/services/retrieval/query_analysis.py
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class QueryAnalysis:
    normalized_query: str
    query_type: str
    enable_rewrite: bool
    enable_rerank: bool
    enable_graph: bool
    reasons: list[str]


_PATH_RE = re.compile(r"(?:^|/)[\w.-]+\.(?:py|ts|tsx|md|json)$")


def analyze_query(query: str) -> QueryAnalysis:
    normalized = (query or "").strip()
    lowered = normalized.lower()
    reasons: list[str] = []

    if _PATH_RE.search(normalized) or "/" in normalized:
        reasons.append("symbolic_exact_match")
        return QueryAnalysis(normalized, "symbolic", False, False, False, reasons)

    if any(token in lowered for token in ["depends on", "relationship", "关联", "依赖", "which module"]):
        reasons.append("relation_question")
        return QueryAnalysis(normalized, "relation", False, True, True, reasons)

    if any(token in lowered for token in ["它", "那它", "这个", "that", "why does it"]):
        reasons.append("context_dependent")
        return QueryAnalysis(normalized, "follow_up", True, True, True, reasons)

    reasons.append("default_semantic")
    return QueryAnalysis(normalized, "semantic", False, True, False, reasons)
```

```python
# backend/app/models/schemas.py
from dataclasses import dataclass, field

@dataclass
class QueryAnalysisDebug:
    query_type: str
    enable_rewrite: bool
    enable_rerank: bool
    enable_graph: bool
    reasons: list[str] = field(default_factory=list)
```

```ts
// frontend/src/types/index.ts
export interface QueryAnalysisDebug {
  query_type: string
  enable_rewrite: boolean
  enable_rerank: boolean
  enable_graph: boolean
  reasons: string[]
}
```

- [ ] **Step 4: Run the routing tests again**

Run: `cd backend && pytest tests/test_query_analysis.py -q`
Expected: PASS

- [ ] **Step 5: Commit the routing layer**

```bash
git add backend/app/services/retrieval backend/tests/test_query_analysis.py backend/app/models/schemas.py frontend/src/types/index.ts
git commit -m "feat: add query analysis and routing decisions"
```

### Task 3: Extract Fusion Logic from Retrieval

**Files:**
- Create: `backend/app/services/retrieval/fusion.py`
- Create: `backend/tests/test_fusion.py`
- Modify: `backend/app/services/vectorstore/retrieval.py`
- Modify: `backend/app/models/schemas.py`

- [ ] **Step 1: Write the failing fusion tests**

```python
# backend/tests/test_fusion.py
from app.models.schemas import RetrievalHit
from app.services.retrieval.fusion import fuse_ranked_hits


def _hit(chunk_id: str, score: float, *, keyword: float | None = None, entity: float | None = None) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id="doc-1",
        score=score,
        rerank_score=None,
        keyword_score=keyword,
        fused_score=None,
        entity_score=entity,
        platform="ChatGPT",
        title="Title",
        excerpt="Excerpt",
        path="AI-Chats/test.md",
        created_at="2026-01-01",
    )


def test_keyword_exact_match_stays_ahead_of_graph_only_hit() -> None:
    ranked = fuse_ranked_hits(
        keyword_hits=[_hit("chunk-a", 0.9, keyword=0.9)],
        dense_hits=[_hit("chunk-b", 0.8)],
        graph_hits=[_hit("chunk-c", 0.95, entity=0.95)],
        query_type="symbolic",
    )
    assert ranked[0].chunk_id == "chunk-a"


def test_graph_score_is_capped_but_can_help_relation_query() -> None:
    ranked = fuse_ranked_hits(
        keyword_hits=[],
        dense_hits=[_hit("chunk-a", 0.45)],
        graph_hits=[_hit("chunk-b", 0.20, entity=0.95)],
        query_type="relation",
    )
    assert [hit.chunk_id for hit in ranked][:2] == ["chunk-b", "chunk-a"]
```

- [ ] **Step 2: Run the fusion tests to verify they fail**

Run: `cd backend && pytest tests/test_fusion.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.retrieval.fusion'`

- [ ] **Step 3: Implement fusion and wire it into retrieval**

```python
# backend/app/services/retrieval/fusion.py
from __future__ import annotations

from app.models.schemas import RetrievalHit


def _score_hit(hit: RetrievalHit, *, query_type: str) -> float:
    keyword = float(hit.keyword_score or 0.0)
    dense = float(hit.score or 0.0)
    graph = min(float(hit.entity_score or 0.0), 0.35 if query_type != "relation" else 0.55)
    exact_bonus = 0.40 if query_type == "symbolic" and keyword > 0 else 0.0
    return dense + (keyword * 0.8) + graph + exact_bonus


def fuse_ranked_hits(
    *,
    keyword_hits: list[RetrievalHit],
    dense_hits: list[RetrievalHit],
    graph_hits: list[RetrievalHit],
    query_type: str,
) -> list[RetrievalHit]:
    merged: dict[str, RetrievalHit] = {}

    for hit in [*dense_hits, *keyword_hits, *graph_hits]:
        current = merged.get(hit.chunk_id)
        if current is None:
            merged[hit.chunk_id] = hit
            continue
        current.keyword_score = max(current.keyword_score or 0.0, hit.keyword_score or 0.0) or None
        current.entity_score = max(current.entity_score or 0.0, hit.entity_score or 0.0) or None
        current.score = max(current.score, hit.score)

    ranked = list(merged.values())
    for hit in ranked:
        hit.fused_score = _score_hit(hit, query_type=query_type)
    ranked.sort(key=lambda item: item.fused_score or 0.0, reverse=True)
    return ranked
```

```python
# backend/app/services/vectorstore/retrieval.py
from app.services.retrieval.fusion import fuse_ranked_hits
from app.services.retrieval.query_analysis import analyze_query

analysis = analyze_query(user_query)
final_candidates = fuse_ranked_hits(
    keyword_hits=keyword_hits,
    dense_hits=dense_hits,
    graph_hits=[],
    query_type=analysis.query_type,
)
```

- [ ] **Step 4: Run the fusion tests again**

Run: `cd backend && pytest tests/test_fusion.py -q`
Expected: PASS

- [ ] **Step 5: Commit the fusion refactor**

```bash
git add backend/app/services/retrieval/fusion.py backend/app/services/vectorstore/retrieval.py backend/tests/test_fusion.py backend/app/models/schemas.py
git commit -m "feat: extract retrieval fusion logic"
```

### Task 4: Add Lightweight Graph Metadata and Ingestion Support

**Files:**
- Create: `backend/app/services/graph/__init__.py`
- Create: `backend/app/services/graph/relation_extractor.py`
- Modify: `backend/app/db/sqlite.py`
- Modify: `backend/app/services/ingest/entity_extractor.py`
- Create: `backend/tests/test_graph_retrieval.py`

- [ ] **Step 1: Write the failing graph-ingestion tests**

```python
# backend/tests/test_graph_retrieval.py
from app.services.graph.relation_extractor import extract_relations


def test_extract_relations_finds_dependency_pattern() -> None:
    relations = extract_relations(
        chunk_id="chunk-1",
        text="retrieval.py depends on reranker.py for final ordering.",
        entity_names=["retrieval.py", "reranker.py"],
    )
    assert relations[0]["relation_type"] == "depends_on"
    assert relations[0]["source_entity"] == "retrieval.py"
    assert relations[0]["target_entity"] == "reranker.py"
```

- [ ] **Step 2: Run the graph-ingestion tests to verify they fail**

Run: `cd backend && pytest tests/test_graph_retrieval.py::test_extract_relations_finds_dependency_pattern -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.graph'`

- [ ] **Step 3: Implement relation extraction and SQLite graph tables**

```python
# backend/app/services/graph/relation_extractor.py
from __future__ import annotations


def extract_relations(*, chunk_id: str, text: str, entity_names: list[str]) -> list[dict[str, str]]:
    lowered = text.lower()
    relations: list[dict[str, str]] = []
    if "depends on" in lowered and len(entity_names) >= 2:
        relations.append(
            {
                "chunk_id": chunk_id,
                "source_entity": entity_names[0],
                "target_entity": entity_names[1],
                "relation_type": "depends_on",
            }
        )
    if "used for" in lowered and len(entity_names) >= 2:
        relations.append(
            {
                "chunk_id": chunk_id,
                "source_entity": entity_names[0],
                "target_entity": entity_names[1],
                "relation_type": "used_for",
            }
        )
    return relations
```

```python
# backend/app/db/sqlite.py
def ensure_graph_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_relations (
            chunk_id TEXT NOT NULL,
            source_entity TEXT NOT NULL,
            target_entity TEXT NOT NULL,
            relation_type TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_graph_relations_source
        ON graph_relations(source_entity, relation_type)
        """
    )
```

```python
# backend/app/services/ingest/entity_extractor.py
from app.services.graph.relation_extractor import extract_relations

def extract_entities_and_relations_from_chunk(chunk) -> tuple[list[str], list[dict[str, str]]]:
    entities = [item.name for item in extract_entities_from_text(chunk.text)]
    relations = extract_relations(chunk_id=chunk.chunk_id, text=chunk.text, entity_names=entities)
    return entities, relations
```

- [ ] **Step 4: Run the graph-ingestion test again**

Run: `cd backend && pytest tests/test_graph_retrieval.py::test_extract_relations_finds_dependency_pattern -q`
Expected: PASS

- [ ] **Step 5: Commit the graph metadata groundwork**

```bash
git add backend/app/services/graph backend/app/db/sqlite.py backend/app/services/ingest/entity_extractor.py backend/tests/test_graph_retrieval.py
git commit -m "feat: add lightweight graph metadata ingestion"
```

### Task 5: Implement Graph-Gated Dual-Level Retrieval

**Files:**
- Create: `backend/app/services/graph/retrieval.py`
- Modify: `backend/app/services/vectorstore/retrieval.py`
- Modify: `backend/app/models/schemas.py`
- Modify: `backend/app/api/routes_search.py`
- Modify: `backend/tests/run_rag_benchmark.py`
- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Write the failing graph-routing retrieval tests**

```python
# backend/tests/test_graph_retrieval.py
from app.models.schemas import RetrievalHit
from app.services.graph.retrieval import merge_graph_candidates


def test_merge_graph_candidates_prioritizes_relation_hit_for_relation_query() -> None:
    dense_hit = RetrievalHit(
        chunk_id="chunk-dense",
        doc_id="doc-1",
        score=0.51,
        rerank_score=None,
        platform="ChatGPT",
        title="Dense",
        excerpt="dense",
        path="AI-Chats/test.md",
        created_at="2026-01-01",
    )
    graph_hit = RetrievalHit(
        chunk_id="chunk-graph",
        doc_id="doc-1",
        score=0.20,
        rerank_score=None,
        entity_score=0.92,
        platform="ChatGPT",
        title="Graph",
        excerpt="graph",
        path="AI-Chats/test.md",
        created_at="2026-01-01",
    )
    ranked = merge_graph_candidates(
        dense_hits=[dense_hit],
        graph_hits=[graph_hit],
        query_type="relation",
    )
    assert ranked[0].chunk_id == "chunk-graph"
```

- [ ] **Step 2: Run the graph-routing tests to verify they fail**

Run: `cd backend && pytest tests/test_graph_retrieval.py::test_merge_graph_candidates_prioritizes_relation_hit_for_relation_query -q`
Expected: FAIL with `ImportError` because `merge_graph_candidates` does not exist yet

- [ ] **Step 3: Implement graph retrieval, gating, and debug output**

```python
# backend/app/services/graph/retrieval.py
from __future__ import annotations

from app.models.schemas import RetrievalHit
from app.services.retrieval.fusion import fuse_ranked_hits


def merge_graph_candidates(*, dense_hits: list[RetrievalHit], graph_hits: list[RetrievalHit], query_type: str) -> list[RetrievalHit]:
    return fuse_ranked_hits(
        keyword_hits=[],
        dense_hits=dense_hits,
        graph_hits=graph_hits,
        query_type=query_type,
    )
```

```python
# backend/app/services/vectorstore/retrieval.py
from app.services.graph.retrieval import merge_graph_candidates

graph_hits: list[RetrievalHit] = []
graph_routed = analysis.enable_graph
if graph_routed:
    graph_hits = search_graph_candidates(
        query=user_query,
        query_entities=query_entities,
        top_k=max(top_k, 10),
    )

final_candidates = fuse_ranked_hits(
    keyword_hits=keyword_hits,
    dense_hits=dense_hits,
    graph_hits=graph_hits,
    query_type=analysis.query_type,
)

debug["graph_routed"] = graph_routed
debug["graph_hit_count"] = len(graph_hits)
debug["graph_hits"] = [asdict(hit) for hit in graph_hits]
debug["query_analysis"] = asdict(analysis)
```

```python
# backend/app/models/schemas.py
@dataclass
class RetrievalDebugState:
    graph_routed: bool
    graph_hit_count: int
    graph_hits: list[dict[str, Any]] = field(default_factory=list)
    query_analysis: dict[str, Any] = field(default_factory=dict)
```

```python
# backend/tests/run_rag_benchmark.py
MODES = [
    ModeConfig(name="vector", retrieval_mode="vector", rerank_mode="off"),
    ModeConfig(name="hybrid", retrieval_mode="hybrid", rerank_mode="off"),
    ModeConfig(name="mix", retrieval_mode="mix", rerank_mode="off"),
    ModeConfig(name="mix_rerank", retrieval_mode="mix", rerank_mode="on"),
    ModeConfig(name="mix_graph", retrieval_mode="mix", rerank_mode="auto"),
]
```

- [ ] **Step 4: Run the graph-routing tests again**

Run: `cd backend && pytest tests/test_graph_retrieval.py -q`
Expected: PASS

- [ ] **Step 5: Commit the graph-gated retrieval path**

```bash
git add backend/app/services/graph/retrieval.py backend/app/services/vectorstore/retrieval.py backend/app/models/schemas.py backend/app/api/routes_search.py backend/tests/run_rag_benchmark.py frontend/src/types/index.ts backend/tests/test_graph_retrieval.py
git commit -m "feat: add graph-gated dual-level retrieval"
```

### Task 6: Add Grounding Checks and Conservative QA Downgrade

**Files:**
- Create: `backend/app/services/qa/grounding.py`
- Create: `backend/tests/test_grounding.py`
- Modify: `backend/app/services/qa/pipeline.py`
- Modify: `backend/app/api/routes_qa.py`
- Modify: `backend/app/models/schemas.py`

- [ ] **Step 1: Write the failing grounding tests**

```python
# backend/tests/test_grounding.py
from app.models.schemas import RetrievalHit
from app.services.qa.grounding import evaluate_grounding


def test_grounding_marks_answer_supported_when_cited_text_overlaps() -> None:
    hits = [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=0.8,
            rerank_score=None,
            platform="ChatGPT",
            title="Title",
            excerpt="Reranker timeout falls back to original ordering.",
            path="AI-Chats/test.md",
            created_at="2026-01-01",
        )
    ]
    result = evaluate_grounding(
        answer="The reranker timeout falls back to original ordering.",
        hits=hits,
    )
    assert result.supported is True
    assert result.should_downgrade is False


def test_grounding_requests_downgrade_for_unsupported_claim() -> None:
    hits = []
    result = evaluate_grounding(
        answer="The system uses a graph database in production.",
        hits=hits,
    )
    assert result.supported is False
    assert result.should_downgrade is True
```

- [ ] **Step 2: Run the grounding tests to verify they fail**

Run: `cd backend && pytest tests/test_grounding.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.qa.grounding'`

- [ ] **Step 3: Implement grounding evaluation and integrate it into QA**

```python
# backend/app/services/qa/grounding.py
from __future__ import annotations

from dataclasses import dataclass
from app.models.schemas import RetrievalHit


@dataclass
class GroundingResult:
    supported: bool
    should_downgrade: bool
    support_rate: float
    message: str


def evaluate_grounding(*, answer: str, hits: list[RetrievalHit]) -> GroundingResult:
    answer_terms = {term.lower() for term in answer.split() if len(term) > 3}
    if not hits:
        return GroundingResult(False, True, 0.0, "no_supporting_hits")

    support_matches = 0
    for hit in hits:
        excerpt_terms = {term.lower().strip(".,:;()") for term in hit.excerpt.split()}
        if answer_terms & excerpt_terms:
            support_matches += 1

    support_rate = support_matches / len(hits)
    supported = support_rate >= 0.5
    return GroundingResult(supported, not supported, support_rate, "ok" if supported else "weak_support")
```

```python
# backend/app/services/qa/pipeline.py
from app.services.qa.grounding import evaluate_grounding

grounding = evaluate_grounding(answer=result.answer, hits=hits)
if grounding.should_downgrade:
    result.uncertainty = "现有来源对结论支撑不足，以下答案已降级为保守表述。"
    result.answer = build_fallback_answer(hits)

if include_debug:
    result.debug["grounding"] = {
        "supported": grounding.supported,
        "should_downgrade": grounding.should_downgrade,
        "support_rate": grounding.support_rate,
        "message": grounding.message,
    }
```

- [ ] **Step 4: Run the grounding tests again**

Run: `cd backend && pytest tests/test_grounding.py -q`
Expected: PASS

- [ ] **Step 5: Commit the grounding layer**

```bash
git add backend/app/services/qa/grounding.py backend/app/services/qa/pipeline.py backend/app/api/routes_qa.py backend/app/models/schemas.py backend/tests/test_grounding.py
git commit -m "feat: add grounding checks to qa pipeline"
```

### Task 7: Surface Retrieval Decisions in the Frontend and Reporting

**Files:**
- Create: `backend/tests/test_reporting.py`
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/pages/KnowledgeBaseQA.tsx`
- Modify: `frontend/src/components/AnswerCard.tsx`
- Modify: `frontend/src/components/SourcePreview.tsx`
- Modify: `backend/app/services/evaluation/reporting.py`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/EVALUATION.md`

- [ ] **Step 1: Write the failing reporting test and extend frontend debug types**

```python
# backend/tests/test_reporting.py
from app.services.evaluation.models import EvaluationSummary
from app.services.evaluation.reporting import format_markdown_summary


def test_markdown_summary_includes_graph_metrics() -> None:
    summary = EvaluationSummary(
        mode="hybrid_graph",
        recall_at_5=0.75,
        recall_at_10=0.85,
        mrr_at_10=0.61,
        hit_rate_at_5=0.75,
        graph_route_rate=0.40,
        relation_win_rate=0.67,
        avg_elapsed_seconds=0.53,
    )
    markdown = format_markdown_summary(summary)
    assert "Graph route rate" in markdown
    assert "Relation win rate" in markdown
```

- [ ] **Step 2: Run the reporting test to verify it fails**

Run: `cd backend && pytest tests/test_reporting.py -q`
Expected: FAIL because `format_markdown_summary()` does not yet include graph-specific metrics

- [ ] **Step 3: Implement the UI and reporting surfaces**

```tsx
// frontend/src/components/AnswerCard.tsx
{debug?.query_analysis && (
  <div className="debug-block">
    <strong>Query Type:</strong> {debug.query_analysis.query_type}
    <br />
    <strong>Graph Routed:</strong> {debug.graph_routed ? 'yes' : 'no'}
    <br />
    <strong>Grounding:</strong> {debug.grounding?.message ?? 'n/a'}
  </div>
)}
```

```ts
// frontend/src/types/index.ts
export interface RetrievalDebug {
  query_analysis?: {
    query_type: string
    enable_rewrite: boolean
    enable_rerank: boolean
    enable_graph: boolean
    reasons: string[]
  }
  graph_routed?: boolean
  graph_hit_count?: number
  graph_hits?: RetrievalHit[]
  grounding?: {
    supported: boolean
    should_downgrade: boolean
    support_rate: number
    message: string
  }
}
```

```python
# backend/app/services/evaluation/reporting.py
from __future__ import annotations

from app.services.evaluation.models import EvaluationSummary


def format_markdown_summary(summary: EvaluationSummary) -> str:
    return "\n".join(
        [
            f"## Mode: {summary.mode}",
            f"- Recall@5: {summary.recall_at_5:.3f}",
            f"- Recall@10: {summary.recall_at_10:.3f}",
            f"- MRR@10: {summary.mrr_at_10:.3f}",
            f"- Graph route rate: {summary.graph_route_rate:.3f}",
            f"- Relation win rate: {summary.relation_win_rate:.3f}",
            f"- Avg elapsed seconds: {summary.avg_elapsed_seconds:.3f}",
        ]
    )
```

- [ ] **Step 4: Run the reporting test and frontend build again**

Run: `cd backend && pytest tests/test_reporting.py -q`
Expected: PASS

Run: `cd frontend && npm run build`
Expected: PASS

- [ ] **Step 5: Commit the debug and reporting surfaces**

```bash
git add backend/tests/test_reporting.py frontend/src/types/index.ts frontend/src/api/client.ts frontend/src/pages/KnowledgeBaseQA.tsx frontend/src/components/AnswerCard.tsx frontend/src/components/SourcePreview.tsx backend/app/services/evaluation/reporting.py docs/ARCHITECTURE.md docs/EVALUATION.md
git commit -m "feat: expose graph and grounding debug state"
```

## Self-Review

### Spec Coverage

- Evaluation backbone: covered by Task 1 and Task 7
- Query analysis and routing: covered by Task 2
- Mature hybrid retrieval path: covered by Task 3
- Graph metadata and dual-level retrieval: covered by Task 4 and Task 5
- Citation and grounding-aware answering: covered by Task 6
- Debug and explainability: covered by Task 2, Task 5, and Task 7

### Placeholder Scan

- No `TBD`, `TODO`, or deferred placeholders remain in the task steps
- Every task includes exact file paths, commands, and explicit test targets

### Type Consistency

- `QueryAnalysis` is introduced in Task 2 and reused as `query_analysis` debug state later
- `RetrievalHit` remains the shared candidate structure across fusion, graph retrieval, QA, and UI
- `GroundingResult` is introduced in Task 6 and only referenced after that task

### Sequencing Check

- Task 1 establishes evaluation before retrieval changes
- Task 2 and Task 3 define the stable control plane and fusion layer
- Task 4 and Task 5 add graph retrieval after the baseline path is structured
- Task 6 adds answer trustworthiness after retrieval output is stable
- Task 7 surfaces the behavior and reporting at the end

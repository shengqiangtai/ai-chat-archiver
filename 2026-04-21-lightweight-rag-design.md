# Lightweight RAG Design

Date: 2026-04-21

## 1. Project Positioning

This project is a lightweight, local-first RAG system for AI chat archives and Markdown documents.

It is not intended to become a heavy enterprise knowledge platform. The target positioning is:

- algorithm/retrieval-oriented RAG project
- local retrieval with optional cloud generation
- benchmark-driven optimization loop
- strong focus on retrieval quality and answer trustworthiness

The final external narrative should be:

> A lightweight local RAG system for AI chat histories and Markdown documents, with an evaluation-driven optimization loop around hybrid retrieval, conditional rewrite, reranking, citations, and grounding checks to improve retrieval hit rate and answer trustworthiness.

## 2. Product Goal

The project should prove two things:

1. Retrieval quality can be improved beyond baseline strategies in a stable, measurable way.
2. Answers can become more trustworthy under citation and grounding constraints.

Success is not defined by number of features or number of data sources. Success is defined by:

- measurable retrieval gains on a fixed benchmark
- lower unsupported answer rate
- clear fallback behavior under local resource constraints
- debug visibility for retrieval and answer-generation decisions

## 3. Data Scope

The first-stage knowledge base only supports:

- AI chat records
- local Markdown documents

This scope is intentionally fixed to keep the project focused on retrieval optimization rather than ingestion breadth.

The following are explicitly out of scope for the MVP:

- PDF parsing
- web crawling
- Notion or external SaaS connectors
- multi-tenant data management

## 4. Runtime Strategy

The default runtime strategy is:

- local retrieval stack
- optional cloud generation through an OpenAI-compatible API

This means:

- indexing, metadata, keyword retrieval, dense retrieval, and most optimization logic remain local
- answer generation can use either a local model or a cloud model, but the retrieval system must not depend on cloud access to function

If generation is unavailable, the system should still be able to return retrieved passages and sources.

## 5. Architecture Overview

The architecture remains lightweight and close to the current codebase:

- `FastAPI` for backend APIs
- `SQLite` for metadata and FTS retrieval
- `Chroma` for dense vector retrieval
- `React + Vite + TypeScript` for UI and debug surfaces

The system is divided into six logical layers:

1. `Ingestion Layer`
   - parse AI chat records and Markdown documents
   - normalize content
   - chunk content
   - extract metadata and entities
   - support incremental indexing

2. `Index Layer`
   - store metadata and chunk records in `SQLite`
   - store embeddings in `Chroma`
   - preserve source linkage for later citations

3. `Retrieval Layer`
   - query analysis
   - keyword retrieval
   - vector retrieval
   - optional entity expansion
   - candidate fusion
   - conditional rerank

4. `Answer Layer`
   - assemble context from retrieved chunks
   - generate answers with citations
   - perform weak grounding checks
   - degrade to conservative answers when support is insufficient

5. `Evaluation Layer`
   - load benchmark samples
   - execute fixed retrieval strategies
   - compute retrieval and answer-trust metrics
   - export comparable experiment summaries

6. `Debug and Explain Layer`
   - expose why a retrieval path was chosen
   - show rewrite and rerank behavior
   - display citation support and answer downgrade signals

## 6. Core Retrieval Pipeline

The default retrieval pipeline should be:

`query -> query analysis -> candidate generation -> rerank gating -> context expansion -> answer with citations -> grounding checks`

### 6.1 Query Analysis

The system should classify queries with lightweight heuristics rather than heavy orchestration. At minimum it should distinguish:

- factual lookup
- file or command lookup
- chat history recall
- multi-fragment synthesis
- pronoun or context-dependent query
- time-reference query

The purpose of query analysis is to decide:

- whether rewrite should run
- whether entity expansion should run
- whether rerank should run

### 6.2 Candidate Generation

Candidate generation should combine:

- `SQLite FTS` for exact keyword and symbolic matches
- `Chroma` for semantic retrieval

The system should keep `mix` as the primary default mode, with explicit fusion behavior:

- keyword retrieval protects precise matches
- dense retrieval extends recall
- entity expansion is supplemental and must not dominate the ranking

### 6.3 Rewrite Gating

Rewrite is not a mandatory step. It should only run for high-value cases such as:

- pronoun-heavy follow-up queries
- time references that require clarification
- context-dependent questions that likely under-specify the target

Rewrite should be skipped for explicit queries involving:

- filenames
- commands
- model names
- other precise symbolic tokens

### 6.4 Rerank Gating

Reranking is a conditional enhancer, not a default dependency.

It should run only when:

- the candidate set is limited
- the query is ambiguous enough to justify reranking cost
- the reranker is available within configured time budget

If rerank fails or times out, the system should immediately fall back to the original candidate order.

### 6.5 Context Expansion

Context expansion should be data-type aware:

- chat records expand through adjacent turns with tight limits
- Markdown expands through local chunk neighborhood or same-document local context

Expansion must stay bounded to avoid irrelevant context pollution.

## 7. Answering and Trustworthiness

The answer layer must stay grounded in retrieved evidence.

### 7.1 Citation Rules

Answers should not be treated as successful outputs unless they include source references when evidence is available.

The system should:

- keep source linkage from chunks to original files or chat records
- attach citations to answer segments
- expose citation details in both API responses and the UI

### 7.2 Weak Grounding Checks

The first-stage grounding design is intentionally lightweight.

The system should check:

- whether an answer includes source-backed statements
- whether key claims have at least weak textual support in the cited chunks
- whether the answer is overconfident relative to the available evidence

If support is insufficient, the answer should degrade to a conservative response rather than fabricate certainty.

### 7.3 Conservative Degradation

When grounding support is weak, the system should:

- state uncertainty clearly
- point the user to relevant retrieved passages
- avoid strong unsupported conclusions

The default bias is: answer less, but answer with support.

## 8. Evaluation Framework

The evaluation system is the core of the project narrative.

### 8.1 Evaluation Goals

The project evaluates two layers separately:

1. retrieval quality
2. answer trustworthiness

These must not be mixed into a single vague "quality" number.

### 8.2 Benchmark Dataset

The benchmark dataset should be small, curated, and manually checkable.

Each sample should include:

- `question`
- `expected_source_ids` or expected chunk identifiers
- `question_type`
- `difficulty`
- optional notes for edge cases

The first benchmark version should cover:

- exact fact lookup
- multi-turn chat recall
- multi-fragment synthesis
- filename or command lookup
- pronoun or omitted-reference queries
- time-reference queries

### 8.3 Retrieval Metrics

The first-stage retrieval metrics should stay minimal and hard:

- `Recall@K`
- `MRR`
- `HitRate@K`

Priority metrics:

- `Recall@5`
- `MRR@10`

`nDCG@K` can be added later if graded relevance labels become available.

### 8.4 Answer Trustworthiness Metrics

The first-stage trustworthiness metrics should be lightweight and reproducible:

- `citation_coverage`
- `source_support_rate`
- `unsupported_answer_rate`
- `abstain_rate`

These metrics are designed to show that the system is less likely to produce unsupported answers.

### 8.5 Experimental Comparisons

The benchmark should compare at least these strategies:

- `keyword`
- `vector`
- `hybrid`
- `mix`
- `mix + rerank`
- `mix + rewrite gating`
- `mix + rewrite gating + rerank`
- `mix + rerank + grounding checks`

Important boundary:

- rewrite and rerank may affect retrieval quality
- grounding checks affect answer trustworthiness and must not be misrepresented as retrieval gains

### 8.6 Outputs

Evaluation outputs should include:

- CLI benchmark summary
- structured machine-readable results
- Markdown experiment summary for docs
- per-query debug visibility in the UI

## 9. Module Boundaries and Code Evolution

This project should evolve by clarifying boundaries rather than performing a broad rewrite.

### 9.1 Existing Core Boundaries to Preserve

- `ingest`
  - loading, normalization, chunking, metadata extraction, incremental indexing
- `embedding`
  - embedding interface and model adapters
- `vectorstore/retrieval`
  - candidate generation and fusion
- `rerank`
  - rerank execution and fallback behavior
- `qa`
  - rewrite, prompt context, citations, grounding checks, answer generation

### 9.2 New or Strengthened Boundaries

The following boundaries should be introduced or made more explicit:

- `backend/app/services/evaluation/`
  - benchmark loading
  - experiment execution
  - metrics
  - result reporting

- `backend/app/services/retrieval/query_analysis.py`
  - query typing
  - rewrite gating
  - entity expansion gating
  - rerank gating decisions

- `backend/app/services/qa/grounding.py`
  - weak support checks
  - conservative downgrade logic

- `backend/app/services/evaluation/reporting.py`
  - CLI and Markdown-friendly summaries

### 9.3 Frontend Responsibility

The frontend should stay focused on explainability, not orchestration.

It should show:

- whether rewrite was applied, skipped, or failed
- whether rerank was applied, skipped, timed out, or failed
- which candidates came from keyword, vector, or entity paths
- whether answer downgrade happened
- how citations map back to sources

## 10. Failure Handling and Performance Constraints

The system must degrade gracefully under local resource constraints.

### 10.1 Failure Handling

- if dense retrieval is unavailable, fall back to keyword retrieval
- if rewrite fails, keep the original query
- if rerank fails or times out, keep original candidate order
- if generation fails, return retrieved passages and citations when possible
- if entity expansion is noisy, keep it supplemental and bounded

No enhancement layer should become a single point of failure.

### 10.2 Performance Constraints

The project should explicitly optimize for ordinary personal hardware.

The design constraints are:

- rerank only on limited candidate sets
- rewrite only for high-value cases
- no heavy multi-agent reasoning pipeline
- incremental indexing preferred over full rebuilds
- benchmark runs must remain practical on a personal machine

### 10.3 Logging and Explainability

All degradations should be visible:

- backend logs should record the reason
- debug responses should expose the actual path taken

The system should avoid silent fallback behavior.

## 11. MVP Scope

The MVP includes:

- AI chat record ingestion
- Markdown ingestion
- incremental indexing
- keyword retrieval
- vector retrieval
- `mix` retrieval
- conditional rewrite
- conditional rerank
- citation generation
- weak grounding checks
- benchmark dataset and evaluation script
- debug visibility for retrieval and downgrade decisions

The MVP explicitly excludes:

- PDF support
- web data ingestion
- agent workflows
- GraphRAG
- heavy claim verification
- enterprise auth or permissions
- feature expansion unrelated to retrieval optimization

## 12. Delivery Phases

### Phase 1: Fix the Evaluation Baseline

Primary outputs:

- benchmark schema definition
- first curated benchmark set
- stable baseline comparisons
- retrieval metric outputs
- first trustworthiness metric outputs

This phase creates the measurement backbone for the entire project.

### Phase 2: Retrieval Optimization

Priority work:

- explicit query analysis and rewrite gating
- cleaner `mix` fusion behavior
- rerank gating and fallback behavior
- bounded context expansion
- stronger debug explainability

This phase should produce measurable improvements against the fixed benchmark.

### Phase 3: Answer Trustworthiness

Priority work:

- citation coverage checks
- weak source support checks
- unsupported answer downgrades
- frontend support-state visibility

This phase should reduce unsupported answers without introducing heavy verification infrastructure.

## 13. Resume and Demo Narrative

The project should be explainable in one sentence:

> Built a lightweight local RAG system for AI chat histories and Markdown documents, and established an evaluation-driven optimization loop around hybrid retrieval, reranking, citations, and grounding checks to improve retrieval accuracy and answer trustworthiness.

Representative resume bullets:

- Designed and implemented a lightweight local RAG system supporting incremental indexing, hybrid retrieval, and citation-based QA for AI chat histories and Markdown documents.
- Built an offline benchmark and retrieval evaluation framework using metrics such as `Recall@K` and `MRR` to compare keyword, vector, hybrid, rewrite, and rerank strategies.
- Improved answer trustworthiness through conditional query rewrite, bounded reranking, citation coverage, and grounding checks that reduce unsupported responses.

## 14. Risks and Open Questions

### 14.1 Main Risks

- benchmark size may be too small to make results convincing
- retrieval gains may be confused with answer-layer safeguards
- rewrite may hurt exact symbolic queries
- rerank may cost more than it helps on some query types
- grounding rules may become too strict and over-suppress answers

### 14.2 Open Questions

- what is the right initial benchmark size for credible but fast iteration
- whether Markdown and chat data should use different chunk strategies from the first implementation
- whether answer trustworthiness should remain binary or evolve into a small support-level scale
- how broad the first OpenAI-compatible generation interface should be

These questions affect implementation details but do not block the architecture.

## 15. Explicit Non-Goals

This project will not optimize for:

- enterprise deployment
- multi-tenant SaaS behavior
- broad connector coverage
- graph-based retrieval
- autonomous agent systems
- heavy online learning or auto-tuning
- heavyweight judge-model evaluation
- infrastructure complexity added for appearance rather than value

## 16. Final Design Summary

The project should remain a small, local-first RAG system, but become a stronger algorithm/retrieval case study by adding:

- a fixed offline benchmark
- explicit retrieval decision logic
- measurable retrieval comparisons
- citation-aware answering
- lightweight grounding checks
- graceful fallback behavior
- strong debug explainability

That combination is what turns the codebase from a useful local tool into a resume-ready RAG optimization project.

# Graph-Enhanced Lightweight RAG Design

Date: 2026-04-21

## 1. Project Positioning

This project is a lightweight, local-first, graph-enhanced RAG system for AI chat histories and Markdown documents.

Its target positioning is not a generic knowledge base product. It is a resume-ready RAG engineering and retrieval optimization project for LLM application, search, and algorithm-oriented roles.

The system should combine two qualities at the same time:

- mature production-style RAG architecture
- modern retrieval ideas inspired by systems such as LightRAG, RAGFlow, and hybrid retrieval frameworks

The final external narrative should be:

> Built a lightweight local-first RAG system for AI chat histories and Markdown documents, combining production-grade hybrid retrieval, conditional reranking, citations, and offline evaluation with a LightRAG-inspired dual-level graph-enhanced retrieval layer to improve retrieval accuracy and answer trustworthiness.

This project is explicitly designed to look mature enough for engineering discussion and advanced enough for algorithm and retrieval discussion.

## 2. Core Product Goal

The project must prove three things:

1. The system improves retrieval quality over strong baselines in a measurable way.
2. The system improves answer trustworthiness with citations and grounding constraints.
3. The system introduces a graph-enhanced dual-level retrieval layer that provides additional gains on the right query types without turning the whole system into a heavy GraphRAG platform.

Success is defined by:

- measurable gains on a fixed benchmark
- a clean comparison between retrieval variants
- explainable query routing and fallback behavior
- conservative answer behavior when evidence is insufficient
- a system architecture that is small enough to deploy locally but sophisticated enough to discuss in interviews

## 3. Design Principles

The design should follow these principles:

- local-first, not cloud-dependent retrieval
- strong default path before advanced path
- graph enhancement as controlled augmentation, not heavy infrastructure
- retrieval gains and answer trust gains measured separately
- every expensive optimization must be gated
- every intelligent behavior must be inspectable in debug output

The project must not become:

- a heavy multi-tenant SaaS
- a full GraphRAG research platform
- a complex agent workflow system
- an ingestion breadth contest

## 4. Scope

### 4.1 Supported Data

The first-stage knowledge base supports only:

- AI chat histories
- local Markdown documents

This scope is intentionally narrow so the project can go deep on retrieval quality, graph augmentation, evaluation, and answer grounding.

### 4.2 Explicit Non-Goals

The following are out of scope for the MVP and near-term roadmap:

- PDF parsing
- web crawling
- Notion and SaaS connectors
- GraphRAG community detection and graph summarization
- graph path reasoning agents
- enterprise permissions
- multi-tenant deployment
- online learning and auto-tuning loops
- heavyweight judge-model evaluation pipelines

## 5. Runtime Strategy

The runtime model should be:

- local indexing
- local keyword retrieval
- local dense retrieval
- local graph-enhanced retrieval signals
- optional local or cloud answer generation through an OpenAI-compatible interface

The retrieval stack must remain usable without cloud generation.

If generation is unavailable, the system should still return:

- retrieved passages
- source references
- debug explanations for why those passages were selected

## 6. System Narrative

The system should be described as a two-track retrieval architecture:

### 6.1 Mature Primary Track

This is the always-available production-style path:

- query analysis
- hybrid candidate generation from keyword and vector retrieval
- candidate fusion
- optional reranking
- context assembly
- answer generation with citations
- grounding checks

This track is inspired by mature production RAG systems and should be the operational backbone of the project.

### 6.2 Advanced Secondary Track

This is the LightRAG-inspired graph-enhanced path:

- entity and relation extraction during ingestion
- graph-aware concept retrieval at query time
- chunk-level retrieval in parallel
- dual-level fusion between concept/entity-level signals and chunk-level evidence
- automatic gating so graph retrieval only activates when likely to help

This track exists to add a modern retrieval research flavor without breaking the lightweight deployment model.

## 7. Final End-to-End Project Flow

The final project flow should be:

1. Parse AI chat histories and Markdown documents.
2. Normalize text and preserve source metadata.
3. Create bounded chunks using data-type-specific chunking.
4. Extract entities and lightweight relations during ingestion.
5. Write chunk metadata and FTS records into `SQLite`.
6. Write embeddings into `Chroma`.
7. Write entity and relation data into graph-friendly metadata tables in `SQLite`.
8. Run query analysis to detect query type and route behavior.
9. Generate baseline retrieval candidates from:
   - keyword retrieval
   - dense retrieval
10. If the query is suitable, activate graph-enhanced retrieval to generate:
   - entity-level candidates
   - relation-neighborhood candidates
11. Fuse all eligible candidates with explicit scoring and source attribution.
12. Optionally rerank a bounded candidate set.
13. Expand local context in a controlled way.
14. Build answer context from top-ranked evidence.
15. Generate an answer with citations.
16. Run weak grounding checks on the answer.
17. If support is weak, downgrade to a conservative response.
18. Return answer, citations, retrieval path, rerank status, graph routing status, and grounding status.
19. Evaluate all retrieval variants offline with benchmark tasks and report retrieval metrics separately from answer trust metrics.

This flow is the final intended product behavior and the basis for implementation planning.

## 8. Architecture Overview

The architecture stays close to the current repository:

- `FastAPI` backend
- `SQLite` for metadata, FTS, and lightweight graph metadata storage
- `Chroma` for dense vector retrieval
- `React + Vite + TypeScript` frontend

The system is divided into seven layers:

1. `Ingestion Layer`
2. `Index Layer`
3. `Graph Layer`
4. `Retrieval Layer`
5. `Answer Layer`
6. `Evaluation Layer`
7. `Debug and Explain Layer`

## 9. Ingestion Layer

The ingestion layer is responsible for:

- parsing AI chat histories
- parsing Markdown documents
- normalization and cleanup
- data-type-aware chunking
- metadata extraction
- entity extraction
- relation extraction
- incremental indexing

### 9.1 Chunking Strategy

Chunking should differ by source type:

- AI chats should chunk by turn or tightly bounded adjacent turns
- Markdown should chunk by headings, paragraphs, and local semantic blocks

The goal is to preserve retrieval precision while keeping context windows interpretable.

### 9.2 Entity and Relation Extraction

Entity extraction should identify:

- file names
- commands
- model names
- project concepts
- people, tools, libraries, and components when clearly present

Relation extraction should stay lightweight and practical. It is not a full knowledge graph pipeline.

Useful relation examples:

- component depends on component
- file belongs to module
- model used for task
- command associated with tool
- concept mentioned with concept

The relation layer exists to support retrieval augmentation, not symbolic reasoning.

## 10. Index Layer

The index layer should keep three classes of retrievable state:

### 10.1 Chunk Index

Stored in `SQLite` and `Chroma`:

- chunk id
- source id
- source type
- source path
- chunk text
- normalized text
- chunk order
- local neighborhood references

### 10.2 Dense Index

Stored in `Chroma`:

- embedding vector
- chunk id mapping
- retrieval metadata

### 10.3 Graph Metadata Index

Stored in `SQLite`:

- entity table
- relation table
- chunk-to-entity links
- entity-to-entity links
- source-level graph references

This graph metadata store should remain lightweight and local. No separate graph database is required for the MVP.

## 11. Retrieval Architecture

The retrieval architecture has a default path and an advanced path.

## 11.1 Default Retrieval Path

The default retrieval path is:

`query -> query analysis -> keyword retrieval + dense retrieval -> fusion -> rerank gating -> context expansion`

This path should always be available and must deliver strong baseline performance.

## 11.2 Graph-Enhanced Retrieval Path

The graph-enhanced path is:

`query -> query analysis -> entity detection -> concept-level retrieval -> relation-neighborhood expansion -> dual-level fusion with chunk retrieval`

This path should not replace the default path. It should augment it.

## 11.3 Automatic Graph Gating

Graph-enhanced retrieval should only activate when query analysis predicts likely benefit.

Candidate trigger cases:

- entity-centric questions
- multi-hop concept queries
- component relationship questions
- underspecified queries where keyword and dense signals alone are often weak
- chat or document questions that refer to named concepts with strong graph anchors

Graph-enhanced retrieval should be skipped for:

- direct file lookup
- exact command lookup
- short symbolic queries
- situations where baseline retrieval is already highly confident

## 12. Query Analysis

Query analysis is the control plane for the entire system.

It should classify or score queries along these dimensions:

- exact symbolic query vs semantic query
- entity-centric vs chunk-centric
- single-hop vs relation-seeking
- context-dependent vs self-contained
- rerank-worthy vs straightforward
- rewrite-worthy vs rewrite-harmful
- graph-worthy vs graph-unnecessary

This analysis can begin with rules and heuristics. A small learned router can be added later only if benchmark evidence justifies it.

## 13. Candidate Generation

Candidate generation should be explicitly multi-source.

### 13.1 Baseline Candidate Sources

- `SQLite FTS`
- `Chroma dense retrieval`
- optional entity lookup for exact known entities

### 13.2 Graph Candidate Sources

- entity-level hits
- relation-neighborhood expansions
- chunk candidates linked to top-ranked entities
- source-level concept anchors

Every candidate should carry provenance:

- retrieval source
- raw score
- normalized score
- graph path or entity linkage if applicable

## 14. Fusion Strategy

Fusion should be an explicit subsystem, not an opaque detail.

The system should support:

- keyword contribution
- dense contribution
- graph contribution
- optional rerank score override

The first fusion implementation should be simple, inspectable, and benchmarkable.

Recommended first-stage design:

- normalize scores by retriever family
- assign source-aware weights
- cap graph contribution so graph retrieval augments rather than dominates
- preserve exact-match protection for symbolic queries

Potential later enhancement:

- RRF-style rank fusion
- query-type-aware weight presets

## 15. Dual-Level Retrieval Design

This is the core advanced feature of the project.

The project should retrieve at two levels:

### 15.1 Concept or Entity Level

The system retrieves:

- relevant entities
- related concepts
- relation-neighborhood signals

This level is useful for questions about components, relationships, tools, or named concepts.

### 15.2 Chunk Level

The system retrieves:

- textual evidence chunks
- local context windows
- directly answerable passages

This level is required for answer generation and citation.

### 15.3 Dual-Level Fusion

Entity-level signals should help:

- recall the right chunk neighborhood
- rescue semantically relevant but textually weak chunk matches
- improve ranking for relationship-style questions

Chunk-level evidence should still dominate final answer grounding.

The graph layer points the system toward evidence. It does not replace textual evidence.

## 16. Rewrite Strategy

Query rewrite remains conditional.

Rewrite should run only when likely beneficial:

- pronoun-heavy follow-up questions
- unresolved time-reference questions
- underspecified semantic questions

Rewrite should be skipped for:

- file names
- commands
- model names
- clearly exact symbolic queries

Rewrite is an enhancement, not a required stage.

## 17. Rerank Strategy

Reranking should remain bounded and selective.

The reranker should operate after fusion on a limited candidate set.

Rerank should be activated only when:

- the query is ambiguous enough
- the candidate set is small enough
- the latency budget allows it

Rerank should never become the only reason the system works.

If rerank fails or times out:

- keep the pre-rerank ordering
- record the fallback in debug output

## 18. Context Expansion

Context expansion should be conservative and source-aware.

For AI chats:

- expand by adjacent turns
- keep expansion bounded
- preserve conversational chronology

For Markdown:

- expand within the same local section
- preserve heading structure when useful
- avoid document-wide spillover

Graph-level neighbors must not be blindly appended to answer context. Only textual evidence selected after ranking should enter final answer context.

## 19. Answer Layer

The answer layer is responsible for:

- answer context construction
- prompt building
- citation generation
- answer generation
- grounding checks
- conservative downgrade behavior

### 19.1 Citation Requirements

Every successful answer should expose:

- cited chunks
- source file or chat reference
- chunk-level evidence links

The answer layer should treat uncited strong claims as suspicious by default.

### 19.2 Grounding Checks

The first-stage grounding system should stay lightweight and deterministic enough to benchmark.

It should check:

- whether answer claims have supporting cited chunks
- whether cited chunks weakly support the answer
- whether the answer is stronger than the evidence warrants

### 19.3 Conservative Downgrade

If support is insufficient, the system should:

- reduce certainty
- explicitly say evidence is limited
- point to the best supporting passages
- avoid unsupported synthesis

## 20. Failure Handling and Fallbacks

The system must be robust under local constraints.

### 20.1 Retrieval Fallbacks

- if dense retrieval fails, fall back to keyword retrieval
- if graph retrieval fails, continue with baseline hybrid retrieval
- if entity extraction is noisy, graph contribution stays capped
- if rewrite fails, keep the original query
- if rerank fails, keep fused ordering

### 20.2 Generation Fallbacks

- if answer generation fails, return retrieved evidence and citations
- if grounding fails, prefer conservative response over confident synthesis

### 20.3 Design Rule

No advanced feature may become a single point of failure.

The ordering of reliability should be:

`hybrid retrieval > graph enhancement > rerank > generation polish`

## 21. Evaluation Framework

The evaluation framework is the backbone of the project narrative.

It must separately evaluate:

- retrieval quality
- answer trustworthiness
- graph-enhanced retrieval gains on the right query types

## 21.1 Benchmark Dataset

The benchmark should be curated, manually checkable, and typed.

Each benchmark sample should include:

- `question`
- `expected_chunk_ids` or equivalent source ids
- `question_type`
- `difficulty`
- `source_type`
- `requires_relation_reasoning` flag
- `requires_context_resolution` flag
- optional notes

The benchmark should cover:

- exact lookup
- semantic retrieval
- chat recall
- multi-fragment synthesis
- entity-centric lookup
- relation-style questions
- pronoun follow-up questions
- time-reference questions

## 21.2 Retrieval Metrics

First-stage core metrics:

- `Recall@5`
- `Recall@10`
- `MRR@10`
- `HitRate@5`

Optional later metrics:

- `nDCG@K`
- per-query-type win rate

## 21.3 Trustworthiness Metrics

First-stage trust metrics:

- `citation_coverage`
- `source_support_rate`
- `unsupported_answer_rate`
- `abstain_rate`

## 21.4 Graph Retrieval Metrics

Because graph enhancement is a headline feature, it needs targeted evaluation.

The system should report:

- graph-gated query count
- graph-path retrieval win rate over baseline
- query-type-specific gain for entity and relation questions
- graph activation false-positive rate

This is necessary to prove the graph layer is useful rather than decorative.

## 21.5 Experimental Baselines

The benchmark should compare at least:

- `keyword`
- `vector`
- `hybrid`
- `hybrid + rerank`
- `hybrid + rewrite gating`
- `hybrid + graph gating`
- `hybrid + graph gating + rerank`
- `hybrid + graph gating + rerank + grounding`

Important evaluation rule:

- graph routing and reranking affect retrieval
- grounding affects answer trustworthiness
- results must be reported separately

## 22. Debug and Explainability

The frontend and API should make retrieval decisions inspectable.

Every query response should ideally expose:

- query type
- rewrite status
- graph gating status
- retriever contributions
- fusion summary
- rerank status
- grounding status
- fallback status

For graph-enhanced queries, debug should also show:

- triggered entities
- relation-neighborhood usage
- whether graph retrieval changed final top results

This is a major maturity signal for interviews and demos.

## 23. Module Boundaries

The project should evolve by clarifying module boundaries rather than rewriting the repository from scratch.

### 23.1 Existing Modules to Preserve

- `backend/app/services/ingest/`
- `backend/app/services/embedding/`
- `backend/app/services/vectorstore/`
- `backend/app/services/rerank/`
- `backend/app/services/qa/`

### 23.2 New or Strengthened Modules

The design should add or clarify:

- `backend/app/services/retrieval/query_analysis.py`
  - query typing
  - graph gating
  - rewrite gating
  - rerank gating

- `backend/app/services/retrieval/fusion.py`
  - cross-source candidate fusion
  - normalized scoring
  - provenance preservation

- `backend/app/services/graph/`
  - entity extraction adapters if needed
  - relation extraction
  - graph candidate generation
  - neighborhood expansion

- `backend/app/services/qa/grounding.py`
  - grounding checks
  - unsupported answer detection
  - conservative downgrade logic

- `backend/app/services/evaluation/`
  - benchmark loading
  - metrics
  - experiment runners
  - reporting

## 24. MVP Definition

The MVP should deliver:

- local ingestion for AI chats and Markdown
- chunk-level and entity-level indexing
- hybrid retrieval as the stable default
- automatic graph gating
- dual-level graph-enhanced retrieval
- bounded reranking
- citations
- weak grounding checks
- benchmark-based evaluation
- debug visibility for all major retrieval decisions

This MVP is intentionally more advanced than a basic RAG demo, but still bounded enough for a single coherent implementation plan.

## 25. Delivery Phases

### Phase 1: Evaluation Backbone

Deliver:

- benchmark schema
- first typed benchmark set
- baseline evaluation runner
- retrieval metrics
- trust metrics

This phase fixes the measurement problem first.

### Phase 2: Mature Retrieval Backbone

Deliver:

- stronger hybrid retrieval baseline
- explicit query analysis
- explicit fusion logic
- bounded rerank logic
- improved debug explainability

This phase creates the mature primary path.

### Phase 3: Graph-Enhanced Retrieval

Deliver:

- entity and relation extraction
- graph metadata storage in `SQLite`
- graph candidate generation
- automatic graph gating
- dual-level fusion between graph and chunk retrieval
- graph-specific benchmark reporting

This phase creates the advanced retrieval differentiation.

### Phase 4: Trustworthy Answering

Deliver:

- citation coverage checks
- grounding checks
- conservative downgrade behavior
- richer answer/debug output

This phase turns strong retrieval into trustworthy QA behavior.

## 26. Resume Narrative

The project should support resume bullets like:

- Designed and implemented a lightweight local-first RAG system for AI chat histories and Markdown documents, combining hybrid retrieval, conditional reranking, citations, and offline evaluation.
- Built a LightRAG-inspired dual-level graph-enhanced retrieval layer with automatic query gating, using lightweight entity and relation metadata to improve recall for entity-centric and relationship-style questions.
- Established an offline benchmark framework with metrics such as `Recall@K`, `MRR`, citation coverage, and unsupported answer rate to evaluate retrieval gains and answer trustworthiness separately.

## 27. Interview Narrative

The project should be discussable from two angles.

### 27.1 Engineering Angle

- why hybrid retrieval is the stable default
- why graph enhancement is gated
- how fallback behavior preserves reliability
- how debug signals make the system observable

### 27.2 Algorithm and Retrieval Angle

- why dual-level retrieval helps on entity and relation queries
- why graph signals must not dominate textual evidence
- how evaluation isolates graph gains from generic hybrid gains
- how routing and fusion trade off recall, precision, and latency

## 28. Risks and Open Questions

### 28.1 Main Risks

- graph enhancement may add complexity without enough benchmark gain
- entity or relation extraction may be noisy on real chat data
- graph gating may misfire and waste latency on easy queries
- rerank and graph gains may overlap and become hard to interpret
- grounding rules may become too strict

### 28.2 Open Questions

- whether relation extraction should remain rule-based or become partially model-assisted
- whether graph fusion should use weighted scores or rank-based fusion first
- how large the first benchmark should be for credible per-query-type analysis
- whether a small learned query router is worth adding later

These questions should be resolved by benchmark evidence, not intuition.

## 29. Final Summary

The final intended system is:

- a lightweight local-first RAG project
- with a mature hybrid retrieval backbone
- with a LightRAG-inspired dual-level graph-enhanced retrieval layer
- with automatic graph gating
- with bounded reranking
- with citation-aware and grounding-aware answering
- with an evaluation framework that proves where gains come from

That combination is what makes the project both mature and forward-looking enough for LLM application, search, and algorithm-oriented resume positioning.

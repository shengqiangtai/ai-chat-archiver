# Architecture

AI Chat Archiver 是一个本地优先的 AI 对话归档与 RAG 系统，当前由 4 层组成：

1. 浏览器插件：采集 ChatGPT、Claude、Gemini、DeepSeek、Poe 等页面中的对话。
2. FastAPI 后端：负责持久化、索引、检索、问答与流式输出。
3. 本地存储层：文件系统 + SQLite + ChromaDB。
4. Dashboard：查看归档、执行检索、发起知识库问答。

## Data Flow

### 1. Collection

- `extension/content_scripts/*` 从页面 DOM 提取消息。
- 插件将结构化对话发送给后端。
- 后端将每次对话落盘为 `AI-Chats/<platform>/<year>/<date_title>/chat.md` 与 `meta.json`。
- 同时把聊天级元数据与全文内容写入 SQLite 的 `chats` / `chats_fts`，用于聊天列表和全文搜索。

### 2. Indexing

- `backend/app/services/ingest/loader.py` 扫描 `chat.md + meta.json`。
- `backend/app/services/ingest/parser.py` 将 markdown 解析为消息列表，并按问答轮次分组。
- `backend/app/services/ingest/chunker.py` 优先按对话轮次切块，必要时对超长轮次做滑动窗口切分。
- `backend/app/services/embedding/embedder.py` 为 chunk 生成向量。
- chunk 会同时写入：
  - ChromaDB：用于 dense retrieval。
  - SQLite `kb_chunks` / `kb_chunks_fts`：用于 chunk 级关键词检索、邻近轮次扩展和 metadata 查询。
- 索引阶段还会抽取轻量实体，并写入 SQLite `kb_entities` / `kb_entity_mentions` / `kb_entity_edges`。
- `backend/app/services/ingest/deduper.py` + SQLite `file_index` / `chunk_hashes` 用于增量索引与重复检测。

### 3. Retrieval

- `backend/app/services/retrieval/query_analysis.py` 会先对查询做轻量分类，决定：
  - 是否适合 graph/entity 增强
  - 是否适合 rerank
  - 当前查询更接近 symbolic、relation、follow-up 还是普通 semantic
- `backend/app/services/qa/query_rewrite.py` 会在提问含有“上次 / 之前 / 那个”等指代时，先做独立查询改写。
- `backend/app/services/vectorstore/retrieval.py` 提供五种检索模式：
  - `vector`：纯向量检索。
  - `keyword`：纯 FTS 检索。
  - `hybrid`：向量检索 + FTS 双路召回，再用 RRF 融合。
  - `entity`：基于实体索引和共现边扩展后的实体检索。
  - `mix`：向量检索 + FTS + entity/graph 三路融合。
- `backend/app/services/graph/retrieval.py` 会在 query analysis 允许时，基于 `kb_graph_relations` 生成轻量 graph-assisted candidates。
- graph 路径默认走 `graph_mode=auto`，可以在 benchmark 里用 `graph_mode=off` 做 baseline 对比。
- 召回后可选 cross-encoder rerank。
- QA 模式下会根据命中的 `turn_index` 自动扩展相邻轮次，减少片段化上下文。
- 检索结果会进入 SQLite 查询缓存。
- debug 输出会显式带出 `query_analysis`、`graph_routed`、`graph_hit_count`、`rerank_effective_mode` 等可解释字段。

### 4. QA

- `backend/app/services/qa/pipeline.py` 编排：
  - hybrid retrieval
  - graph-gated retrieval augmentation
  - rerank
  - neighbor turn expansion
  - prompt construction
  - local LLM generation
  - citation parsing
  - grounding checks
  - conservative downgrade when support is weak
  - answer cache
- `backend/app/services/llm/generator.py` 支持 LM Studio / Ollama / transformers。
- `backend/app/services/qa/citation.py` 对回答做引用解析，并校验引用编号是否与检索片段一致。
- `backend/app/services/qa/grounding.py` 对回答做轻量支撑检查；当回答和来源片段重叠不足时，QA 会降级成保守的 retrieval summary。

## Storage Layout

- `AI-Chats/`: 原始归档文件。
- `AI-Chats/index.db`: 聊天级搜索、chunk 级关键词索引、增量索引状态。
- `backend/data/chroma/`: 向量索引。
- `backend/data/cache/query_cache.db`: 检索缓存与回答缓存。

## Key Modules

- `backend/app/api/routes_docs.py`: 聊天记录 CRUD 与全文搜索。
- `backend/app/api/routes_ingest.py`: 全量 / 增量索引管理。
- `backend/app/api/routes_search.py`: 知识库检索接口。
- `backend/app/api/routes_qa.py`: 知识库问答接口。
- `backend/app/services/qa/query_rewrite.py`: 查询改写。
- `backend/app/services/vectorstore/retrieval.py`: hybrid retrieval 主链路。
- `backend/app/services/qa/pipeline.py`: RAG orchestration。

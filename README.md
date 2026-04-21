# AI Chat Archiver

本地优先的 Graph-Enhanced RAG 系统，面向 `AI chats + Markdown` 知识库场景，提供归档、混合检索、图增强召回、引用问答与离线评测能力。

项目主线不是“做一个功能很多的知识库工具”，而是围绕轻量部署前提下的检索优化与回答可信度，构建一套可运行、可评测、可解释的 RAG 工程实践。

## Highlights

- `SQLite FTS + Chroma` 作为默认 hybrid retrieval 主路径
- `query-aware graph routing` 按查询类型自动启用轻量图增强双层检索
- `fusion + bounded rerank` 在保持轻量性的前提下提升候选排序质量
- `citations + grounding checks` 支持带引用回答与证据不足时的保守降级
- `typed benchmark + reporting` 可对比 `graph-off` 和 `graph-auto` 等检索策略收益
- 本地优先运行，适合作为 RAG/检索优化项目、实验基线和求职展示项目

## System Overview

系统由 4 个核心部分组成：

1. `Collection`
   浏览器插件采集 ChatGPT、Claude、Gemini、DeepSeek、Poe 等平台的对话内容，并保存到本地。
2. `Indexing`
   后端对 `AI chats + Markdown` 做解析、切块、向量化、实体/关系抽取和增量索引。
3. `Retrieval + QA`
   在线查询默认走 hybrid retrieval，在合适场景下自动启用 graph-enhanced dual-level retrieval，并输出带 citation 的回答。
4. `Evaluation`
   基于 typed benchmark 运行 `Recall@K`、`HitRate@K`、`MRR@10` 等离线评测，对比不同检索策略。

## Retrieval Pipeline

默认问答链路如下：

```text
query
  -> query analysis
  -> hybrid retrieval (FTS + dense)
  -> graph-enhanced routing when useful
  -> explicit fusion
  -> bounded rerank
  -> context expansion
  -> answer with citations
  -> grounding checks
  -> conservative fallback when support is weak
```

其中图增强路径参考了 LightRAG 一类系统的思路，但保持为轻量、可门控、可回退的增强层，而不是引入重型 GraphRAG 基础设施。

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

默认服务地址：

- Dashboard: `http://localhost:8765`
- API Docs: `http://localhost:8765/docs`

### 2. Frontend Dev

```bash
cd frontend
npm install
npm run dev
```

生产构建：

```bash
cd frontend
npm run build
```

### 3. Browser Extension

在 Chrome 的 `chrome://extensions/` 中开启开发者模式，然后加载仓库中的 `extension/` 目录，即可归档当前 AI 对话页面。

## Evaluation

当前仓库内置了 typed benchmark 与评测骨架，适合对比不同检索模式：

- `keyword`
- `vector`
- `hybrid`
- `mix`
- `mix_graph`
- `mix_rerank`

主要指标包括：

- `Recall@5 / Recall@10`
- `HitRate@5`
- `MRR@10`

运行方式：

```bash
cd backend
. .venv/bin/activate
python tests/run_rag_benchmark.py
```

## Repository Structure

```text
ai-chat-archiver/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   │   ├── evaluation/
│   │   │   ├── graph/
│   │   │   ├── ingest/
│   │   │   ├── qa/
│   │   │   ├── retrieval/
│   │   │   ├── rerank/
│   │   │   └── vectorstore/
│   │   └── db/
│   └── tests/
├── frontend/
├── extension/
├── docs/
│   ├── ARCHITECTURE.md
│   └── EVALUATION.md
└── README.md
```

## Docs

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Evaluation: [docs/EVALUATION.md](docs/EVALUATION.md)

## Privacy

项目默认面向本地部署。聊天原始数据、索引缓存和模型文件不应上传到公开仓库。公开分支建议仅保留代码、测试、文档与可复现配置。

## License

MIT

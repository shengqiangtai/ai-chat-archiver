# AI Chat Archiver

本地优先的 Graph-Enhanced RAG 系统，面向 `AI chats + Markdown` 知识库场景，提供聊天归档、混合检索、图增强召回、引用问答与离线评测能力。

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

## Quick Start

### 1. Install LM Studio and a Generation Model

推荐使用 LM Studio 管理生成模型，尤其在 macOS 上更省内存。

1. 从 [lmstudio.ai](https://lmstudio.ai) 下载并安装 LM Studio
2. 在 LM Studio 中下载一个生成模型，例如：
   - `Qwen3.5-0.8B-GGUF`
   - `Qwen2.5-1.5B-Instruct-GGUF`
3. 打开 `Local Server` 并启动服务
4. 确认接口运行在 `http://localhost:1234`

### 2. Download Embedding and Reranker Models

生成模型可以由 LM Studio 管理；仓库默认还需要本地 Embedding 与 Reranker 模型。

```bash
cd backend

# 使用国内镜像加速下载
HF_ENDPOINT=https://hf-mirror.com python download_models.py --model embedding
HF_ENDPOINT=https://hf-mirror.com python download_models.py --model reranker

# 或一次全部下载
HF_ENDPOINT=https://hf-mirror.com python download_models.py
```

### 3. Start the Backend

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

默认生成后端为 `LM Studio`。如果 LM Studio 不可用，系统可降级到 `Ollama`，再到 `transformers` 本地推理。

### 4. Start the Frontend in Dev Mode

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

### 5. Install the Browser Extension

在 Chrome 的 `chrome://extensions/` 中：

1. 开启开发者模式
2. 点击“加载已解压的扩展程序”
3. 选择仓库中的 `extension/` 目录
4. 打开任意 AI 对话页面，点击插件图标归档当前对话

### 6. Build the Knowledge Base Index

首次使用或新增对话后，需要构建索引。

方式一：通过 Dashboard

```text
打开 http://localhost:8765
→ 进入“知识库”页
→ 管理
→ 点击“全量重建索引”
```

方式二：通过命令行

```bash
curl -X POST http://localhost:8765/api/kb/reindex
```

索引完成后即可进行语义检索和知识库问答。

## LLM Backends

系统支持三种生成后端：

| Backend | 说明 | 推荐场景 |
|---|---|---|
| `LM Studio` | OpenAI 兼容 API，适合本地 GGUF 模型 | macOS 本地部署首选 |
| `Ollama` | 模型管理简单 | Linux 或轻量服务环境 |
| `transformers` | Python 直接推理 | 兜底方案 |

### LM Studio

优点：

- GGUF 量化模型更轻
- Metal GPU 加速友好
- 独立进程管理内存
- 流式输出体验更好

### Ollama

如果你更偏向 Ollama：

```bash
brew install ollama
ollama serve
ollama pull qwen2.5:3b
```

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

## API

### Knowledge Base

| Method | Path | 描述 |
|---|---|---|
| `POST` | `/api/kb/reindex` | 全量重建索引 |
| `POST` | `/api/kb/reindex/incremental` | 增量索引 |
| `GET` | `/api/kb/reindex/progress/{task_id}` | 查询索引进度 |
| `GET` | `/api/kb/status` | 知识库状态 |
| `POST` | `/api/kb/search` | 检索 |
| `POST` | `/api/kb/qa` | 非流式问答 |
| `POST` | `/api/kb/qa/stream` | SSE 流式问答 |

### Chats

| Method | Path | 描述 |
|---|---|---|
| `POST` | `/save` | 保存聊天记录 |
| `GET` | `/chats` | 获取记录列表 |
| `GET` | `/chats/{id}` | 获取记录详情 |
| `DELETE` | `/chats/{id}` | 删除记录 |
| `POST` | `/search` | 全文关键词搜索 |
| `GET` | `/stats` | 统计信息 |

## Environment Variables

常用环境变量如下：

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `ARCHIVER_STORAGE_ROOT` | `AI-Chats` | 聊天数据存储目录 |
| `ARCHIVER_PORT` | `8765` | 服务端口 |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding 模型路径或 repo_id |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Reranker 模型路径或 repo_id |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API 地址 |
| `LMSTUDIO_MODEL` | `""` | LM Studio 当前模型名 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 地址 |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama 模型名 |
| `RETRIEVAL_TOP_K` | `15` | 初始召回数量 |
| `RERANK_TOP_N` | `6` | Rerank 保留数量 |
| `CHUNK_TARGET_SIZE` | `700` | 目标切块大小 |

## Repository Structure

```text
ai-chat-archiver/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── services/
│   │   │   ├── evaluation/
│   │   │   ├── graph/
│   │   │   ├── ingest/
│   │   │   ├── qa/
│   │   │   ├── retrieval/
│   │   │   ├── rerank/
│   │   │   └── vectorstore/
│   │   └── utils/
│   ├── data/
│   ├── tests/
│   ├── download_models.py
│   └── requirements.txt
├── frontend/
├── extension/
├── models/
├── AI-Chats/
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

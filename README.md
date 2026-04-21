# AI Chat Archiver

本项目不是一个普通的“聊天导出工具”，而是一个面向 AI 聊天记录与 Markdown 文档的 `local-first, graph-enhanced RAG system`。

它的目标有两条：

- 作为可落地的软件系统：完成聊天归档、增量索引、混合检索、引用问答和前端调试
- 作为可写进简历的 RAG 项目：围绕 `hybrid retrieval + graph-enhanced retrieval + rerank + grounding + offline evaluation` 建立一条成熟且可解释的优化闭环

当前系统支持 ChatGPT、Claude、Gemini、DeepSeek、Poe 等平台的对话归档，并在知识库层实现：

- `SQLite FTS + Chroma` 的混合检索主链路
- 轻量 `entity / relation` 图元数据增强检索
- `query analysis` 驱动的 graph gating / rerank gating
- citation-aware answering 与 grounding downgrade
- typed benchmark 与离线评测

如果你把它当成求职项目，它更适合被描述为：

> 一个受 LightRAG / RAGFlow 思路启发、但保持本地优先和轻量部署的 Graph-Enhanced RAG 系统。

**项目关键词：`Hybrid RAG` `Graph-Enhanced Retrieval` `Lightweight GraphRAG` `Offline Evaluation` `Grounded QA`**

---

## 为什么这个项目值得展示

和常见的 Demo 型 RAG 项目相比，这个仓库的区别不在“接了向量库”，而在于它已经形成了比较完整的工程与检索闭环：

- `成熟工程主线`
  - 浏览器插件归档聊天
  - FastAPI + React 全链路可运行
  - 增量索引、缓存、fallback、debug 面板都已经接好

- `前沿检索增强`
  - baseline 不是单一路径，而是 `keyword + dense + fusion`
  - 在此基础上增加了轻量 `graph-assisted retrieval`
  - graph 路径默认不是强依赖，而是由 `query analysis` 自动门控

- `可验证，不靠主观体感`
  - 仓库内有 typed benchmark
  - 可以比较 `vector / hybrid / mix / mix_graph / mix_rerank`
  - 检索指标与回答可信度指标是分开看的

- `可信回答`
  - 回答必须带 citation
  - 回答生成后会经过轻量 grounding 检查
  - 支撑不足时自动降级为保守回答

这也是它适合大模型开发岗、RAG/检索岗、LLM 应用工程岗的原因。

## 技术亮点

- `Lightweight Graph-Enhanced Retrieval`
  - 在不引入独立图数据库的前提下，把 `entity / relation` 元数据落到 SQLite
  - 基于 `kb_graph_relations` 做轻量 graph candidate retrieval
  - graph candidates 只做增强，不替代 chunk-level textual evidence

- `Query-Aware Routing`
  - 轻量 query analysis 先判断当前查询更偏 `symbolic / relation / follow-up / semantic`
  - 再决定是否允许 graph path、是否允许 rerank
  - 保证 graph 路径不会污染显式文件名/命令查询

- `Dual-Level Retrieval`
  - chunk-level: dense / keyword / entity hits
  - concept-level: graph-assisted relation hits
  - 通过可解释 fusion 合并，形成最终候选集

- `Grounded QA`
  - 生成回答后做弱支撑检查
  - 对 unsupported claims 自动降级
  - 前端和 debug payload 都能看到 grounding 状态

- `Evaluation-Driven Iteration`
  - typed benchmark case
  - `Recall@5 / Recall@10 / HitRate@5 / MRR@10`
  - graph route rate / avg graph hits
  - 后续可以继续扩成 relation-specific win rate

## 适合简历的项目描述

一句话版本：

> 构建了一个本地优先、图增强的轻量级 RAG 系统，围绕混合检索、graph-assisted retrieval、citation 和 grounding 建立离线评测闭环，用于提升 AI 聊天知识库的检索准确率和回答可信度。

更完整的简历版描述和面试讲法在：

- [docs/RESUME_AND_INTERVIEW.md](/Users/shengqiangtai/Desktop/ai_doc/ai-chat-archiver/docs/RESUME_AND_INTERVIEW.md)

## 快速启动

这一节只关心一件事：

`让一个第一次使用的人，在本地成功归档一条聊天，并完成第一次问答。`

如果你只想最快跑通，请按下面顺序走，不要跳步。

### 0. 使用前准备

需要准备：
- Python `3.11+`
- Node.js `18+`
- Chrome 或兼容 Chromium 的浏览器
- 至少一条你能打开的 AI 聊天页面

推荐机器：
- Apple Silicon Mac
- 或能本地跑 LM Studio / Ollama 的普通家用电脑

### 1. 先把生成模型准备好

最推荐的方式是 `LM Studio`，因为它对本地机器更友好。

1. 从 [lmstudio.ai](https://lmstudio.ai) 下载并安装 LM Studio
2. 打开 LM Studio，下载一个小模型即可：
   - `Qwen3.5-0.8B-GGUF`：更适合轻量本地使用
   - `Qwen2.5-1.5B-Instruct-GGUF`：更强，但更占资源
3. 加载模型后，进入 **Local Server**
4. 点击 **Start Server**
5. 确认地址是 `http://localhost:1234`

你做到这里时，应该满足：
- LM Studio 正在运行
- 本地有一个已经加载好的生成模型
- Local Server 已启动

### 2. 安装后端依赖

```bash
cd backend

python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

> Python 版本必须是 `3.11+`。如果你用的是 `3.8 / 3.9 / 3.10`，后端会在类型解析阶段报错。

### 3. 下载本地检索模型

这个项目默认把“生成”交给 LM Studio，把“检索”交给本地 Python。

因此首次只需要准备：
- embedding 模型
- reranker 模型

```bash
cd backend

# 使用国内镜像下载
HF_ENDPOINT=https://hf-mirror.com python download_models.py --model embedding
HF_ENDPOINT=https://hf-mirror.com python download_models.py --model reranker
```

如果你想一次性全下，也可以：

```bash
cd backend
HF_ENDPOINT=https://hf-mirror.com python download_models.py
```

### 4. 启动服务

```bash
cd backend
python -m app.main
```

启动成功后，打开：
- Dashboard：`http://localhost:8765`
- API 文档：`http://localhost:8765/docs`

如果这里打不开，先不要继续后面的归档和索引。

### 5. 安装浏览器插件并归档第一条聊天

1. 打开 Chrome，进入 `chrome://extensions/`
2. 开启右上角“开发者模式”
3. 点击“加载已解压的扩展程序”
4. 选择仓库里的 `extension/` 目录
5. 打开任意一条 AI 聊天
6. 点击插件图标，执行归档

归档成功后，你应该看到：
- 项目根目录下出现 `AI-Chats/`
- 对应平台目录下出现一条新聊天
- 目录里至少有 `chat.md` 和 `meta.json`

### 6. 建立知识库索引

首次使用建议直接做一次 `全量重建索引`。

方式一：在页面里操作

```text
打开 http://localhost:8765
→ 进入「知识库」
→ 切换到「管理」
→ 点击「全量重建索引」
```

方式二：命令行

```bash
curl -X POST http://localhost:8765/api/kb/reindex
```

如果你之后只是新增了聊天记录，通常跑 `增量索引` 就够了。

### 7. 验证第一次是否真的跑通

索引完成后，去 Dashboard 的「知识库」页做两件事：

1. 在语义搜索里搜一个你刚归档聊天里明确出现过的词
2. 在问答页直接问一个刚才那条聊天里的问题

正常情况下你应该看到：
- 检索结果能命中对应聊天片段
- 回答里带来源引用
- `mix + auto rerank` 默认生效

### 8. 常见首次使用问题

如果你已经归档了聊天，但问答页还是空的，优先检查：

1. `AI-Chats/` 里是否真的有 `chat.md` 和 `meta.json`
2. 是否执行过一次索引
3. 后端是否正在运行在 `8765`
4. LM Studio 是否已经启动本地 server

如果你新增聊天后没有立刻搜到，优先做：

1. 进入「管理」页执行一次 `增量索引`
2. 如果还是不对，再执行一次 `清理失效索引`
3. 最后再做一次 `全量重建索引`

---

## LLM 生成后端说明

系统支持三种生成后端，可在 Dashboard → 知识库 → 管理 中切换：

| 后端 | 说明 | 推荐场景 |
|------|------|----------|
| **LM Studio**（默认） | OpenAI 兼容 API，GGUF 量化模型 | Mac 用户首选，内存占用低，速度快 |
| Ollama | 轻量级模型管理工具 | Linux 服务器部署 |
| Transformers | Python 直接推理，不依赖外部服务 | 兜底方案，内存占用大 |

### LM Studio（推荐 Mac 用户）

LM Studio 的优势：
- **GGUF 量化模型**：0.8B 模型量化后仅约 0.6 GB，比 transformers 的 1.8 GB 小很多
- **Metal GPU 加速**：M1/M2/M3 原生 GPU 推理，比 CPU 快 5-10 倍
- **独立进程管理内存**：不占 Python 进程内存，不影响检索和索引
- **真流式输出**：token 级别的实时流式回答
- **GUI 管理**：可视化切换模型、调参

### Ollama（可选）

```bash
brew install ollama
ollama serve
ollama pull qwen2.5:3b
```

---

## 前端开发模式（可选）

```bash
cd frontend
npm install       # 首次需要安装依赖
npm run dev       # 前端开发服务器运行在 http://localhost:5173
```

构建生产版本：

```bash
cd frontend
npm run build     # 产物输出到 frontend/dist/，后端会自动加载
```

---

## 项目结构

```
ai-chat-archiver/
├── backend/                    # Python 后端
│   ├── app/
│   │   ├── main.py             # 启动入口
│   │   ├── api/                # API 路由层
│   │   │   ├── routes_docs.py      # 聊天记录 CRUD
│   │   │   ├── routes_ingest.py    # 索引管理
│   │   │   ├── routes_search.py    # 语义检索
│   │   │   └── routes_qa.py        # 知识库问答
│   │   ├── core/               # 配置 & 日志
│   │   ├── models/             # Pydantic 数据模型
│   │   ├── services/           # 核心业务逻辑
│   │   │   ├── ingest/         # 文档加载/解析/切块/去重
│   │   │   ├── embedding/      # Embedding 模型封装
│   │   │   ├── rerank/         # Reranker 模型封装
│   │   │   ├── llm/            # 生成模型 + Prompt 构建
│   │   │   ├── vectorstore/    # ChromaDB 向量库
│   │   │   ├── qa/             # RAG Pipeline + 引用解析
│   │   │   └── cache/          # 两层查询缓存
│   │   ├── db/                 # SQLite 操作封装
│   │   └── utils/              # 哈希/文本清洗/Token 估算
│   ├── data/                   # 运行时数据（chroma、cache、sqlite）
│   ├── download_models.py      # 模型下载脚本
│   └── requirements.txt
├── frontend/                   # React SPA（Vite + TS + Tailwind）
│   ├── src/
│   │   ├── pages/              # 页面：聊天记录 / 知识库问答
│   │   ├── components/         # UI 组件
│   │   ├── api/                # 前端 API 客户端
│   │   └── types/              # TypeScript 类型定义
│   └── dist/                   # 构建产物（后端静态文件服务）
├── extension/                  # Chrome 浏览器插件
├── models/                     # 本地模型目录
│   ├── Qwen3-Embedding-0.6B/
│   ├── Qwen3-Reranker-0.6B/
│   └── Qwen3.5-0.8B/
├── AI-Chats/                   # 聊天记录存储目录（已加入 .gitignore）
│   ├── ChatGPT/
│   ├── Claude/
│   ├── Gemini/
│   ├── DeepSeek/
│   └── Poe/
└── README.md
```

---

## 模型说明

| 角色 | 模型 | 大小 | 用途 |
|------|------|------|------|
| Embedding | Qwen3-Embedding-0.6B | ~1.2 GB | 将文本转为 1024 维向量 |
| Reranker | Qwen3-Reranker-0.6B | ~1.2 GB | 对检索结果精排 |
| Generator | Qwen3.5-0.8B | ~1.8 GB | 根据上下文生成答案 |

---

## RAG 工作链路

```
用户提问
  → Query Rewrite（含指代词时改写为独立检索问题）
  → Query Analysis
      - 查询类型：symbolic / relation / follow-up / semantic
      - 是否启用 rerank
      - 是否允许 graph-enhanced retrieval
  → Query 预处理（去停用词、规则改写）
  → 多路召回
      - 向量检索（ChromaDB，top_k=15）
      - 关键词检索（SQLite FTS5，chunk 级）
      - 实体检索（entity → chunks）
      - Graph-assisted retrieval（SQLite relation metadata）
  → Hybrid / Mix / Graph-Mix 融合
  → Rerank 精排（Qwen3-Reranker-0.6B，保留 top_n=6）
  → 相邻轮次扩展（补齐上下文）
  → Context 打包（控制总长度 ≤ 2800 字符）
  → 生成回答（LM Studio / Ollama / transformers，严格约束 Prompt）
  → 引用解析（提取 [Source X] 标注）
  → Grounding Checks（检测回答是否有来源支撑）
  → Conservative Downgrade（支撑不足时回退到保守回答）
  → 引用校验（验证引用编号与来源一致）
  → 两层缓存（retrieval cache + answer cache）
  → 返回带来源引用的最终答案
```

语义搜索页支持查看检索调试信息，包括：
- 改写后的查询
- Query Analysis（query type / graph gate / rerank gate）
- Query / Expanded entities
- Graph Routed / Graph Hits
- Dense / Keyword / Entity / Final Top hits
- 是否命中缓存
- 各阶段候选数量
- Grounding 与回答降级状态（QA 路径）

---

## 轻量评测

仓库内已经带了一个本地 benchmark 基线，用于比较：

- `vector`
- `hybrid`
- `entity`
- `mix`
- `mix_graph`
- `mix + rerank`

graph 路径的对比不是单独做一个重型 GraphRAG baseline，而是：

- `mix`: graph 关闭，作为 baseline
- `mix_graph`: graph 自动门控，只在适合的 query 上启用

这更接近真实系统里的部署方式，也更适合在面试里讲“收益 / 成本 / 触发条件”的权衡。

相关文件：

- `backend/tests/fixtures/rag_benchmark.json`
- `backend/tests/run_rag_benchmark.py`
- `docs/EVALUATION.md`

---

## API 参考

### 知识库

| Method | Path | 描述 |
|--------|------|------|
| POST | `/api/kb/reindex` | 全量重建索引 |
| POST | `/api/kb/reindex/incremental` | 增量索引（只处理新增） |
| GET | `/api/kb/reindex/progress/{task_id}` | 查询索引进度 |
| GET | `/api/kb/status` | 知识库状态统计 |
| POST | `/api/kb/search` | 语义检索（返回原始 Chunk） |
| POST | `/api/kb/qa` | 知识库问答（非流式） |
| POST | `/api/kb/qa/stream` | 知识库问答（SSE 流式） |

### 聊天记录

| Method | Path | 描述 |
|--------|------|------|
| POST | `/save` | 保存聊天记录 |
| GET | `/chats` | 获取记录列表 |
| GET | `/chats/{id}` | 获取记录详情 |
| DELETE | `/chats/{id}` | 删除记录 |
| POST | `/search` | 全文关键词搜索 |
| GET | `/stats` | 统计信息 |

---

## 环境变量配置

所有配置均可通过环境变量覆盖（也可写入 `backend/.env.local`）：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ARCHIVER_STORAGE_ROOT` | `ai-chat-archiver/AI-Chats` | 聊天数据存储根目录 |
| `ARCHIVER_PORT` | `8765` | 服务监听端口 |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding 模型路径或 HF repo_id |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Reranker 模型路径或 HF repo_id |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API 地址 |
| `LMSTUDIO_MODEL` | `""` (自动) | LM Studio 使用的模型名 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama 使用的模型名 |
| `RETRIEVAL_TOP_K` | `15` (低内存: 10) | 向量检索初筛数量 |
| `RERANK_TOP_N` | `6` (低内存: 4) | Rerank 后保留数量 |
| `CHUNK_TARGET_SIZE` | `700` (低内存: 500) | 目标切块大小（字符数） |

---

## License

MIT

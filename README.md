# AI Chat Archiver

本地 AI 聊天归档 + 知识库问答系统。支持 ChatGPT、Claude、Gemini、DeepSeek、Poe 等平台的对话记录归档、语义检索和本地 RAG 问答。

**核心特性：完全本地运行，零联网，保护隐私。**

---

## 快速启动

### 第一步：安装 LM Studio + 加载模型

> **推荐方式**：用 LM Studio 管理生成模型，Mac 上内存管理远优于 Python 直接推理。

1. 从 [lmstudio.ai](https://lmstudio.ai) 下载并安装 LM Studio
2. 打开 LM Studio，搜索并下载以下模型（任选其一）：
   - `Qwen3.5-0.8B-GGUF`（推荐，约 0.6 GB 量化后）
   - `Qwen2.5-1.5B-Instruct-GGUF`（更强但更大）
3. 加载模型，点击左侧 **Local Server** → **Start Server**
4. 确认服务运行在 `http://localhost:1234`

---

### 第二步：下载 Embedding & Reranker 模型（仅首次）

> 生成模型由 LM Studio 管理，只需下载 Embedding 和 Reranker 两个模型到本地。

```bash
cd backend

# 使用国内镜像加速下载
HF_ENDPOINT=https://hf-mirror.com python download_models.py --model embedding   # ~1.2 GB
HF_ENDPOINT=https://hf-mirror.com python download_models.py --model reranker    # ~1.2 GB

# 也可以一次全部下载（包含生成模型，用于 transformers 兜底）
HF_ENDPOINT=https://hf-mirror.com python download_models.py
```

---

### 第三步：安装后端依赖

```bash
cd backend

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 安装依赖
pip install -r requirements.txt
```

> **Python 版本要求：3.11+**

---

### 第四步：启动后端服务

```bash
cd backend
python -m app.main
```

服务启动后访问：
- **Dashboard**：http://localhost:8765
- **API 文档**：http://localhost:8765/docs

> 系统默认使用 LM Studio 作为生成后端。如 LM Studio 未运行，会自动降级到 Ollama，再到 transformers 本地推理。

---

### 第五步：安装浏览器插件（采集对话）

1. 打开 Chrome，进入 `chrome://extensions/`
2. 开启右上角「开发者模式」
3. 点击「加载已解压的扩展程序」
4. 选择项目中的 `extension/` 目录
5. 打开任意 AI 对话页面，点击插件图标即可归档当前对话

---

### 第六步：构建知识库索引（首次使用或新增对话后）

方式一：通过 Dashboard

```
打开 http://localhost:8765 → 切换到「知识库」选项卡 → 管理 → 点击「全量重建索引」
```

方式二：通过命令行

```bash
curl -X POST http://localhost:8765/api/kb/reindex
```

索引完成后即可在「知识库」页面进行语义检索和 AI 问答。

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
  → Query 预处理（去停用词、规则改写）
  → Embedding 编码（Qwen3-Embedding-0.6B，查询侧加指令前缀）
  → 向量检索（ChromaDB，top_k=15）
  → Rerank 精排（Qwen3-Reranker-0.6B，保留 top_n=6）
  → Context 打包（控制总长度 ≤ 2800 字符）
  → 生成回答（LM Studio / Ollama / transformers，严格约束 Prompt）
  → 引用解析（提取 [Source X] 标注）
  → 幻觉检测（验证引用真实性）
  → 返回带来源引用的最终答案
```

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

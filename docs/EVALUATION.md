# 检索评测说明

这份文档描述当前的 RAG 检索评测骨架。目标很直接：

`每次调检索、rerank 或图增强逻辑时，都能用同一套 typed benchmark 判断是否真的变好了。`

## 现在的评测骨架

- 固定问题集：`backend/tests/fixtures/rag_benchmark.json`
- 评测脚本：`backend/tests/run_rag_benchmark.py`
- 评测核心模块：
  - `backend/app/services/evaluation/models.py`
  - `backend/app/services/evaluation/metrics.py`
  - `backend/app/services/evaluation/runner.py`
  - `backend/app/services/evaluation/reporting.py`

脚本仍然支持两种 transport：

- `local`: 直接调用后端检索链路
- `http`: 通过 `/api/kb/search` 请求后端

## Benchmark Schema

每个 case 现在使用 typed schema，而不是 `title_contains` / `keyword_any` 这类启发式判定。

```json
{
  "id": "skill-install-01",
  "question": "codex skills 安装",
  "expected_chunk_ids": ["codex-skills-install-guide"],
  "question_type": "installation",
  "difficulty": "easy",
  "source_type": "chat",
  "requires_relation_reasoning": false,
  "requires_context_resolution": false
}
```

字段含义：

- `id`: 用例唯一标识
- `question`: 要检索的问题
- `expected_chunk_ids`: 这个问题应该命中的一个或多个 chunk
- `question_type`: 问题类型，例如 `installation`、`path_lookup`、`relation`
- `difficulty`: `easy` / `medium` / `hard`
- `source_type`: 数据来源类型，例如 `chat`
- `requires_relation_reasoning`: 是否需要关系推理
- `requires_context_resolution`: 是否需要上下文消解

## Metrics

当前评测骨架提供 3 个核心指标：

- `Recall@K`: 前 `K` 个结果里命中的相关 chunk 占所有相关 chunk 的比例
- `HitRate@K`: 前 `K` 个结果里是否至少命中一个相关 chunk
- `MRR@10`: 前 10 个结果里第一个相关 chunk 的倒数排名

其中：

- `Recall@5` 和 `Recall@10` 用于衡量召回覆盖面
- `HitRate@5` 用于衡量最少命中能力
- `MRR@10` 用于衡量相关结果排在前面的程度

## Runner Behaviour

`evaluate_retrieval_case` 会：

- 接收 `BenchmarkCase`
- 将检索结果归一化成 `ranked_chunk_ids`
- 计算 recall / hit rate / MRR
- 原样保留 case 上的关系推理和上下文消解标记

这意味着评测结果里可以继续看：

- 这个 case 是否需要 relation reasoning
- 这个 case 是否需要 context resolution
- 这些类型的 case 在不同 retrieval mode 下表现如何

## Running

运行 benchmark：

```bash
cd backend
. .venv/bin/activate
python tests/run_rag_benchmark.py
```

只跑前 5 个 case：

```bash
cd backend
. .venv/bin/activate
python tests/run_rag_benchmark.py --case-limit 5
```

走 HTTP：

```bash
cd backend
. .venv/bin/activate
python tests/run_rag_benchmark.py --transport http --case-limit 5
```

只看指定模式：

```bash
cd backend
. .venv/bin/activate
python tests/run_rag_benchmark.py --mode mix --mode mix_rerank
```

输出完整 JSON：

```bash
cd backend
. .venv/bin/activate
python tests/run_rag_benchmark.py --json
```

## Notes

当前 fixture 里的 `expected_chunk_ids` 已经从标题/关键词启发式切到了 typed schema，但它们仍然需要和真实标注的 chunk id 对齐。

后续如果要把 benchmark 做成真正稳定的回归集，最重要的是：

- 把每个问题的 `expected_chunk_ids` 和真实 corpus 中的 chunk id 对齐
- 继续扩充 relation / context resolution 类 case
- 再在此基础上做更高级的 graph-enhanced retrieval 对比

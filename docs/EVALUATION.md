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

当前 benchmark 里的 graph 对比使用两条路径：

- `mix`: `graph_mode=off`，作为 graph baseline
- `mix_graph`: `graph_mode=auto`，只在 query analysis 判断合适时启用 graph 路径

## Benchmark Schema

每个 case 现在使用 typed schema，而不是 `title_contains` / `keyword_any` 这类启发式判定。

```json
{
  "id": "skill-install-01",
  "question": "codex skills 安装",
  "expected_source_titles": ["Codex Skills 安装指南"],
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
- `expected_chunk_ids`: 这个问题应该命中的一个或多个 chunk，必须是精确 chunk id
- `expected_source_titles`: 当仓库里还没有可直接引用的 chunk id 时，使用精确 source title 作为稳定回退
- `question_type`: 问题类型，例如 `installation`、`path_lookup`、`relation`
- `difficulty`: `easy` / `medium` / `hard`
- `source_type`: 数据来源类型，例如 `chat`
- `requires_relation_reasoning`: 是否需要关系推理
- `requires_context_resolution`: 是否需要上下文消解

每个 case 只需要提供一种 ground truth 入口：

- 有精确 chunk id 时，填 `expected_chunk_ids`
- 没有可直接复用的 chunk id 时，填 `expected_source_titles`

runner 会优先按 chunk id 评测；如果 case 没有 chunk id，则按 source title 做精确匹配。这里仍然是 typed ground truth，不回退到标题包含或关键词包含那种启发式规则。

## Metrics

当前评测骨架提供 3 个核心指标：

- `Recall@K`: 前 `K` 个结果里命中的相关 chunk 占所有相关 chunk 的比例
- `HitRate@K`: 前 `K` 个结果里是否至少命中一个相关 chunk
- `MRR@10`: 前 10 个结果里第一个相关 chunk 的倒数排名

其中：

- `Recall@5` 和 `Recall@10` 用于衡量召回覆盖面
- `HitRate@5` 用于衡量最少命中能力
- `MRR@10` 用于衡量相关结果排在前面的程度

除了主检索指标，当前 summary 还会输出 graph 相关的辅助指标：

- `graph_routed_cases`: 本轮 benchmark 中实际走到 graph 路径的 case 数量
- `avg_graph_hits`: 平均每个 case 获得的 graph candidate 数

这些指标不替代 Recall/MRR，它们只用于解释 graph 路径究竟有没有真正被触发。

## Runner Behaviour

`evaluate_retrieval_case` 会：

- 接收 `BenchmarkCase`
- 将检索结果归一化成可比较的 ranked identifiers
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

默认情况下脚本会取 `top_k=10`，这样 `Recall@10` 和 `MRR@10` 才是按完整前 10 个结果计算的。当前 fixture 使用 `expected_source_titles` 作为 ground truth，所以这些指标衡量的是“前 10 个结果里是否命中该 source title 对应的真实聊天”。

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
python tests/run_rag_benchmark.py --mode mix --mode mix_graph --mode mix_rerank
```

输出完整 JSON：

```bash
cd backend
. .venv/bin/activate
python tests/run_rag_benchmark.py --json
```

## Notes

当前 fixture 使用的是精确 `expected_source_titles`，因为仓库里没有公开的真实 chunk 标注可以直接写进 fixture。等后续补齐稳定的 chunk 标注后，可以把对应 case 切回 `expected_chunk_ids`，runner 会继续兼容。

如果你把 `--top-k` 调小到 10 以下，`Recall@10` 和 `MRR@10` 仍然会显示，但它们只会基于实际返回的候选结果计算，不再代表“完整前 10 条”的评测。

后续如果要把 benchmark 做成真正稳定的回归集，最重要的是：

- 把每个问题的 `expected_chunk_ids` 和真实 corpus 中的 chunk id 对齐
- 继续扩充 relation / context resolution 类 case
- 再在此基础上做更细的 graph-enhanced retrieval 对比，例如 relation question 的单独 win-rate

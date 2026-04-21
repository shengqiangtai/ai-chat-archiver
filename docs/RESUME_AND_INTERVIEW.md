# Resume And Interview Notes

这份文档不是用户文档，而是求职材料。

目标是把这个项目稳定地讲成：

- `成熟的 LLM 应用工程项目`
- `有检索优化深度的 RAG 项目`
- `不是只会接 API 的 Demo`

## 一句话介绍

> 我做了一个本地优先、图增强的轻量级 RAG 系统，面向 AI 聊天记录和 Markdown 知识库，围绕 hybrid retrieval、graph-assisted retrieval、rerank、citation 和 grounding 建立了离线评测闭环。

## 简历版本

### 版本 A：通用大模型开发岗

- 设计并实现本地优先的 Graph-Enhanced RAG 系统，支持 AI 聊天记录归档、增量索引、混合检索、引用问答与前端调试面板
- 基于 `SQLite FTS + Chroma` 构建 hybrid retrieval 主链路，并引入轻量 `entity / relation` 图元数据增强检索，提升关系类查询的召回能力
- 建立 typed benchmark 与离线评测框架，使用 `Recall@K`、`MRR`、graph route rate 等指标比较 `vector / hybrid / mix / mix_graph / mix_rerank`
- 在回答阶段增加 citation parsing 与 grounding checks，对支撑不足的回答自动降级，降低 unsupported answer 风险

### 版本 B：算法 / 检索岗

- 构建轻量级双层检索 RAG 系统：chunk-level 走 `dense + keyword + entity`，concept-level 走 graph-assisted retrieval，并通过可解释 fusion 合并候选
- 设计 query-aware routing，对查询进行 `symbolic / relation / follow-up / semantic` 分类，控制 graph gating 与 rerank gating，避免高成本路径污染精确查询
- 设计并实现离线 benchmark，分离评估 `检索质量` 与 `回答可信度`，支持对 relation-like query 的 graph routing 行为进行对比分析

### 版本 C：全栈 / LLM 应用工程岗

- 独立完成浏览器插件归档、FastAPI 检索问答后端、React 调试前端和本地向量/关系索引集成
- 在语义搜索和 QA 页面暴露 query rewrite、query analysis、graph routing、rerank 和 grounding 状态，增强系统可解释性与调试效率

## 项目亮点怎么讲

### 1. 为什么不是普通 RAG Demo

你可以直接说：

“这个项目的重点不是接一个向量库然后调用模型回答，而是把检索增强和工程可解释性做完整。它有 baseline、有 graph-enhanced retrieval、有 rerank、有 grounding、有 benchmark，所以我能明确知道每一步到底有没有带来收益。”

### 2. 为什么要做 graph-enhanced retrieval

推荐讲法：

“很多 AI 聊天和技术文档问题并不是纯语义近邻检索能解决的，尤其是组件关系、依赖关系、工具关联这种查询。  
所以我没有直接上重型 GraphRAG，而是把 entity / relation 元数据落到 SQLite，用一个轻量 graph candidate retriever 去增强 chunk retrieval，这样既保留本地部署的轻量性，也能在 relation query 上提升召回。”

### 3. 为什么要做 query-aware routing

推荐讲法：

“graph retrieval 和 rerank 都不是越多越好。  
如果是显式文件名、命令名这类 symbolic query，graph 路径反而可能引入噪声。  
所以我先做 query analysis，再决定要不要开 graph path、要不要开 rerank，这样系统既稳又更容易解释。”

### 4. 为什么要做 grounding

推荐讲法：

“检索提升只是第一层，真正上线时更重要的是回答不要乱答。  
所以我在生成后增加了轻量 grounding checks：如果回答和来源片段重叠不足，就把它降级成保守回答，而不是让模型继续强行总结。”

## 面试讲法

### 30 秒版本

“我做了一个本地优先的 Graph-Enhanced RAG 系统。底层是 `SQLite FTS + Chroma` 的 hybrid retrieval，上面加了一个轻量 graph-assisted retrieval 层，用 query-aware routing 决定什么时候启用。项目里还有离线 benchmark、citation 和 grounding checks，所以我可以从检索效果和回答可信度两个维度解释优化收益。”

### 2 分钟版本

“这个项目的输入主要是 AI 聊天记录和 Markdown 文档。  
索引阶段我会做 chunking、embedding、实体抽取和轻量关系抽取，把 chunk-level 数据写进 SQLite + Chroma，把 entity/relation 元数据写进 SQLite。  
查询阶段不是单一路径，而是先做 query analysis，再走 dense、keyword、entity 这些 baseline 路径；如果查询更像 relation query，就启用 graph-assisted retrieval。  
所有候选会经过可解释 fusion，必要时再做 rerank。  
生成回答时必须带 citation，并且会做 grounding checks，如果来源支撑不足就自动降级。  
同时我还做了离线 benchmark，比较不同 retrieval path 的 Recall/MRR，以及 graph route rate 这类辅助指标，所以不是凭主观感觉调参。” 

### 5 分钟版本

建议按这 5 段讲：

1. `问题定义`
   - 聊天知识库的 query 既有精确文件查询，也有 relation-like query
   - 单一路径检索不够稳

2. `系统设计`
   - hybrid retrieval 主链路
   - graph-enhanced retrieval 副路径
   - query-aware routing 控制成本和噪声

3. `关键取舍`
   - 不上重型 GraphRAG
   - 不上独立 graph DB
   - graph 只做 augment，不替代 textual evidence

4. `评测和可解释性`
   - typed benchmark
   - retrieval 和 trustworthiness 分开评估
   - 前端 debug 暴露 query_analysis / graph / rerank / grounding

5. `结果和价值`
   - 项目既能跑，也能讲清楚每层优化为什么存在
   - 更接近真实 LLM 系统，而不是纯研究原型

## 面试高频问题

### 为什么不用 Neo4j / GraphDB？

答：

“这个项目的目标不是做企业级 GraphRAG，而是做一个本地优先、可部署、可解释的轻量系统。  
把 entity / relation 元数据落在 SQLite 已经足够支持轻量 graph retrieval，而且部署复杂度低很多。  
如果一开始就上图数据库，会让工程复杂度超过项目想证明的核心问题。”

### graph retrieval 和 entity retrieval 的区别是什么？

答：

“entity retrieval 更像基于命中的实体去找 chunk；  
graph retrieval 则会利用 relation metadata，把和这些实体存在依赖、使用、关联关系的 chunk 也带进候选集。  
前者更像 direct entity hit，后者更像 relation-neighborhood expansion。”

### 为什么 graph path 不默认强开？

答：

“因为显式路径、文件名、命令查询更适合精确检索，graph 路径可能引入噪声。  
我把它做成 query-aware gated path，本质上是收益/成本/噪声的平衡。”

### 你怎么证明 graph path 有价值？

答：

“不是看单条 case，而是看 benchmark。  
我会比较 `mix` 和 `mix_graph`，同时记录 graph route rate，确保 graph 路径真的被触发，而且主要在 relation query 上体现收益。”

### grounding 是不是很弱？

答：

“是的，它不是严格的 claim verification。  
但这是一个刻意的工程取舍：我先做轻量、可复现、可落地的 grounding checks，把 unsupported answer 降下来，而不是一开始就上重型 judge 模型。”

## 如果要量化结果

如果你后面补了真实 benchmark 数字，可以直接把下面这段替换掉：

- `Recall@5`：从 `X` 提升到 `Y`
- `MRR@10`：从 `X` 提升到 `Y`
- relation query 上 `mix_graph` 相比 `mix` 的 win rate：`X%`
- unsupported answer rate：从 `X` 降到 `Y`

如果还没有稳定数值，不要在简历里编。  
先写“建立了离线 benchmark 与评测闭环”，等数字稳定后再替换成量化结果。

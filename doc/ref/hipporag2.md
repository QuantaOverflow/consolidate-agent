# HippoRAG2 参考笔记

> 整理时间：2026-05-13
> 用途：评估是否走 HippoRAG2 路线作为本项目 GraphRAG 实现
> 来源：GraphRAG-Bench (ICLR'26) / HippoRAG NeurIPS'24 paper / OSU-NLP-Group 开源实现

## 1. 一句话本质

**OpenIE 三元组 → 知识图 → Personalized PageRank（PPR）**，用图上的随机游走代替 ER 查询，实现多跳关联检索。

不需要清洁的 ontology，不需要 community 检测，不需要 entity identity 完美——是 GraphRAG 路线里最"轻"也最有效的。

## 2. 生物学动机

模拟人脑海马体（Teyler-Discenna memory indexing theory）：

- **模式分离（pattern separation）**：每段经验独立存储 → 每条 record / 三元组独立存
- **模式完成（pattern completion）**：从片段线索重构完整记忆 → PPR 从查询 seed 沿图传播激活整片关联

PPR 就是模式完成的算法实现。

## 3. 算法详解

### 索引阶段（offline，一次性）

**Step 1 — OpenIE 抽三元组**
LLM 从每条 passage 抽 `(subject, predicate, object)`。

**Step 2 — 构建知识图**
- 节点：所有抽出的 entity
- 边：每个三元组形成一条带类型的有向边

**Step 3 — 加 synonym edges（关键）**
- 给每个 entity 计算 embedding
- 两两 cosine sim 超过阈值（典型 0.8-0.85）→ 加一条 synonym 边
- 作用：让 `"alzheimer's disease"` 和 `"Alzheimer"` 在图上**邻居**，但不强制合并
- **优雅地绕过了 entity identity resolution 死结**

**Step 4 — 算 passage-node 矩阵 P**
- 矩阵：行=node，列=passage
- `P[node, passage] = node 在 passage 中出现的频次`
- 作用：把"节点重要性"映射回"passage 重要性"

### 查询阶段（online，每次查询）

**Step 1 — 从 query 抽实体**
LLM 从查询里抽 named entities。

**Step 2 — Entity linking**
查询实体 → embedding → 图节点最相似的几个 → 这些是 PPR 的 **seed nodes**

**Step 3 — 跑 Personalized PageRank**

标准 PageRank 是随机游走的 stationary distribution。Personalized 版本里，游走时有 α 概率瞬间跳回 seed 节点。

一步迭代的语义：
```
score[node] = α × seed_prior[node] + (1-α) × Σ score[neighbor] / degree[neighbor]
```

- 离 seed 越近、可达路径越多的节点，分越高
- α（teleport prob）一般 0.1-0.5，控制游走传播多远
- **天然处理多跳**：30-50 次迭代后，2-3 跳外的相关节点也被激活

**Step 4 — 节点分 → passage 分**

`passage_score = P^T × node_score`

即"passage 里出现的高分节点越多，passage 越相关"。

**Step 5 — 返回 top-K passages**
排序，取前 K 条。可选用 reranker / LLM 二次筛。

## 4. HippoRAG v1 → v2 的进化

| 维度 | v1 | v2 |
|------|----|----|
| 图节点 | 只有 entity | entity + passage 混合 |
| 检索单位 | 间接（节点 → 矩阵 P）| 直接（passage 也参与 PPR）|
| 在线 LLM 用法 | query entity 抽取 | + filter/rerank |
| associative memory | 基线 | +7% |
| sense-making | 弱 | 大幅提升 |

v2 核心变化：**passage 作为一等公民放进图**。每个 passage 节点连接到它里面的所有 entity。查询时 PPR 在 entity+passage 混合图上跑。

## 5. 实测性能（GraphRAG-Bench, ICLR'26）

### 复杂推理任务 accuracy

| 方法 | 小说数据集 | 医学数据集 | Token/query |
|------|-----------|-----------|------------|
| **HippoRAG2** | **53.38%** | **61.98%** | ~1,000 |
| LightRAG | 49.07% | 61.32% | ~100,000 |
| MS-GraphRAG | 50.93% | 47.04% | ~331,000 |
| 基础 RAG | 42.93% | 58.64% | ~880 |
| RAPTOR | 38.59% | 53.20% | ~3,400 |

### Evidence Recall（多跳题）

- HippoRAG 在 Level 2-3 题目上：**87.9-90.9%**
- HippoRAG2 在 Context Relevance：**85.8-87.8%**
- 基础 RAG：59.8-64.5%

### 关键结论

- **图密度与性能正相关**：HippoRAG2 平均 598 节点 / 10K corpus token，3,979 边 / 10K token
- **图不是万能的**：简单事实检索（Level 1）基础 RAG 优于 GraphRAG
  - 小说 fact retrieval：基础 RAG 60.92% vs MS-GraphRAG 49.29%
  - 医学：基础 RAG 64.73% vs MS-GraphRAG 38.63%
- **MS-GraphRAG 路线在退潮**：成本太高、social community summary 引入冗余
- **HippoRAG2 是 2026 共识答案**：质量+成本双领先

## 6. 适用判断

### HippoRAG2 适合

- 数据有自然 entity（人名、工具名、概念名）可作为图节点
- 查询需要多跳推理或关联检索
- 数据规模中等（千-万级 passage）
- 增量更新友好（笔记会持续增长）
- 想低成本部署

### HippoRAG2 不适合

- 简单事实检索（基础 RAG 更好）
- 数据无明显 entity（纯叙事文本、纯代码）
- 需要严格 ER 推理（如医疗诊断需要规则推导）
- **跨领域抽象类比**（PPR 是"图上关联游走"，不是"高层概念抽象"）

## 7. 关键参数与超参

| 参数 | 典型值 | 调节方向 |
|------|--------|---------|
| Synonym edge cosine 阈值 | 0.8-0.85 | 高→召回低；低→噪音多 |
| PPR teleport α | 0.1-0.5 | 高→局部；低→全局 |
| PPR 迭代次数 | 30-50 | 多→远距离激活更充分 |
| 每 query 抽取的 seed 数 | 3-10 | 多→更鲁棒；过多→噪音 |
| Top-K passages | 5-20 | 看下游 LLM 上下文预算 |

## 8. 开源实现

- **官方**：[OSU-NLP-Group/HippoRAG](https://github.com/osu-nlp-group/hipporag)
  - 默认假设英文 wikipedia 风格 passage
  - 索引流水线会重做 OpenIE
- **扩展**：[CatRAG](https://github.com/kwunhang/CatRAG)（基于 HippoRAG2，加 query-adaptive navigation）
- **PPR 引擎**：`networkx.pagerank(personalization=...)` 现成可用

## 9. 对本项目的应用判断

### 现有资产对照

| HippoRAG2 需要 | 本项目已有 | 缺什么 |
|--------------|----------|--------|
| Passages | 458 records (`source_knowledge_records`) ✅ | - |
| 三元组（OpenIE）| v6_full triples ✅ | 抽取粒度偏抽象，密度偏低 |
| Entity embeddings | normalize spike 算过 ✅ | 需要完整重算 |
| Synonym 边 | normalize cluster 可改造 ✅ | 改成"边"而不是"合并" |
| Passage 节点 + entity-passage 边 | ❌ | 新建 |
| PPR 引擎 | ❌ | `networkx.pagerank` |
| Query entity 抽取 prompt | ❌ | 一个 LLM prompt |

### 密度对比（关键风险）

- 本项目：5.7 entity / 150 token insight ≈ **380 entity / 10K token**
- HippoRAG2 best perf：**598 entity / 10K token**
- 本项目密度是论文 best 的 **~0.63 倍**

**密度偏低的根本原因**：

1. record 本身已是 distilled 后的"一句话洞见"（avg 392 字 insight），不是原始文本
2. v6_full 抽取 prompt 只要"抽象概念"（Pattern/Mechanism 等 v0.2 ontology 类别），主动过滤具体提及
3. 有硬上限 max_entity_count=10

**修复方向**：

- 重抽时不再限制只抽抽象，把具体工具名、文件名、错误名、动作名都抽进来
- 三元组用宽松关系（mentioned_with、located_in 等都算）
- 保留 v6_full 作为高质量带语义的子图，HippoRAG2 用新的密图做检索

### 工作量估算

| 方案 | 工作量 | 备注 |
|------|--------|------|
| **A. 复用 v6_full 直接套** | 13-18h | 密度偏低，效果可能打折，但能快速验证方向 |
| **B. 重抽 + 套 HippoRAG2** | 16-22h | 解决密度问题，效果上限更高 |

建议：**先 A 做 dry-run 验证 PPR pipeline 走得通，再决定要不要 B**。

### 局限性提醒

1. **不解决跨域抽象类比**：PPR 是图上关联游走，不是高层概念抽象。如果 `codex.scope_management` 和 `langgraph.scope_management` 在图上没有桥接路径，PPR 仍连不起来。
2. **synonym 边阈值难调**：要花时间调。
3. **数据本身不是 narrative**：每条 record 已自包含，没有跨 passage 的自然链接，PPR 多跳效果取决于 entity 共现密度。

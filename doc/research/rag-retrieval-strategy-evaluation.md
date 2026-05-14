# RAG 召回策略评估报告

**日期**: 2026-05-06  
**场景**: session turn-level 检索，用于 Evidence Agent 验证 knowledge insight 的证据 turn  
**评估脚本**: `scripts/eval_rag_ab.py`

---

## 问题描述

Knowledge 提取 pipeline 产出 insight（抽象原则），需要在 session turn 中找到支撑证据。query（insight）和 document（turn 内容）存在两类语义 gap：

1. **抽象层次差异**：insight 是通用工程原则，turn 是具体操作对话
2. **语言差异**：insight 为英文，turn 内容中英混合，大量中文

---

## 评估方法

- **数据集**：3 个 golden session，9 条 knowledge records，每条有 LLM 标注的 `evidence_turns`
- **指标**：Recall@5 = |retrieved ∩ evidence_turns| / |evidence_turns|
- **Ground truth 说明**：evidence_turns 由 LLM 生成，有噪声，但是当前唯一可用的标注

---

## 策略与结果

| 策略 | avg Recall@5 | 描述 |
|---|---|---|
| A | 0.68 | baseline：raw XML embedding + cosine |
| B | 0.71 | turn summary embedding（LLM 生成摘要后 embed）|
| C | 0.67 | BM25 + vector hybrid（EnsembleRetriever + RRF）|
| D | 0.73 | query 翻译成中文后 cosine |
| **E** | **0.83** | **query 翻译 + summary embedding（最优）** |
| F | 0.74 | query 翻译 + BM25+vector hybrid |

---

## 关键发现

### 1. 语言对齐是最有效的单一改进

Strategy D（query 翻译，+0.05）优于 Strategy B（turn summary，+0.03）。根本原因是 embedding 模型虽然支持多语言，但跨语言的抽象词汇匹配效果不如同语言匹配。

### 2. query 翻译 + summary embedding 有正向叠加效应

Strategy E = D + B = +0.15，超过两者之和（+0.08）。原因：翻译后的中文 query 和中文 turn 摘要在同一语义空间，匹配精度更高。

### 3. BM25 hybrid 对本场景无明显收益

Strategy C（0.67）略差于 baseline（0.68）。原因：BM25 对关键词匹配有效，但 insight 的关键词（如 "polling heuristics"）在 turn 中以完全不同的词汇形式出现（"心跳轮询检测机制"），BM25 无法跨语言匹配。在 query 翻译后（Strategy F，0.74），BM25 的贡献也只是边际改善。

### 4. 高度抽象的跨语言 insight 是最难召回的

HITL 那条（"HITL recovery semantics must be decoupled from polling heuristics"）：
- Strategy A/B/C：Recall@5 = 0.25
- Strategy D/F：Recall@5 = 0.75  
- Strategy E：Recall@5 = **1.00**

summary embedding 把 turn 的核心语义提炼成抽象中文摘要，配合中文 query，完全解决了这个问题。

### 5. 简单操作类 session 任何策略都召回很好

019d80f4 的三条记录全部 Recall@5 = 1.0，无论哪个策略。说明策略选择主要影响高度抽象 insight 的召回。

---

## 结论与建议

**落地策略：Strategy E**（query 翻译 + summary embedding）

实施要点：
- `embed_session` 时为每个 turn 生成中文摘要（一次性，约 20 turn/session × LLM 调用）
- `search_turns` 调用前翻译 English query 为中文（per search，1 次 LLM 调用）
- `get_turn` 仍返回完整 turn 原文（不受影响）

成本评估：
- 索引时：303 session × 20 turn × 1 LLM = ~6000 LLM 调用（一次性）
- 检索时：每次 search +1 LLM 翻译调用

---

## 局限性

- 评估集只有 9 条记录，样本量小，结论有统计不确定性
- Ground truth（evidence_turns）由 LLM 生成，本身有噪声
- 仅测试了 golden sessions，不代表全量 303 session 的分布
- 对全英文 session 的效果未单独评估

---

## 未评估的方向

- **Multi-query Retrieval**：生成多个查询变体，合并结果
- **Step-back Prompting**：抽象 query 再检索
- **LLM Reranking**：cosine 粗召回后 LLM 精排
- **MMR（Maximal Marginal Relevance）**：相关性+多样性平衡

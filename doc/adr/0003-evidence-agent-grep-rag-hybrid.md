# ADR-0003: Evidence Agent 使用 grep-first + RAG 组合检索策略

**日期**：2026-05-07  
**状态**：已采纳

## 背景

Evidence Agent 需要在 session turns 中找到支撑某条 knowledge insight 的具体证据。有两种检索方式可选：

- **RAG（语义向量检索）**：适合抽象语义匹配，能跨语言、跨表达形式召回相关内容，但对精确技术标识符的匹配效果不稳定。
- **Grep（关键词/正则检索）**：确定性、零成本、零延迟，适合精确命中函数名、错误信息、工具名等标识符，但无法处理语义层次的匹配。

RAG 评估（`eval_strategy_e.py`）显示 Hit@5=0.95，但仍有 5% 的 case 完全召回失败，且这些 case 的 insight 往往提到了具体的技术标识符（如「cross-process browser」、「AbortController」），这类词在 turn 里以原始形式出现，grep 能直接命中。

灵感来源：Claude Code 在代码库探索时优先使用 grep 而非语义搜索，因为代码和技术日志里有大量精确标识符，字符串匹配比向量匹配更可靠。

## 决策

**Evidence Agent 采用 grep-first + RAG 组合策略，两者互补，不互相替代。**

搜索动作：
1. `search_text(pattern: str)`：对 session XML 所有 turns 做关键词/正则搜索，返回匹配的 turn index 列表。确定性，零 API 调用。
2. `search_turns(query: str, top_k: int)`：RAG 语义检索，返回向量相似度最高的 turn index 列表。
   semantic score 使用 Chroma raw distance 在本次检索结果内做 0-1 相对归一化，不使用
   LangChain `similarity_search_with_relevance_scores()` 的全局 relevance 映射。
3. `get_turn(turn_index: int)`：读取指定 turn 的完整内容。

Agent 工作流采用 Plan-Execute-Judge，而不是 ReAct tool loop：
1. Plan：LLM 根据 insight 生成 1-4 个 `text` / `semantic` 搜索动作
2. Execute：确定性执行搜索，对候选 turn 调 `get_turn` 读取完整内容
3. Judge：LLM 判断 admit / reject / need_more
4. Replan：仅当 `need_more` 时执行一次补充搜索
5. Final Judge：最终 LLM 判断只能 admit / reject，不能继续搜索

Semantic score 处理：

- Chroma 返回的 raw distance 语义是“越小越相关”。
- `search_turns` 对同一次 query 返回的候选做 min-max 归一化：

  ```text
  score = 1 - (distance - min_distance) / (max_distance - min_distance)
  ```

- 当本次结果只有一个 distance，或所有 distance 相同时，score 记为 `1.0`。
- 该 score 只表示“同一次检索内部的相对排序”，不能跨 query 比较，也不能作为
  admit/reject 的置信度阈值。
- 不使用 LangChain 的 relevance score 映射，因为当前 Chroma collection 使用 L2
  distance，而 LangChain 的 L2 relevance 公式假设归一化 embedding：

  ```text
  relevance = 1 - distance / sqrt(2)
  ```

  DashScope embedding / Chroma 返回的 L2 distance 可能超过 `sqrt(2)`，导致负分和
  warning；warning 会打印命中的 Document 内容，存在泄露 session 原文的风险。

## 理由

- **互补性**：RAG 擅长语义层次（「跨进程状态隔离」），grep 擅长字面层次（`playwright`、`process`、具体错误信息）。两者的召回集合有交集但不完全重叠。
- **成本**：`search_text` 是纯字符串搜索，无 LLM 调用、无 embedding API，几乎零成本。RAG 已经有 Chroma 索引，也是低成本。
- **可解释性**：grep 命中的 turn 有明确的命中原因（哪个关键词），比向量相似度更可解释。
- **覆盖 RAG 盲区**：行动类 turn（bash 命令输出、file_edit）的摘要语义离抽象 insight 较远，RAG 容易漏掉，但 turn 里往往有具体的工具名或错误信息，grep 能直接找到。

## 后果

- Evidence Agent 依赖三个检索接口：`search_text`（新增）、`search_turns`（已有）、`get_turn`（已有）
- `search_text` 实现在 `SessionTurnStore`，输入 session XML 和 pattern，输出匹配 turn 列表
- `search_turns` 应使用 raw distance 接口并自行归一化，避免 LangChain relevance warning
  泄露原始 turn 内容
- 不改变 RAG 基础设施（Strategy E embedding 继续使用）
- LLM 调用上限固定为 3 次，消除 ReAct recursion limit 风险

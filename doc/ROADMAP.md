# Knowledge Pipeline Roadmap

## 已完成

### Sprint 1 — Pitfall 提取基础管道
- chunk-based pitfall 提取 + admission
- canonical 化 + TagGoverAgent
- 增量处理（ProcessedIndex）

### Sprint 2 — 规模化与整合
- 分块 session 处理
- 知识分类法（knowledge taxonomy）工作流
- governed tag 去重

### Sprint 3 — Knowledge Embedding 检索
- KnowledgeVectorStore：canonical pitfall + knowledge record embedding
- `find_related_knowledge`：给定 pitfall，找相关 knowledge（US-2）
- `search_knowledge`：CLI `--query` 文本检索接口
- 架构重构：DRY embedding CRUD、SRP 修复、N+1 修复

### Sprint 4 — Session 知识提取管道
- SessionContextEngineer：JSONL → 压缩 XML（97-99% 压缩率）
- KnowledgeExtractor：session 级别单次提取
- 双路并行：pitfall pipeline + knowledge pipeline
- `--extract-knowledge` CLI + `--embed` CLI

### Sprint 5 — 提取质量提升（Prompt B）
- 收紧提取标准：必须有 trial-and-error 证据，minimum 2 evidence turns
- 过滤纯问答 session（无 `<bash>` 或 `<file_edit>` 的 session 跳过）
- Golden set 建立：5 个代表性 session + 逐条人工标注
- A/B 测试：1250 条（Prompt A）→ 395 条（Prompt B），evidence=1 占比从 46.5% 降到 9.6%

### Sprint 0 — Session Turn Embedding 基础设施
**目标**：建立通用的 session turn-level RAG 基础设施，供 Evidence Agent、大 session 分层摘要、未来 session 检索复用。

**范围**：
- 新建基于 Chroma 的 `session_turns` 向量库（替代 SQLite）
- `embed_session`：为 session turn 生成 embedding
- `search_turns(session_id, query, top_k)`：embedding 检索
- `get_turn`：读取 turn 原文

**验收**：对已处理的 session 跑 embed，用 `search_turns` 查几个 query 看召回质量。

### Sprint 6 — Evidence Agent
**目标**：用 agent 替代 LLM 幻觉生成的 evidence_turns，让 evidence 字段真正可信。

**背景**：
- 当前 `evidence_turns` 是 LLM 自填，不可验证（已在 schema 清理阶段移除）
- Evidence Agent 用 bounded Plan-Execute-Judge 流程在 session 里找证据，找到 → admit，找不到 → reject
- 见 ADR-0003：grep-first + RAG 组合检索策略

**搜索接口设计**（见 ADR-0003）：
- `search_turns(query, top_k)`：RAG 语义检索，召回语义相关 turns
- `search_text(pattern)`：关键词/正则精确检索，零成本，适合精确标识符
- `get_turn(turn_index)`：读取指定 turn 完整内容

**Agent 工作流**：
1. Plan LLM 生成 1-4 个 text / semantic 搜索动作
2. 确定性执行搜索，并对候选 turn 调 `get_turn` 读完整内容
3. Judge LLM 判断：admit / reject / need_more
4. `need_more` 最多触发一次补充搜索
5. Final Judge 只能 admit / reject，不能继续搜索

**范围**：
- 新增 `src/consolidate_agent/knowledge/evidence_agent.py`
- 新增 `src/consolidate_agent/prompts/evidence_plan_system.md`
- 新增 `src/consolidate_agent/prompts/evidence_judge_system.md`
- pipeline 集成：`run_knowledge_extraction` 在 admit 后接 Evidence Agent
- `KnowledgeRecord.evidence_turns` 由 Evidence Agent 写入，`evidence_count` 从中计算

**验收**：
- golden set 中 Conventional Commits 那条（`019d491d`）被 reject（找不到证据）
- 其余有效记录 `evidence_turns` 可追溯到真实 turn 内容
- Hit@Evidence ≥ 0.90（在 golden set 上，Evidence Agent 的最终 admit 结果与 golden set 一致）

### Sprint 6.5 — RAG 策略升级（Strategy E）
**目标**：将 `search_turns` 升级为 query 翻译 + summary embedding，从 Recall@5=0.68 提升到 0.83。

**背景**：详见 `doc/research/rag-retrieval-strategy-evaluation.md`。评估结论：query 翻译 + summary embedding（Strategy E）是最优组合，两个改进有正向叠加效应。

**时机**：在 Sprint 7（大 session）开始前实施，因为大 session 必须依赖 RAG，此时 search 质量才有实际消费方。

**范围**：
- `embed_session`：为每个 turn 生成中文摘要，用摘要做 embedding（替换 raw XML）
- `search_turns`：调用前翻译 English query 为中文（+1 LLM 调用/search）

---

## 待做

### Sprint 7 — 大 Session 分层摘要 ⬅️ 下一个
**目标**：覆盖当前被跳过的部分大 session（> 100k chars），这些是最复杂、最有价值的工作记录。

**方案**：
- 按 turn 分块（每块 ≤ 60k chars）
- 每块独立摘要（小模型，便宜）
- 合并摘要 → 提取知识候选
- 候选走 Evidence Agent 验证

**依赖**：Sprint 0、Sprint 6

### Sprint 8 — Refine Agent（待评估）
**目标**：对通过 Evidence Agent 的候选做进一步增强——精化 insight 措辞、纠正 scope、强化 applicability。

**决策点**：Sprint 6 完成后评估是否值得做。如果 Evidence Agent 后的质量已经足够好，此 sprint 可跳过。

**依赖**：Sprint 6

---

## 架构决策记录

见 `doc/adr/` 目录：
- [ADR-0001](adr/0001-filter-qa-only-sessions.md)：过滤纯问答 session
- [ADR-0002](adr/0002-dual-granularity-summarization.md)：双粒度摘要策略——per-turn 用于检索，semantic chunk 用于提取
- [ADR-0003](adr/0003-evidence-agent-grep-rag-hybrid.md)：Evidence Agent 使用 grep-first + RAG 组合检索策略

## 关键设计原则

- evidence 字段必须客观可验证，不依赖 LLM 自律
- 提取宽松，验证严格（admission 在 Evidence Agent 而非 prompt）
- 基础设施通用化优先（turn embedding 服务多个 sprint）
- 消费场景驱动质量优化方向（`--query` 检索体验是质量的最终判断）

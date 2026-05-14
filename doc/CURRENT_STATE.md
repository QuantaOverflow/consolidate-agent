# Current State

> 最后更新：2026-05-13
> 用途：项目当前真实状态的单一参考点。下次进项目先读这个，再决定怎么干。
> 维护：每次完成一个阶段（commit、决策、废弃方向）时同步更新。

## TL;DR

- 项目从 **pitfall 蒸馏** 已经迁移到 **knowledge record + KG** 方向
- 当前活跃问题：**评估 HippoRAG2 作为 GraphRAG 路线**
- 已经验证过的死路：跨域抽象类比（normalize spike 给出否定结论）
- 主线程上次会话尾停在：HippoRAG2 路线落地策略评估

## 数据现状

### 主数据库

**`outputs/knowledge.db`**（20MB）— 唯一权威。

真实表（2026-05-13 query 结果）：

| 表 | 行数 | 含义 |
|----|-----|------|
| `source_knowledge_records` | 696 | 知识 record（title/insight/applicability/scope）|
| `graph_nodes` | 2296 | v6_full 抽取的实体节点 |
| `graph_edges` | 2912 | v6_full 抽取的语义边 |
| `knowledge_tags` | 66 | 当前 tag 列表 |
| `knowledge_tag_assignments` | 657 | record↔tag 链接 |
| `processed_sessions` | 261 | 增量处理标记 |
| `canonical_knowledge` | 0 | 已清空（旧 pitfall canonicalize 产物）|
| `source_pitfall_records` | 0 | pitfall pipeline 已删 |

### 历史 DB（不要当主线读）

`outputs/` 下另有 10 个 `knowledge-*.db`（共 ~270MB），都是历史实验快照：

```
knowledge-prda.db (64M)       agent_graph 实验（含 concept_nodes/tool_domain_nodes/graph_evolution_log 表）
knowledge-validate.db (64M)   PRDA 验证产物
knowledge-prompt-v2/v3.db     prompt 迭代实验
knowledge-warmstart-test.db   热启动测试
knowledge-merge-test.db       合并测试
knowledge-fresh.db            一次 fresh 跑产物
knowledge-v2.db               早期版本
knowledge-audit-e2e.db        audit 端到端
knowledge-e2e-test.db         一般 e2e 测试
knowledge.db.bak/.before_audit/.dirty_post_audit_5-11   knowledge.db 的备份
```

待清理（低优先级）。读数据**只读 `knowledge.db`**。

### 关键探索资产（在 outputs/ 下）

| 资产 | 路径 | 用途 |
|------|------|------|
| v6_full 三元组（typed, 12 关系）| `outputs/spike_normalize/triples_v6_normalized.jsonl` | KG 语义层 |
| Entity normalize cluster | `outputs/spike_normalize/entity_clusters.json` + `entity_mapping.json` | 94 cluster + 371 canonical |
| Q3 cross-domain 验证 | `outputs/spike_normalize/q3_verification.md` | 否定结论 |
| Ontology v0.2 | `outputs/ontology_drafts/v0.2.md` | 8 type + 12 relation |

## 代码模块（按 git 状态分类）

### 已 commit / 稳定

```
src/consolidate_agent/
├── cli.py / config.py / types.py / observability.py / prompt_loader.py
├── knowledge/             evidence_agent, store, vector, session_turn_store, obsidian_export, consolidation
└── knowledge_extraction/  extractor, pipeline
```

对应 CLI stages（在 `cli.py` 的 `_STAGES`）：
- `extract` → knowledge_extraction.pipeline
- `verify` → knowledge.evidence_agent
- `tag` → knowledge.consolidation (tag 路径)
- `audit-plan` / `audit-tags` → knowledge.audit（部分已并入）
- `embed` → knowledge.vector
- `export` → knowledge.obsidian_export
- `full` → 全流程

### 未 commit / 活跃实验（main worktree 本地）

git status `??` 的三个目录，**从未进过任何分支的 commit**，是 main worktree 的本地实验：

| 目录 | 文件数 | 用途 | 对应 CLI stages | 决断 |
|------|-------|------|---------------|------|
| `src/consolidate_agent/graph/` | 5 | 3-type entity 二部图（早期 PoC）| `graph_extract`, `graph_inspect`, `seed_bootstrap` | 已被 v6_full + vocab_maintenance/ 超越，建议废弃 |
| `src/consolidate_agent/agent_graph/` | 14 | PRDA Pipeline（concept reuse + lifecycle）| `agent_extract`, `agent_inspect`, `agent_search`, `agent_evolve`, `agent_health`, `agent_govern`, `agent_check`, `migrate_tool_domain` | 工程思路被 vocab_maintenance/ 重做且更成熟，建议废弃 |
| `src/consolidate_agent/knowledge/audit/` | 5 | Audit Agent | （部分已并入 `--stage audit-*`）| 已 commit 一部分，剩余待整理 |

### 另一个 worktree 的活跃主线

**`worktree-multidim-tag-system` 分支**（路径 `.claude/worktrees/multidim-tag-system`）：

- 基线 commit：同 main 最新 `fa66622`
- 比 main 多 **38 个 commit**
- 全部工作集中在新模块 `src/consolidate_agent/vocab_maintenance/`
- **工程成熟度最高**：LangGraph StateGraph + checkpointer + TDD invariants + HITL gate + eval suite + structured logging + TECH_DEBT.md

模块结构：
```
src/consolidate_agent/vocab_maintenance/
├── network.py          TagRecordNetwork（vocab + record-tag assignments 作 first-class state）
├── bootstrap.py        冷启动：distill → synthesize → reverse_check
├── agent.py + graphs/  Agent loop as LangGraph StateGraph
├── measure.py          health metrics (hit_rate / coverage / co-occurrence)
├── diagnose.py         LLM 看 metrics 决定下一步 action
├── propose/            4 种 propose
│   ├── merge.py        合并语义相近 tag
│   ├── deprecate.py    废弃低用 tag
│   ├── new.py          提议新 tag
│   └── additive.py     新 tag 反向找老 record
├── apply.py            执行 proposal + consistency invariants
├── probes.py           diagnostic 探针
├── similarity.py       embedding 相似度
└── observability.py    structured run log
```

**核心**：维护多维 tag 词表的 governance lifecycle。把 vocab + record-tag assignments 当作一张 bipartite network state，agent loop 持续 measure → diagnose → propose → apply。

**数据**：读 `outputs/knowledge.db`，自己另起 JSON 持久化（不写回 knowledge.db）。

### 未提交 tests（git status `??`）

- `tests/unit/test_audit_*.py`（6 个）
- `tests/unit/test_tag_taxonomy.py`
- `tests/e2e/test_tag_taxonomy.py`

跟上面 audit 模块、tag 路径关联。

## Prompts 索引

`src/consolidate_agent/prompts/` 下 48 个 prompt，混着多代实验。按 pipeline 分组（未来可单独写 `doc/ref/prompts_index.md`）：

| Pipeline | Prompts |
|---------|---------|
| 当前 knowledge extract | `knowledge_extraction_*`, `turn_summarization_*`, `evidence_*` |
| 当前 tag consolidation | `knowledge_tag_assignment_*`, `knowledge_taxonomy_draft_*`, `rule_*` |
| v6_full 图抽取 | `graph_entity_extraction_*`（含 batch 变体）|
| agent_graph 实验 | `concept_decide_*`, `concept_perceive_*`, `seed_bootstrap_*` |
| 已废 pitfall 路径 | `canonicalization_*`, `taxonomy_*`, `taxonomy_governance_*`, `group_rewrite_*` |
| 查询/检索 | `query_translation_*` |

`canonicalization_*` 等 pitfall 路径 prompts 已是 dead weight，但不删，作为历史参考。

## 当前讨论线索（如果接手要继续）

### 已决断

- **走 KG 方向**（不只做向量检索）
- **不追求跨域抽象类比**（normalize spike 验证此路不通；详见 5-12 会话末段）
- **GraphRAG 选型倾向 HippoRAG2**（论文：HippoRAG NeurIPS'24 + GraphRAG-Bench ICLR'26 共识，详见 `doc/ref/hipporag2.md`）
- **vocab_maintenance/ 是当前工程主线**（worktree 上 38 commit），不是 graph/ 或 agent_graph/

### 未决断

1. **HippoRAG2 落地：Option A vs B**
   - A：用现有 v6_full 直接套 PPR，13-18h，dry-run 验证
   - B：重新设计 entity 抽取 prompt（不限抽象，密度从 380/10K → ~1200/10K），16-22h
2. **HippoRAG2 与 vocab_maintenance/ 的关系**：
   - 是新模块还是 vocab_maintenance/ 内加一层 graph retrieval？
   - vocab 维护的 tag 是否作为 HippoRAG2 entity 节点的一类（"topic anchor"）？
3. **三个未提交模块的去留**：建议正式废弃，转 archive
4. **历史 DB 清理**：低优先级

### 不再追的

- 跨域类比能力（KG 数据规模不够）
- Layer 2 模式族 ontology（成本高、收益不确定）
- MS GraphRAG 风格的社区检测（成本太高且 2026 benchmark 显示被 HippoRAG2 超越）
- Pitfall 提取（pipeline 已删）

## 关键参考

- `doc/ref/hipporag2.md` — HippoRAG2 算法详解（外部参考）+ 本项目应用判断
- `doc/research/pipeline_routes_evaluation.md` — 2026-05-13 四条 KG 路线对比 + archive 建议
- `doc/DEV_NOTES.md` — 开发笔记累积（详细但日志风格，不要当 spec 看）
- `doc/ROADMAP.md` — sprint 历史（用于看演进，不是当前 plan）
- `outputs/spike_normalize/q3_verification.md` — 跨域类比死路的实证
- `outputs/ontology_drafts/v0.2.md` — 历史 ontology 设计

## 更新规则

完成以下事件时更新本文档：
- commit 一个稳定 feature
- 废弃一个方向（写入"不再追的"）
- 新增一个未提交实验模块
- 主数据库结构变化（表增删、数量级变化）
- 决定走某个具体技术路径（如最终选定 HippoRAG2 A 或 B）

不需要每次小改动都更新——保持"30 秒拉直状态"的密度。

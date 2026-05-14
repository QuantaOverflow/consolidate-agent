# Tag Taxonomy 三代演化

> 状态：整条线废弃（2026-05-15）
> Legacy tag：`legacy/multidim-tag-system`（38 commit 的最成熟版）/ `legacy/pre-cleanup`（main 上的 KnowledgeTag governance 版）
> 关联 memory：`project_tag_taxonomy_lesson.md`

## 总览

| 代 | 模块 | commit 状态 | 工程成熟度 |
|---|------|-----------|----------|
| Gen-1 MechanismTag | `consolidation/taxonomy.py` | 已 commit（main）| PoC |
| Gen-2 KnowledgeTag + Governance | `knowledge/consolidation.py` + `knowledge/audit/` | 部分 commit | spike |
| Gen-3 Multidim vocab | `src/consolidate_agent/vocab_maintenance/`（独立 worktree）| 38 commit，活跃迭代 | **产品级** |

## Gen-1 — MechanismTag (single-dim flat vocab)

**做法**：所有 record 共享一个 tag 词表（`knowledge_tags` 表，66 个 tag），每条 record 至少打 1 个 tag。LLM 提议 + dedup（cosine sim + LLM judge）。

**失败原因**：
- tag 维度混了 Matter（"sqlite_storage"）/ Activity（"prompt_engineering"）/ Pattern（"fallback_strategy"），同一 record 想打多个维度时只能选一个，信息损失
- 词表混乱，hit_rate 低（657 assignment / 696 record，但分布严重长尾）

## Gen-2 — KnowledgeTag + Audit Governance

**升级动机**：用 agent governance（merge/deprecate/create proposal）治理 Gen-1 的混乱词表。

**做法**：`knowledge/consolidation.py` 73KB LangGraph pipeline；`knowledge/audit/` 提供 audit-plan / audit-tags / commit-proposal / trace。Prompts：`knowledge_tag_assignment_*`、`knowledge_taxonomy_draft_*`、`rule_*`。

**失败原因**：
- 治理本身不解决 vocab 维度选错的问题 — 在错维度上 merge/deprecate 只是重新洗牌
- audit pipeline 复杂度爆炸（5 个子模块 + 6 个测试文件），但 hit_rate 未显著改善
- agent loop 跑一次要几十次 LLM call，成本与收益不匹配

## Gen-3 — Multidim Vocab Maintenance

**升级动机**：放弃 single flat vocab，改为 **N 个 facet 各自一个独立词表**（faceted classification 思路）。把 vocab + record-tag assignments 当 first-class state（`TagRecordNetwork`），用 agent loop 持续 measure → diagnose → propose → apply 治理。

**工程成熟度**（项目内最高）：
- LangGraph StateGraph + checkpointer
- 4 种 propose：merge / deprecate / new / additive（新 tag 反向找老 record）
- TDD apply_proposal + consistency invariants
- HITL bootstrap gate
- Eval suite + structured run log
- TECH_DEBT.md 主动维护

**失败原因**（也是停止理由）：
- **bottom-up 聚类不浮现 cross-domain 共享**（详见 `kg-experiments.md` 的 Q3 验证）—— 维度对了，但 458-697 record 规模下，归并后 cross-domain 真共享只有 1 个
- vocab_maintenance 解决的是 tag 词表治理，不是 retrieval，**没有 passage-level 检索增强**
- 当 ingestion baseline 不依赖 tag 也能跑（context-engineering + embedding 已经覆盖检索基本需求），多维 vocab 的边际价值不足以支撑维护成本

## Lessons

1. **vocab 是 indexing 问题，不是分类问题**。先想清"检索时怎么用"，再决定要不要建 vocab。
2. **Faceted classification > flat vocab**，但仍解决不了维度本身选错；维度需要 retrieval 场景反推。
3. **Single-vocab governance 解决不了维度错的问题** —— Gen-2 的教训。在 Gen-1 词表上加 audit 治理是治标，应该早一步推翻维度。
4. **工程成熟度 ≠ 方向正确**。Gen-3 是项目里最严肃的工程产物（StateGraph + invariants + HITL + eval），但底层假设错了仍然要废弃。
5. **38 commit 沉没成本不应影响决策**。打 tag 冻结，干净分手。

## 留下什么资产

- `legacy/multidim-tag-system` tag：Gen-3 全部 38 commit
- `.claude/worktrees/multidim-tag-system/TECH_DEBT.md`：当时记录的已知风险（HITL 端到端未完全验证、ThreadPool race 等）
- `legacy/pre-cleanup` tag：Gen-1/Gen-2 的代码（包含 `knowledge/consolidation.py` 73KB + `knowledge/audit/`）
- `outputs/knowledge.db.legacy-2026-05-15.bak`：包含 `knowledge_tags`(66) + `knowledge_tag_assignments`(657) 数据

## 后续如果重启 tag 路线该怎么做

不要从 Gen-3 续命。先回答："tag 在哪个 retrieval 场景里被消费？怎么消费？" —— 答出来了再设计 vocab，没答出来不要开工。

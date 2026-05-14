# Pipeline Routes Evaluation

> 调研时间：2026-05-13
> 用途：对项目内 4 条 KG/检索路线做完整评估，给出去留建议
> 触发上下文：评估 HippoRAG2 落地路径时，发现还有 vocab_maintenance/ 这条已 commit 的主线

## 调研背景

项目里同时存在 4 条 KG/检索相关的代码路径，但没人系统对照过它们的关系。本次调研：
1. 摸清每条路径的实际流程与数据 schema
2. 评估各自工程成熟度
3. 给出留/废/合并建议
4. 锚定 HippoRAG2 的合理落地位置

## 4 条路线全景

### 路线 1：`src/consolidate_agent/graph/` — 早期 PoC

**状态**：main worktree 本地未 commit，从未进入任何分支历史

**核心数据结构**：3-type entity 二部图

```python
entity_type: Literal["tool", "concept", "domain"]      # 3 类固定
edge_type:   Literal["mentions", "illustrates", "in_domain"]  # 从 entity_type 机械映射
# GraphEdge 是 record_id ↔ node_id（无 entity↔entity 关系）
```

**对应 CLI stages**：`graph_extract` / `graph_inspect` / `seed_bootstrap`

**数据**：写入 `knowledge.db` 的 `graph_nodes` / `graph_edges` 表
- 当前数据：2296 节点（concept=1226 / domain=620 / tool=450）
- 边：2912（illustrates=1311 / in_domain=832 / mentions=769）
- 全是 record→entity 的二部边，**无 entity↔entity**

**流程**：
1. `seed_bootstrap`：抽 200 record 让 LLM 给出 concept 候选 + rationale，输出 `outputs/seed_candidates.md`（人看的报告，**不反馈给抽取器**）
2. `graph_extract`：每条 record 经 LLM 抽 3-类 entity → 按 (type, name) 唯一约束 dedup → upsert
3. `graph_inspect`：终端打印 stats

**identity 处理**：name 字符串归一 + INSERT OR IGNORE

**致命局限**：
- Schema 表达力极弱（3 类、3 关系）
- 无 entity↔entity 关系
- identity 靠字符串匹配，同义词会变多节点

### 路线 2：`src/consolidate_agent/agent_graph/` — PRDA 实验

**状态**：main worktree 本地未 commit

**核心思想**：**P**erceive → **R**etrieve → **D**ecide → **A**pply
- 不靠 name 匹配做 identity，而是用向量检索 top-5 候选 + LLM 决策 reuse/new

**对应 CLI stages**：`agent_extract` / `agent_inspect` / `agent_search` / `agent_evolve` / `agent_health` / `agent_govern` / `agent_check` / `migrate_tool_domain`

**数据**：写入 `knowledge-prda.db`
- `concept_nodes`: 718（已 reuse 收敛过的 concept）
- `concept_edges`: 1206（record↔concept 二部图）
- `tool_domain_nodes/edges`: 1070/1601（从 graph/ migrate 过来的 tool/domain 层）
- 还有 `graph_snapshots`、`graph_evolution_log`、`governance_proposals` 等 lifecycle 表

**流程**：
```python
for record in records:
    observations = LLM(record, prompt=concept_perceive_*)
    for obs in observations:
        candidates = top_5_similar_existing_concepts(embed(obs.description))
    decisions = LLM(record, observations, candidates, prompt=concept_decide_*)
    for d in decisions:
        if d.decision == "reuse": assign_record(...)
        elif d.decision == "new":  propose_concept(...) + assign_record(...)
```

**辅助 stage**：
- `agent_health` — 算 metrics + 写 snapshot + delta
- `agent_govern` — 找 merge / deprecate 候选 → LLM 二次确认 → 写 governance_proposals
- `agent_evolve` — 应用演化
- `agent_search` — Hybrid retriever (`GraphRetriever`)：embedding + tool_domain + concept 三路融合

**关键差异（相比 graph/）**：
- ✅ 解决了 entity identity（LLM 看语义而非 name 匹配）
- ✅ 有 governance lifecycle
- ✅ 有 hybrid 检索
- ❌ 仍是 record↔concept 二部图，**无 concept↔concept 关系**
- ❌ 每条 record 至少 2 次 LLM call，成本翻倍

### 路线 3：`vocab_maintenance/`（worktree-multidim-tag-system 分支） — 产品级主线

**状态**：**38 个 commit，活跃迭代中**。在 `.claude/worktrees/multidim-tag-system/` worktree，基线同 main `fa66622`。

**核心思想**：把 vocab（tag 定义）+ record-tag assignments 当 first-class state（`TagRecordNetwork`），用 agent loop 持续治理这个 network。

**模块结构**：
```
vocab_maintenance/
├── network.py          TagRecordNetwork（state object + 持久化）
├── bootstrap.py        冷启动：distill → synthesize → reverse_check
├── agent.py + graphs/  Agent loop as LangGraph StateGraph + checkpointer
├── measure.py          health metrics (hit_rate / coverage / co-occurrence / similarity)
├── diagnose.py         LLM 看 metrics 决定下一步 action
├── propose/
│   ├── merge.py        合并语义相近 tag（co-occurrence + LLM judge）
│   ├── deprecate.py    废弃低用 tag
│   ├── new.py          提议新 tag（覆盖未分配的 record）
│   └── additive.py     新 tag 反向找老 record（embedding filter + LLM 验证）
├── apply.py            执行 proposal + consistency invariants（TDD）
├── probes.py           diagnostic 探针
├── similarity.py       embedding 相似度
└── observability.py    structured run log + run_id
```

**Agent loop（LangGraph）**：
```
START → initial_measure → check_termination
                              ↓
                          diagnose → route_action
                              ↑           ↓
                              └────── apply → ⟨applied / rolled_back / blocked / error⟩
                                              ↓
                                            done → END
```

**Bootstrap（HITL）**：
```
START → distill(records→themes) → synthesize(themes→vocab)
                                       ↓
                                   vocab_review (HITL gate)
                                       ↓
                                   reverse_check → END
```

**工程成熟度**（项目里最高）：
- ✅ LangGraph StateGraph + checkpointer
- ✅ TDD apply_proposal + consistency invariants
- ✅ HITL gate（虽然 TECH_DEBT.md 提到端到端未完全验证）
- ✅ Eval suite + structured run log
- ✅ ThreadPool 并行（distill、reverse_check）
- ✅ 4 种 propose 路径完整
- ✅ TECH_DEBT.md 主动维护已知风险

**数据**：读 `outputs/knowledge.db` 的 `source_knowledge_records`，**自己另起 JSON 持久化**（vocab / assignments），不写回 knowledge.db

**关键差异**：
- 不是 KG，是**多维 tag 词表治理**
- TagRecordNetwork 仍是 bipartite（record ↔ tag），**无 tag↔tag 语义边**
- 没有 passage-level 检索增强

### 路线 4：HippoRAG2（拟）

**状态**：未实现，调研完成（详见 `doc/ref/hipporag2.md`）

**核心思想**：OpenIE 三元组 → 知识图 → Personalized PageRank 检索

**关键设计**：
- 节点：entity（具体 + 抽象）+ passage（每条 record 是节点）
- 边：三元组（OpenIE）+ synonym 边（cosine sim）+ entity-passage（containment）
- 索引：节点 embedding + passage-node 频次矩阵 P
- 查询：query→entity 抽取 → entity linking → PPR seed → score × P → top-K passage

**关键优势**：
- ✅ entity↔entity 关系（首个真有这能力的路线）
- ✅ synonym 边绕过 identity resolution 死结
- ✅ PPR 天然多跳
- ✅ 检索 token 极低（~1000/query）
- ✅ 索引增量友好

## 横向对照表

| 维度 | graph/ | agent_graph/ | vocab_maintenance/ | HippoRAG2 |
|------|--------|-------------|-------------------|-----------|
| commit 状态 | 未 commit | 未 commit | **38 commit，活跃** | 计划 |
| 解决问题 | record 打 entity 标签 | concept reuse + lifecycle | tag 词表 governance | 多跳关联检索 |
| 数据结构 | 3-type 二部图 | concept 二部图 + lifecycle | TagRecordNetwork (state) | OpenIE 密图 + passage |
| Identity 处理 | name 字符串匹配 | LLM reuse/new | propose_merge + invariants | synonym 边（不强制合并）|
| Entity↔entity 关系 | ❌ | ❌ | ❌（tag↔tag 只在 merge/cooccur）| ✅ |
| Governance | ❌ | ✅ partial | ✅✅ 4 propose + invariants + HITL | ❌（可借）|
| Agent loop | ❌ | partial | ✅ LangGraph StateGraph | ❌ |
| 检索能力 | 无 | Hybrid（concept+tool_domain）| 附带 | **PPR 是核心** |
| 数据库 | knowledge.db | knowledge-prda.db | JSON 文件 + 读 knowledge.db | 待定 |
| 工程成熟度 | PoC | spike | **产品级** | 待建 |

## 关键观察

### 观察 1：vocab_maintenance/ 是项目里唯一产品级别的工作

其他三个都是 PoC 或 spike。vocab_maintenance/ 有完整的工程化基础设施（StateGraph + checkpointer + TDD invariants + HITL + eval + observability + tech debt 文档）。这是**应该 commit 回 main 的工作**，但目前还留在 worktree。

### 观察 2：agent_graph/ 被 vocab_maintenance/ 完全超越

agent_graph/ 想做的几件事，vocab_maintenance/ 全部重做且更工程化：

| agent_graph/ 想做 | vocab_maintenance/ 实际 |
|-----------------|----------------------|
| PRDA pipeline (perceive→decide) | measure→diagnose→propose→apply |
| LLM reuse/new 决策 | propose_new + propose_merge + invariants |
| governance proposal | 4 种 propose + apply.check_invariants |
| graph_snapshots | observability.py structured log |
| graph_evolution_log | run_id + IterationRecord |
| Agent retriever (hybrid) | 附带（不是主目标）|

继续维护 agent_graph/ 是重复造轮子。

### 观察 3：vocab_maintenance/ ≠ KG

vocab_maintenance/ 解决的是 **tag 词表治理**，不是 KG：
- TagRecordNetwork 是 bipartite（record↔tag），没有 tag↔tag 语义边
- 没有 passage 级别的检索增强
- 没有多跳推理

如果最终目标是 GraphRAG，vocab_maintenance/ 不能直接给。

### 观察 4：HippoRAG2 跟 vocab_maintenance/ 不冲突也不重复

正交关系：

| 系统 | 管什么 |
|------|--------|
| vocab_maintenance/ | tag 词表的生死（哪些 tag 该存在、合并、废弃）|
| HippoRAG2 | record 怎么被检索（图上 PPR 召回 top-K passage）|

可以叠加：HippoRAG2 图里把 tag 当成"topic anchor"类 entity 节点，由 vocab_maintenance/ 维护其生死。

### 观察 5：graph/ 和 agent_graph/ 是项目演化遗留物

时间线推测：
1. **graph/** — 最早的 KG PoC（3-type 简单 schema）
2. **agent_graph/** — 升级版（PRDA + lifecycle），尝试解决 graph/ 的 identity 问题
3. **outputs/spike_extract_relations.py (v6_full)** — 又一次尝试（8-type ontology + 12 关系），离开 src/ 改在 outputs/ 下做 spike
4. **outputs/spike_normalize_entities.py** — v6_full 之后的 normalize 实验
5. **vocab_maintenance/**（另一个 worktree）— 不走 KG 路线，转向 tag 词表治理
6. **HippoRAG2（新）** — 调研后选定的下一步检索方案

graph/ 和 agent_graph/ **在工程严肃度上都已被 vocab_maintenance/ 超越**，但因为没正式 archive，还在工作树里污染。

## 建议路线

### 立刻执行（高优先级）

**1. 正式 archive graph/ 和 agent_graph/**
- 选项 a：移到独立分支 `archive/legacy-kg-experiments/`
- 选项 b：移到 `outputs/archive/` 目录（保留参考但不在 src/ 下污染）
- CLI 的 `graph_*`、`agent_*`、`migrate_tool_domain`、`seed_bootstrap` 这些 stage 在 cli.py 里同步移除或标 deprecated
- 防止继续误导未来工作

**2. vocab_maintenance/ 分支 merge 回 main**
- 38 commit 的成熟工作长期留 worktree 会让后续工作分裂
- merge 前确认 TECH_DEBT.md 里 HITL 端到端未验证是否需要先补
- merge 后 CLI 加 `vocab_*` 相关 stage（或者它自己有独立入口 `scripts/run_*`）

### 中期方向

**3. HippoRAG2 作为新独立模块**
- 路径：`src/consolidate_agent/retrieval/hipporag/`（新建）
- **不要塞进 vocab_maintenance/**——目标不同，schema 会乱
- 数据耦合：HippoRAG2 读 `knowledge.db` 的 `source_knowledge_records` + vocab_maintenance 维护的 tag，把 tag 作为 entity 节点的一类（"topic anchor"）

**4. HippoRAG2 Option A vs B 的选择**
- A（用现有 v6_full triples）13-18h，dry-run 验证
- B（重新设计 broad entity 抽取）16-22h
- **建议直接走 B**：既然要建新模块，多 3-4 小时换干净起点，避免 v6_full（outputs/ 下的 spike 产物）成为新的历史包袱

### 长期愿景

**5. vocab_maintenance/ + HippoRAG2 = 完整 GraphRAG**

```
vocab_maintenance/         → entity 词表生命周期（governance）
HippoRAG2                  → 图上检索（retrieval）
共享数据                    knowledge.db（697 source_knowledge_records）
agent_graph/ 当年想做的事    → 拆成上面两块分工完成
```

## 决策窗口

下一步如果要往前推，需要选：

1. **clean up 节奏**：先 archive 两个废模块 + merge vocab_maintenance/，还是先把 HippoRAG2 做出 PoC 再统一清理？
2. **HippoRAG2 落地节奏**：直接走 B 重抽 entity，还是先 A dry-run 看 PPR 跑得通再决定？
3. **vocab_maintenance/ 和 HippoRAG2 的耦合度**：完全独立 / 共享 tag 词表 / 共享 agent loop infra？

## 关联文档

- `doc/CURRENT_STATE.md` — 当前项目真实状态
- `doc/ref/hipporag2.md` — HippoRAG2 算法详解 + 项目应用判断
- `outputs/spike_normalize/q3_verification.md` — 跨域类比死路的实证
- `outputs/ontology_drafts/v0.2.md` — 历史 ontology 设计
- `.claude/worktrees/multidim-tag-system/TECH_DEBT.md` — vocab_maintenance 已知风险

## 调研方法

为防止未来重复调研，记录这次的方法：
1. `git worktree list` + `git log <branch>..<branch>` 找隐藏分支
2. 直接读每个模块的 `__init__.py` + 入口 pipeline 类 + types.py（5 分钟摸清核心抽象）
3. 查数据库表统计（`sqlite3 ... SELECT COUNT(*)`）验证模块是否真跑过
4. CLI `_STAGES` 反向找模块的实际暴露面
5. 对照 schema 差异（types.py 之间）判断模块是否互相重复

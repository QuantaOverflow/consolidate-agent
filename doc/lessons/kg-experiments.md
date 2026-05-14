# KG 抽取实验

> 状态：当前 baseline 不走 KG（2026-05-15）
> Legacy tag：`legacy/pre-cleanup`（含 graph/ + agent_graph/ + outputs/spike_normalize 等所有 KG 资产）
> 关联文档：`doc/research/pipeline_routes_evaluation.md`（详细评估）

## 总览

四代 KG 尝试，全部停止。

| 代 | 模块 / 资产 | 数据规模 | 关键差异 |
|---|------------|---------|---------|
| Gen-1 | `src/consolidate_agent/graph/` | graph_nodes=2296 / graph_edges=2912 | 3-type 二部图，name 匹配 identity |
| Gen-2 | `src/consolidate_agent/agent_graph/`（PRDA）| `knowledge-prda.db`：concept_nodes=718 | 二部图 + LLM reuse/new identity + lifecycle |
| Gen-3 | `outputs/spike_extract_relations.py`（v6_full）| triples_v6=12 关系 8 实体 type | 跳出 src/，重设计 ontology |
| Gen-4 | `outputs/spike_normalize_entities.py`（normalize spike）| 1130 abstract → 371 canonical | 试图通过 normalize 浮现 cross-domain 共享 |

## Gen-1 — `graph/` 二部图 PoC

3 类 entity（tool/concept/domain）、3 类边（mentions/illustrates/in_domain），全部是 record→entity 二部边，**无 entity↔entity**。identity 靠 name 字符串归一 + INSERT OR IGNORE。

**失败**：schema 表达力极弱、同义词全变独立节点、做不了多跳。

## Gen-2 — `agent_graph/` PRDA Pipeline

**P**erceive → **R**etrieve → **D**ecide → **A**pply。LLM 看语义而非 name 匹配做 identity（top-5 候选 + LLM 决策 reuse/new）。配套 governance lifecycle（merge/deprecate proposal）+ hybrid retriever。

**失败**：
- 仍然是 record↔concept 二部图，**无 concept↔concept 关系**
- 每条 record 至少 2 次 LLM call，成本翻倍
- 工程动机被 Gen-3 的 `vocab_maintenance/`（见 `tag-taxonomy-evolution.md`）完全超越

## Gen-3 — v6_full（8-type 12-relation Ontology）

跳出 src/，在 `outputs/spike_extract_relations.py` 重新设计 ontology：8 个实体 type、12 个关系。产物：`outputs/spike_normalize/triples_v6_normalized.jsonl`。

**进展**：第一次有 entity↔entity 真关系；写入 knowledge.db 的 graph_nodes/graph_edges。

**局限**：ontology v0.2 已知粒度太粗（见 `outputs/ontology_drafts/v0.2.md`），跨域类比仍不 work——驱动了 Gen-4。

## Gen-4 — Entity Normalize Spike

**假设**：抽取出的 abstract entity 名字不一样但语义相同（"data_contract_composition" / "schema_validated_output_contract"），归并后 cross-domain 共享应该浮现。

**做法**：cluster + LLM canonical naming，1130 abstract entity → 371 canonical（压缩 3x）。产物：`outputs/spike_normalize/entity_clusters.json` + `entity_mapping.json`。

**Q3 验证结果**（详见 `outputs/spike_normalize/q3_verification.md`）：
- 归并前 Codex vs LangGraph 共享 abstract entity：**0**
- 归并后共享：**1**（`output_contract_design`）
- 这 1 个共享，进一步看是 LLM 把不同 record 命名重合到同一 canonical name，**不是真语义共享**

**结论**：458-697 record 规模下，bottom-up KG 抽取浮现不出真 cross-domain 共享。**KG 抽象类比作为查询用例已死**。

## Lessons

1. **identity resolution 是 KG 的根本难题**。字符串匹配（Gen-1）、LLM 语义判断（Gen-2）、归并 normalize（Gen-4）都尝试过，都不够。
2. **record↔entity 二部图无法多跳推理**。entity↔entity 真关系是必需的，但难以高质量抽取。
3. **数据规模决定能力上限**。bottom-up 抽 KG 在 1000 量级 record 下浮现不了 cross-domain 共享。要么放弃这个用例（baseline 选择），要么换 top-down ontology 路线，要么扩到 10× 数据。
4. **不要把 retrieval 价值和分类价值混淆**。Gen-1~Gen-4 都在追"实体识别得对不对"，没回答"识别完了怎么用来检索"。HippoRAG2 走的是反方向（先想检索用 PPR，再倒推 KG 长什么样）—— 见 `hipporag2-evaluation.md`。
5. **spike 应该尽快下结论**。Gen-4 normalize spike 是好范例：6 小时跑通 + Q3 验证，明确证伪后立刻停。

## 留下什么资产

- `legacy/pre-cleanup` tag：graph/、agent_graph/、knowledge/audit/ 全部代码
- `outputs/spike_normalize/`：cluster json + mapping + Q3 报告（保留在 outputs，不在 baseline）
- `outputs/ontology_drafts/`：v0.1/v0.2 ontology 草稿 + 6 个 batch 抽取结果
- `outputs/knowledge.db.legacy-2026-05-15.bak`：graph_nodes(2296) + graph_edges(2912) 数据
- `knowledge-prda.db` 等历史 db 文件：PRDA 实验全部产物（Phase 4 之后这些不会动）

## 后续如果重启 KG 该怎么做

读 `hipporag2-evaluation.md` 而非 Gen-1~4。HippoRAG2 是评估完未实现的路线，比从 graph/ 续命起点干净得多。

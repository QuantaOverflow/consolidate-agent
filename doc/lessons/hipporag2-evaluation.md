# HippoRAG2 评估

> 状态：调研完成，未实现（2026-05-13 调研，2026-05-15 baseline 清理时搁置）
> 关联文档：`doc/ref/hipporag2.md`（算法详解 + 项目应用判断）、`doc/research/pipeline_routes_evaluation.md`

## 为什么调研这个

`kg-experiments.md` 里 Gen-1~4 KG 路线全部停止后，仍需要回答："record 多了之后怎么做语义检索 > 单纯向量召回？"。HippoRAG2（NeurIPS'24 / GraphRAG-Bench ICLR'26）是 2025-26 GraphRAG 方向的共识 SOTA。

## 核心思想

OpenIE 三元组 → 知识图 → Personalized PageRank（PPR）检索。

- **节点**：entity（具体 + 抽象）+ passage（每条 record 是节点）
- **边**：三元组（OpenIE 抽）+ synonym 边（cosine sim，**不强制合并**）+ entity-passage（containment）
- **索引**：节点 embedding + passage-node 频次矩阵 P
- **查询**：query → entity 抽取 → entity linking → PPR seed → score × P → top-K passage

## 关键优势（vs Gen-1~4）

| 维度 | Gen-1~4 KG | HippoRAG2 |
|------|-----------|-----------|
| Entity↔entity 关系 | 二部图为主，无 | ✅（OpenIE 三元组）|
| Identity resolution | 字符串/LLM 判断/归并 | ✅ synonym 边（不合并，PPR 自然连通）|
| 多跳推理 | 无 | ✅ PPR 天然支持 |
| 检索 token 成本 | N/A（没走到这步）| ~1000/query |
| 索引增量 | 难 | ✅ 加节点+边即可 |
| 数据规模需求 | 1000 量级浮现不出共享 | passage 节点本身就是 first-class，不依赖共享浮现 |

## 为什么暂搁置

1. **baseline 优先**：当前任务是让 main 干净跑通 ingestion，不混入 retrieval 实验
2. **应独立模块**：HippoRAG2 路径应是 `src/consolidate_agent/retrieval/hipporag/`（独立 worktree），不塞进 ingestion baseline
3. **A vs B 路线未拍**：
   - A：用现有 v6_full triples 直接套 PPR，13-18h 做 dry-run（包袱：v6_full 本身是 spike 产物）
   - B：重新设计 broad entity 抽取（不限抽象），16-22h 干净起点
   - 倾向 B（理由：见 `pipeline_routes_evaluation.md` 决策窗口）

## Lessons（提前埋的）

1. **KG 的检索价值 > 分类价值**。Gen-1~4 都在追"实体识别得对不对"，HippoRAG2 提醒：先想检索召回，再倒推 KG 形态。
2. **passage 仍是底，entity 是导航**。不要试图让 entity 替代 passage 做最终答案。
3. **Retrieval 独立于 ingestion**。不要把检索强行塞进 ingestion pipeline；ingestion 只负责生成高质量 passage（record），检索是消费端的事。
4. **synonym 边的设计哲学**：不合并、不强制 identity，让 PPR 用图结构自己投票。Gen-2 / Gen-4 强制 identity resolution 的路是错的。

## 后续如果要启动

- 起点：`git checkout -b spike/hipporag2-prototype baseline`
- 数据：从 `outputs/knowledge.db`（baseline 版只剩 source_knowledge_records + processed_sessions）读 record
- 决定 A 还是 B（如果 baseline 已经稳定，重抽 entity 不再有包袱，倾向 B）
- 不写回 baseline 的 knowledge.db；HippoRAG2 自有索引结构

## 参考

- `doc/ref/hipporag2.md` — 完整算法 + 应用判断
- `doc/research/pipeline_routes_evaluation.md` — 与 Gen-1~4 对照、决策窗口
- HippoRAG NeurIPS'24 论文 + GraphRAG-Bench ICLR'26

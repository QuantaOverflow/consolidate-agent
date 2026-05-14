# ADR-0002: 双粒度摘要策略——per-turn 用于检索，semantic chunk 用于提取

**日期**：2026-05-07  
**状态**：已采纳

## 背景

Knowledge pipeline 有两个使用 session 摘要的场景，需求截然不同：

1. **Evidence Agent（Sprint 6）**：给定一条 insight，在 session 中找到支撑它的具体 turn。要求能精确定位到 turn index，粒度必须是 per-turn。

2. **大 session 知识提取（Sprint 7）**：对 > 100k chars 的大 session 做知识提取，需要把 session 压缩到 LLM 上下文窗口内。要求保留跨 turn 的因果链和叙事连贯性。

## 决策

**per-turn 摘要和 semantic chunk 摘要共存，服务不同用途，不互相替代。**

- **per-turn 摘要**（已实现，Sprint 6.5）：为每个 turn 独立生成中文摘要，embed 存入 Chroma，供 Evidence Agent 的 RAG 检索使用。
- **semantic chunk 摘要**（Sprint 7 实现）：将语义连续的相邻 turns 合并为 chunk，对 chunk 生成摘要，拼接后送给知识提取 LLM。

## 理由

per-turn 摘要直接拼接不适合知识提取，原因：
- 单个 turn 可能只是「yes」、一条命令或短暂确认，独立摘要信息量极低
- 跨 turn 的因果链被切断（如「发现问题 → 分析原因 → 修复验证」分布在 3 个连续 turn），逐 turn 摘要会丢失这条推理链
- 知识提取 LLM 看到的是碎片化上下文，产出质量下降

反过来，semantic chunk 摘要不适合 Evidence Agent，原因：
- evidence 需要指向具体 turn index，chunk 合并后 turn 边界消失
- 检索粒度变粗，无法精确定位证据来源

## 后果

- Sprint 6（Evidence Agent）依赖 per-turn 的 `SessionTurnStore`，turn index 语义保持稳定
- Sprint 7 需要独立实现 semantic chunk 策略，不复用 `embed_session` 的 per-turn 逻辑
- 两套摘要可以共享同一个 LLM 调用基础设施（`summarizer` callable），但分开存储和调用

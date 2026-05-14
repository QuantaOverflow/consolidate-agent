# Pitfall Pipeline（首版蒸馏路径）

> 状态：已废弃（2026-04 删除）
> Legacy tag：`legacy/pitfall-pipeline` / 分支：`archive/pitfall-pipeline`
> 删除 commit：`306a0a7`

## 当时想解决什么

从 Codex/Claude session 里抽出"陷阱"（pitfall）—— 即作者踩过的错误、误解、低效路径 —— 蒸馏成 canonical 可复用条目。
假设：把 pitfall 单独抽出来，对未来工作的指导价值高于通用 insight。

## 做了什么

两阶段 pipeline：

```
session jsonl
  ↓ extraction          (LLM 提 raw pitfall record, prompt: canonicalization_*)
source_pitfall_records
  ↓ canonicalize        (LLM 聚合同义条目, prompt: taxonomy_*)
canonical_knowledge     (dedup 后的最终条目)
```

对应模块（已删）：
- `src/consolidate_agent/extraction/{chunk,extract,merge,normalize,pipeline}.py`
- `src/consolidate_agent/consolidation/{canonicalize,classify,pipeline}.py`
- 数据表：`source_pitfall_records` / `canonical_knowledge`
- prompts：`canonicalization_*`、`taxonomy_*`、`group_rewrite_*`

## 为什么不 work

1. **维度过窄**：很多有价值的 session 知识不是 pitfall（设计取舍、契约假设、工具用法），强行套 pitfall 框架反而丢信息。
2. **canonicalize 噪声大**：LLM 合并相似 pitfall 时频繁过度合并 / 漏合并，需要不停加约束 prompt，最终 prompt 越改越脆。
3. **下游消费方式没想清楚**：抽完 pitfall 之后怎么用？没有清晰检索场景驱动，逐渐变成"为抽而抽"。

## 换成了什么

直接做通用 knowledge record：`title / insight / applicability / scope`，不预设是不是 pitfall。
- 当前 baseline 走的就是这条
- 实测同样的 session，通用 record 覆盖面 >> pitfall（696 record vs 之前 ~200 pitfall）

## 留下什么资产

- `legacy/pitfall-pipeline` tag：保留所有删除前的代码
- `archive/pitfall-pipeline` 分支：同上，便于 checkout
- `outputs/knowledge.db` 里的 `source_pitfall_records` 和 `canonical_knowledge` 表已清空（Phase 4 migration 中 DROP）

## Lesson

抽取目标的"边界"比"质量"先决定一切。先想清下游怎么用，再决定抽什么粒度。

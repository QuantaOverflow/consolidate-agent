# ADR-0001: 过滤纯问答 Session，不提取知识

**状态**: 已采纳  
**日期**: 2026-05-04

## 背景

Knowledge extraction pipeline 从 Codex session 中提取可复用的工程洞察。在验证 golden set 时发现，纯问答 session（用户提问、LLM 回答，没有任何工具调用）会产出 git 基础知识等低质量记录，例如"git push uploads missing commits"、"bookmarks and pages metaphor models Git branches"，与提取目标不符。

## 决策

在 preprocess 阶段增加结构性过滤：**session XML 中既无 `<bash>` 又无 `<file_edit>` 的 session 直接跳过，不进行 LLM 提取**。

```python
has_action = "<bash>" in session.xml or "<file_edit>" in session.xml
if not has_action:
    skip
```

## 理由

我们定义的 knowledge 类型是**通过实际执行验证过的洞察**——来自失败、调试、设计迭代的产物。纯问答 session 的内容是 LLM 对已有知识的复述，没有经过任何实际操作验证，不符合这个定义。

相比 prompt 约束，结构性过滤更可靠：
- 确定性，不依赖 LLM 自律
- 计算成本为零（字符串检查）
- 在 preprocess 阶段拦截，节省 LLM API 调用

## 后果

- 纯问答 session 被跳过，不计入 processed，不消耗 API
- 该过滤在统计中记为 `skip_reasons["no_action"]`
- 概念解释、教学内容等问答型知识被放弃——**这是有意为之**
- 如未来需要提取问答型知识，应建立独立的 pipeline，使用不同的提取标准和知识类型，而非复用当前 pipeline

## 替代方案（未采纳）

**用 prompt 约束过滤**：要求 LLM 只提取有 trial-and-error 证据的知识。实测无效——LLM 看到问答内容仍会提取，因为无法从文字对话中判断"没有失败"等于"不符合标准"。

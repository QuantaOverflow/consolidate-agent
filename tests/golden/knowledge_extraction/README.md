# Knowledge Extraction Golden Set

5 个代表性 session，用于验证 prompt 改动对提取质量的影响。

## Session 选取理由

| session | 文件大小 | 类型 | 当前质量 | 预期新 prompt 后 |
|---|---|---|---|---|
| 019a1c15 | 40KB | git 诊断，短 session | 差（全是 git 基础知识） | 0 条 |
| 019a33d4 | 78KB | git push，简单任务 | 差（基础知识） | 0-1 条 |
| 019d017b | 4.2MB | streaming 分布式开发 | 中等（有真实踩坑） | 2-4 条 |
| 019d6be3 | 1.4MB | state machine 开发 | 好（高 evidence） | 2-4 条 |
| 019d80f4 | 1.1MB | adaptive query 系统 | 中等（质量不一） | 2-3 条 |

## 验收标准

**判断一条知识是否应该被提取：**
- KEEP：有明确的"踩坑 → 发现 → 修复"链条，或非显而易见的设计决策
- REJECT：任何开发者查文档即可知道的事实，或 evidence=1 且无跨 turn 验证
- BORDERLINE：需要结合 session 原文判断

**新 prompt 通过条件：**
1. 019a1c15 和 019a33d4 产出减少到 0-1 条
2. 019d017b 保留 streaming transport、fallback 可观测性、test scope mismatch
3. 019d6be3 大部分保留（质量本来就好）
4. 019d80f4 过滤掉 evidence=1 的两条

## 验收方式

```bash
# 对 golden session 重新提取（需要先清除这些 session 的 processed index）
uv run python -m consolidate_agent --extract-knowledge --input-dir tests/golden/knowledge_extraction/sessions

# 对比新旧产出，逐条对照 expected/*.yaml 的标注判断
```

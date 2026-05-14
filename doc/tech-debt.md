# Tech Debt

## TD-001 Evidence Agent 语言 gap 导致中文证据漏召

**发现于**：Sprint 6 golden set 验收（2026-05-07）

**现象**：
insight 是英文（如 "Interrupt-driven recovery must distinguish between transient and persistent"），但对应的证据 turn 内容是中文（如 "这个中断来自'无进归即疑似风控'的启发式，不是实际检测到验证码"）。Plan 阶段 LLM 生成的搜索 query 以英文为主，导致 `search_text` 和 `search_turns` 均未命中该 turn，产生误 reject。

**影响**：golden set 68 条中 1 条（019d2947 Interrupt-driven recovery）误 reject，Hit@Evidence=92.6%，仍满足验收标准 ≥ 90%。

**根因**：
Plan prompt 没有要求 LLM 为英文 insight 生成对应的中文搜索词。`search_turns` 的 translator 只在 `SessionTurnStore.search_turns` 层做了 query 翻译，但 Evidence Agent 的 Plan 阶段是直接生成 query 文本，未经翻译。

**尝试过的修复**：
在 plan prompt 里加入"若 insight 是英文，同时生成中文搜索词"的指示。效果为净负：修复了 1 条误 reject，但引入 2 条回归 + 3 条 validation error，整体变差，已回退。

**待做**：
- 在 Plan 阶段单独加一步 query 翻译（类似 Strategy E 的 translator），将 English insight 翻译成中文后加入搜索计划
- 或在 `_execute()` 里对 semantic search 自动走 translator

---

## TD-002 Plan LLM structured output 偶发返回 None

**发现于**：Sprint 6 golden set 验收（2026-05-07）

**现象**：
`EvidenceAgent._plan()` 调用 `with_structured_output(EvidencePlan)` 时，对特定 insight（尤其包含 `{`, `}`, `'` 等特殊字符的 insight）偶发性返回 None，导致 verify() fallback 到原始 record。

**影响**：非确定性，同一 insight 有时成功有时失败。在 68 条 golden set 中偶发 0-6 次（取决于 API 状态）。

**根因**：
`with_structured_output` 使用 function calling 模式。当 qwen-plus 模型对含特殊字符的 input 输出普通文本而非 tool call 时，LangChain 解析失败返回 None。属于模型非确定性行为，不是稳定 bug。

**当前缓解**：
`_plan()` 加了 None 检查，fallback 为 `EvidencePlan(searches=[SearchAction(type="semantic", query=insight)])`，保证不会整条失败，但 Plan 质量会下降。Evidence Agent 已用 `include_raw=True` 包装 structured output，失败 trace 会记录 raw response / parsing error；如果 Plan 失败只是因为 `searches` 超过 4 个，会记录 recovered trace 并截断到前 4 个继续执行。

**待做**：
- 或改用 `method="json_schema"` 模式，对特殊字符更鲁棒

---

## TD-003 ProcessedIndex 状态存储为 JSON 文件

**发现于**：知识提取管道设计讨论（2026-05-07）

**现象**：
`knowledge-processed-index.json` 用 JSON 文件存储 session 处理状态，不支持并发写、无事务保护、进程崩溃可能导致文件损坏。

**影响**：单机单进程的离线 batch pipeline 场景下不影响正确性，但不适合作为生产级持久化方案。

**待做**：
将 processed index 迁移到 `knowledge.db` 的专用表（`knowledge_extraction_state`），利用 SQLite 的原子写入和事务保护。

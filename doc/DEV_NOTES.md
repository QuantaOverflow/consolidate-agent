# 开发笔记

## LLM 集成问题

### Knowledge Extractor 不能无证据拔高

100-session 测试中出现过两类 Evidence Agent reject：session 只证明了 `list/detail/update` CLI 命令存在，但 extractor 把它拔高成降低认知负担/支持组合性；session 只证明了集成测试存在隐式环境 fallback，但 extractor 写成了应该 fail fast 的规范性结论。Extractor prompt 因此加入 evidence boundary few-shot：可以抽象，但不能添加 session 没有直接讨论或证明的动机、收益、设计原则或 best-practice framing。

### Evidence Agent 不再使用 ReAct tool loop

**现象**：`create_react_agent` + `response_format` 在 qwen-plus 上会出现 `result["structured_response"]` 返回 None；ReAct 工具循环还会引入 recursion limit 风险。

**原因**：qwen-plus 支持 tool call，也支持单独的 structured output，但不适合依赖"工具循环结束后再触发 structured output"这个两阶段 agent 模式。

**修法**：Evidence Agent 改为 bounded Plan-Execute-Judge：
1. Plan：`with_structured_output(EvidencePlan, include_raw=True)` 生成 1-4 个搜索动作
2. Execute：代码确定性调用 `SessionTurnStore.search_text` / `search_turns` / `get_turn`
3. Judge：`with_structured_output(JudgmentOutput, include_raw=True)` 判断 admit / reject / need_more
4. Replan 最多一次；Final Judge 只能 admit / reject

这样 LLM 调用上限固定为 3 次，没有 LangGraph recursion limit。只有 `verdict=admit` 且 `evidence_turns` 非空才真正 admit；empty-evidence admit 会作为 `empty_admit_evidence` soft reject 记录。批量验证按 session worker 并发，同一 session 内复用解析后的 turn map 和 text-search cache；semantic search 只读取 `--embed-turns` 预先构建的 Chroma turn index，不在验证阶段重建 embedding。`search_turns` 使用 Chroma raw distance 并在单次 query 结果内归一化到 `0..1`，避免 LangChain relevance warning 打印原始 turn 内容。

当 Plan 或 Judge 的 structured output 返回 None、解析失败或抛异常时，Evidence Agent 会写一条 JSONL 到 `outputs/evidence-agent-failures.jsonl`。structured output 调用包了一层 `include_raw=True`，trace 会尽量记录 raw response 的长度、hash、截断预览、`parsing_error`、`tool_calls` 和 `invalid_tool_calls`。这能区分真正 refusal、schema validation error，以及 Qwen/DashScope 偶发生成 malformed tool-call arguments 的情况。Judge trace 还会包含 record/session、insight、search actions、search_results 的长度、hash 和截断预览，用于复盘"为什么这次 structured output 没解析出来"。当 Judge 正常返回 `reject` 时，也会写 `stage=evidence_reject` trace，记录 `judge_reasoning`、`judge_verdict`、search actions 和当时的 search_results hash/截断预览，便于解释 `evidence_count=0`。Judge prompt 要求 `reasoning` 只写短自然语言摘要，禁止粘贴原始 JSON、代码、正则、shell 或反斜杠密集片段，避免自由文本字段破坏 tool-call JSON。该文件可能包含 session 原文片段，只作为本地调试 artifact。

Plan structured output 最多尝试 3 次。第一次失败后，retry prompt 会带上短错误反馈；如果 raw 里有 `invalid_tool_calls`，会带一个截断的 bad args/error 摘要，提示模型修正 JSON（例如把 malformed `"query:` 改回 `"query"`）。如果 retry 成功，会记录 `recovered=true`、`recovery_method=retry_with_plan_error_feedback`。如果 Plan 解析后的 `searches` 只是超过 schema 上限 4 个，Evidence Agent 会记录一条 `recovered=true` trace，并截取前 4 个 search action 继续执行，这类恢复不消耗 retry。3 次后仍失败才 fallback 到 `semantic(insight)`。

Judge structured output 也最多尝试 3 次，但只重试格式/解析失败，不重试正常的业务 `reject`。retry prompt 会带上短错误反馈和 `invalid_tool_calls` 摘要，并要求重新发出合法 `JudgmentOutput`，同时不要在 `reasoning` 中复制原始 JSON、代码、正则或反斜杠密集文本。如果 retry 成功，会记录 `recovered=true`、`recovery_method=retry_with_judge_error_feedback`。3 次后仍失败时保留 soft reject：当前 record 的 `evidence_turns=[]`、`evidence_count=0`。即使坏 JSON 里看起来包含 `verdict=admit` 和 `evidence_turns`，也不做手工 salvage，避免误放行。

可以用 `uv run python scripts/smoke_evidence_judge_retry.py` 人工构造 Judge malformed tool-call 输出。该 smoke 覆盖两条路径：第一次 Judge 输出非法 JSON、第二次 retry 恢复为 admit；以及连续 3 次非法 JSON 后 soft reject。恢复 trace 会保留导致 retry 的上一轮 `invalid_tool_calls` 摘要和 search results 预览，方便确认是格式问题而非证据不足。

100-session admitted 抽样发现，部分 `evidence_count=1` 记录是 explanation-only false admit：用户问“为什么/是什么意思/解释一下”，assistant 在单个 turn 里给出概念解释，Judge 因文本直接说中 insight 而 admit。Judge prompt 因此加入事件级证据门槛：admit 必须有工具执行、测试/API 调用、文件修改、diff、日志、错误输出、验证结果、用户明确报告真实失败，或 assistant 对刚发生的执行/失败/修复做总结；纯解释、建议、概念说明和最佳实践讨论即使文本匹配也必须 reject 或 need_more。Prompt 同时加入 3 个负例（接口契约、Jupyter outputs、MCP Playwright cookie）和 2 个正例（OpenRouter key 401、update 后 GET 字段缺失）。

---

### `with_structured_output` 偶尔返回 None

**现象**：没有 agent loop，单纯的 structured output 调用有时也返回 None。

**原因**：模型偶发性不遵守 structured output 指令，通常与 prompt 中有空内容或模糊指令有关。

**修法**：加最多 3 次 retry，比 2 次更稳。对 taxonomy governance 这类提案治理节点，3 次仍失败时保守降级为 `reject`，避免单个 tag proposal 中断整条 consolidation pipeline；同时应继续排查 prompt 或 provider 返回的 raw tool call。

---

## SQLite 并发问题

### 单个连接多线程并发写冲突

**现象**：`sqlite3.InterfaceError: bad parameter or other API misuse` 或 `sqlite3.DatabaseError: another row available`。

**原因**：`check_same_thread=False` 只允许多线程共享连接，但不保护并发写操作。`LLMAgentTagGovernor` 用 `ThreadPoolExecutor` 并行处理 proposals，每个线程都会触发 `embed_canonicals` → `save_canonical_embedding` → DB 写入。

**修法**：在 `KnowledgeStore` 加 `threading.Lock`，对所有 embedding 读写方法加锁：
```python
self._lock = threading.Lock()

def save_canonical_embedding(self, ...):
    with self._lock:
        ...
```

### `ALTER TABLE ADD COLUMN` 并发竞争

**现象**：`sqlite3.DatabaseError: another row available`（比上面更早出现）。

**原因**：`_ensure_column` 先检查列不存在，再执行 `ALTER TABLE`，多线程同时通过检查后都尝试 ALTER，第二个会报错。

**修法**：用 `try/except` 忽略重复 ADD COLUMN 的错误：
```python
try:
    self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
except Exception:
    pass  # concurrent threads may race to add the same column
```

---

## 调试方法

### mock 掩盖了真实问题

单元测试全 mock 时，SQLite 并发、LLM 兼容性、空消息等问题完全不会出现。这类问题只有集成测试才能发现。规则：**凡是涉及外部服务（LLM、DB）的模块，单元测试通过后必须跑集成测试**。

### retry 不是万能的

连加两次 retry 问题还在，说明不是偶发而是系统性。遇到 retry N 次还失败，应该先搞清楚为什么失败，而不是继续加次数。

### 直接观察 LLM 输出

遇到 LLM 行为异常，写一个小脚本直接打印消息历史，比猜测有效得多：
```python
result = agent.invoke({"messages": [HumanMessage(content=...)]})
for m in result["messages"]:
    print(f"[{type(m).__name__}] {m.content[:200]}")
```
空 AIMessage 的问题就是这样发现的。

### 每次只改一处

堆修复会让后续无法判断哪个改动有效，尤其是有 retry 的场景。

---

## 架构问题

### `run_consolidation` 的 `has_api_key` 分支覆盖外部传入参数

`run_consolidation` 的函数签名接受 `taxonomy_governor` 等参数，但 `has_api_key=True` 时会完全忽略这些参数，强制使用 `LLMAgentTagGovernor`。

集成测试时想换 governor 必须绕过 `run_consolidation`，直接实例化 `ConsolidationService`。

---

## 热启动设计

### taxonomy_draft 不应无条件触发

冷启动（无 active_tags）才需要完整跑 `taxonomy_draft → governance → rule_classification`。热启动时先跑 `rule_classification`，再用 `LLMTagCoverageChecker` 判断哪些 canonical 真正无法归类，只对这部分跑 `taxonomy_draft`。

### governance 决策的 supporting_canonical_ids 要用于分配（已实现）

当 governance 决定 accept 或 merge 一个 proposal 时，该 proposal 的 `supporting_canonical_ids` 里的 canonical 应该直接分配到 resolved tag，不能依赖后续的 `rule_classification` 来找到匹配——因为新建或 merge 后的 tag 名字可能和 canonical 文本没有 token 重叠。

当前实现已经在 `taxonomy_governance` 后生成 `governance_assignments`，并在 uncovered classification 阶段合并这些直接分配。

---

## 未来方向

### 问答型知识的独立 pipeline（待开发）

100-session 抽样 review 发现，Evidence Agent 的严格事件级证据门槛会拦住一类有价值但来源不同的知识：用户在 session 中真实遭遇了某个问题，通过问答理解了背后机制，但 turn 里没有 bash/API/测试等事件证据。这类知识代表个人认知成长过程，和"执行验证型知识"目标不同：

| | 执行验证型 | 问答理解型 |
|---|---|---|
| 来源 | 执行/失败/修复/验证事件 | 真实遭遇触发的问答解释 |
| 价值 | 注入未来 session 防止重蹈 | 记录个人认知成长轨迹 |
| 注入 Codex 有效吗 | 有效 | 基本无效（LLM 已知） |
| 消费方式 | 动态检索注入上下文 | review report、知识图谱 |

**结论：不做分层，分开存储。**

现有 pipeline 保持严格 Evidence Agent 标准，只产出执行验证型知识。后续另开一条轻量 pipeline 专门处理问答型：直接提取，不过 Evidence Agent，加 `source=explanatory` 标记，进不同的表。核心原则：**不是这个知识类型不值得提取，而是两类知识混在同一条可信度语义里会互相干扰**。

### 知识类型扩展

当前系统有两条平行提取链路：

- pitfall extraction：提取"坑"（pitfall），category 限定为 `execution_strategy` 和 `tooling_environment`
- generic knowledge extraction：对完整 engineered session XML 做一次 structured LLM call，提取可迁移的 `KnowledgeRecord`

generic knowledge extraction 已经覆盖一部分原先未捕获的知识：

- **有效模式**：某个做法在多个 session 里都奏效
- **架构决策**：为什么选择某个方案而不是另一个
- **领域理解**：对某个库/框架/API 的深入认知
- **调试技巧**：定位某类问题的有效手段

这条链路目前完成了“提取 + admission + SQLite 持久化”，并已有手动
embedding 索引与 pitfall → knowledge 检索能力；但还没有
canonicalization、tagging 或注入未来 Codex session 的消费路径。

### Session context engineering

`SessionContextEngineer` 是 pitfall 之外通用知识抽取的上游基础设施：

- 把 Codex JSONL 转换为 compact XML
- 折叠文件读取命令，只保留路径
- 截断长 assistant 消息
- 保留命令结果、patch、web search、rollback、context compaction、sub-agent 摘要
- 按 user message 分组为 `<turn index="...">`

这层目前有单元测试和验证脚本，但缺少一份正式 XML contract 文档。后续如果 retrieval、review report 或其他 agent 都消费这份 XML，应先稳定 contract。

### Generic knowledge extraction 的消费缺口

`source_knowledge_records` 仍然是 generic knowledge 的终点表。现在可以
通过 `--embed` 写入 embedding，并用 `find_related_knowledge` 从一个已
embedding 的 canonical pitfall 找相关 knowledge。后续仍需要决定：

- 是否把 generic knowledge 送入现有 canonical/tag graph
- 是否把现有手动向量索引接入自动检索/注入路径
- 是否导出为 Markdown/Obsidian 供人工 review
- 是否在新 Codex session 启动时按 cwd/技术栈动态注入

当前不建议直接混入 pitfall canonical rules，因为 `KnowledgeRecord` 的语义是 transferable cognition，不一定是 preventive rule。


### 知识图谱可视化

现有数据已有图结构：

```
session → pitfall_candidate → canonical_knowledge → mechanism_tag
                                     ↕（同 tag 共现）
                              mechanism_tag → mechanism_tag（merge 关系）
```

可能的实现路径：
- **最轻量**：导出为 Obsidian Markdown + wikilink，用 Graph View 直接可视化
- **中等**：用 `networkx` + `pyvis` 生成交互式 HTML
- **完整**：接 Neo4j

建议在知识类型扩展之后再做图谱——节点间关系更丰富时才有意义。现在 128 条 canonical rule 之间只有"同属一个 tag"的隐式关系，图谱价值有限。

### 知识消费场景（待定）

- 自动写入 `~/.codex/rules/`，每次 Codex 启动时加载
- 按当前 session 的 cwd/技术栈动态检索相关 rule 注入上下文
- Session 结束后生成 review report（本次触发了哪些已知 pitfall）

---

### 交互式图谱应用（设想，待细化）

#### 动机

Obsidian 图谱作为消费端的核心缺陷：
- **静态快照**：每次需要手动重新导出，无法随 session 增长自动更新
- **图结构浅**：只有 record→tag 的星形结构，record 之间无直接边
- **无语义检索**：Obsidian 只有全文搜索，向量检索（`--query`）完全割裂
- **溯源断链**：有 `session_id` + `evidence_turns` 但点不进去，须手查 JSONL
- **被动消费**：需要主动打开浏览，无触发机制
- **无反馈回路**：在 Obsidian 里的编辑会被下次导出覆盖

#### 设想：Interactive Graph Web App

以图谱为核心 UI 的个人知识库，三层结构：

```
数据层    SQLite（已有）
          ↑ 持续写入
Pipeline  Background Agent（session watcher + extraction）
          ↓ 读取
交互层    Interactive Graph Web App
```

**图谱交互**
- 搜索一个关键词或 query → 高亮/缩放到匹配 records 的局部子图
- 点击 record 节点 → 右侧面板展示：insight、applicability、tag、以及对应 session 的 evidence turns 原文（数据已在 `processed_sessions.xml`）
- 非焦点节点灰化，保留结构感知

**双路召回**
- 向量路：语义相似的 records
- 图扩展路：同 tag 的其他 records
- 两路结果在图上同时高亮，可视化区分（不同颜色或边权重）

**节点类型**

`record ── tag` 二分图已足够。溯源（record → session 原文）是数据查询而非图遍历，点击 record 时直接读 `processed_sessions.xml` 对应 turns 即可，不需要 session 作为图节点。

#### Background Agent

目标：常驻后台，随 Codex session 增长动态更新图谱。

数据层已就绪（processed-index 支持增量），缺的只是触发机制：
- 监听 `~/.codex/sessions/` 目录的新增 JSONL 文件（file watcher）
- 触发 extraction pipeline → 写入 SQLite → 图谱自动刷新

改动范围小，主要是在现有 pipeline 外面加一层 watcher loop。

#### 技术选型（待定）

- 图渲染：`pyvis`（最轻量，基于 vis.js）、`sigma.js`、`d3.js`（交互最强但前端工作量大）
- 后端 API：FastAPI 或直接 SQLite 只读查询
- 局部子图查询：纯 SQL（`knowledge_tag_assignments` 已有所需关系）

---

## Layer 3 Tag Taxonomy 重构设计（进行中）

### 现状问题

**1. 自底向上生成，缺全局视角**
Drafter 每次只看一个 batch，看不到全语料分布。冷启动时 active_tags 为空导致几乎全部 accept，早期低质量 tag 奠定了不良基调。

**2. Governor 缺粒度宪法**
没有被告知"什么是好 tag"，接受了 `akshare`（工具名）、`error_handling`（过宽）、`tun_mode`（一次性）等低质量 tag。

**3. 无动态维护能力**
tag 一旦 accept 就没有回头审查机制，存量质量随时间持续下降。

---

### 已实现设计（当前状态）

**TagMaintainer** 三种模式：
- `cold_start`：UMAP+HDBSCAN 聚类，per-cluster structured LLM 提案
- `warm_start`：bounded 四步（embedding similarity → 贪心聚类 → per-cluster structured LLM）
- `audit`：heuristic（count/overlap 规则）+ ReAct agent，hybrid 合并去重

**TagGovernor**：按 proposal.action 路由（CREATE/MERGE/DEPRECATE 不同工具集），ReAct agent per proposal，并发最多 10 个。

**TagAssigner**：structured LLM per record，从 active tags 里分配，动态 Literal 约束。

**执行层**：`_taxonomy_governance` 实现 Plan→Validate→Commit，处理 split/merge 原子性约束。

---

### Audit 优化路径（待实现）

#### 现状问题

单个 ReAct agent 探索全量 tag，step 预算有限（约 25 步），探索路径随机，结果不稳定。Governor 作为 full ReAct agent per proposal，在 audit 输出高质量时是过度设计。

#### Stage 1：Detection 层 DAG 化（代码，确定性）

把"发现哪些 tag 有问题"从 agent 里剥离，改为代码确定性执行：

- **命名违规**：正则 + 工具名/平台名黑名单 → `naming_candidates`
- **覆盖异常**：`count > 30` → `oversized_candidates`，`count < 3` → `undersized_candidates`
- **语义重叠**：tag representative embedding（assigned records 均值向量）pairwise cosine > 阈值 → `overlap_candidates`（pairs）
- **概念混淆**：暂时靠 oversized 阈值触发，无法靠代码直接检测

#### Stage 2：Remediation 层 per-candidate 聚焦

区分标准：**输出结构是否确定**（不是是否需要搜索——所有类型都需要搜索）。

| 候选类型 | 代码层预备上下文 | LLM 处理 | 输出结构 |
|---|---|---|---|
| naming | 预取 tag records + active tags 列表 | structured LLM per tag | 1 个替代名，确定 |
| overlap | 预取双方 records（sampling） | structured LLM per pair | merge A→B / keep，确定 |
| undersized | embedding 搜 top-N 候选 + 预取候选 tag records | structured LLM per tag | merge into X / deprecate，确定 |
| oversized | 无（agent 自己用工具读） | **agent** per tag | N 个子 tag，N 未知 |

只有 oversized split 需要 agent，原因是 N 未知——需要迭代探索 records 识别子模式，其他三类输出结构在开始前就确定了。

#### Stage 3：输出数据模型统一

用 `AuditOutput` 替代扁平 `list[TagChangeProposal]`，`SplitProposal` 原子绑定 deprecate + creates：

```python
class SplitProposal(BaseModel):
    source_tag_name: str
    creates: list[TagChangeProposal]  # rename = creates 只有 1 个
    reasoning: str

class AuditOutput(BaseModel):
    splits: list[SplitProposal]    # 含 rename（N=1）和 split（N≥2）
    merges: list[TagChangeProposal]
    deprecates: list[TagChangeProposal]
```

rename 是 split 的特例（`creates` 只有 1 个）。`requires_any_of` 字段可随之删除，原子性由结构保证，`_taxonomy_governance` 处理 `SplitProposal` 时不需要推断配对关系。

#### Stage 4：Governor 轻量化

DAG 各步骤已预验证（命名合规、overlap 已核实），Governor 退化为 **structured LLM classifier**，不需要工具：

- 只对照 Constitution 判断：名字合规？覆盖合理？不重复已有 tag？
- 单次 LLM 调用，并发处理所有提案
- 速度从"10 个并发 ReAct agent × 30s"变成"N 个并发 structured LLM × 3s"

#### Stage 5：审计回路

每次 audit 结束记录：候选数量、Governor 通过率、taxonomy 指标变化（平均覆盖数、tag 总数、no_tag 率）。让 audit 质量可量化、可迭代。

#### 当前 vs 最终形态

| | 现在 | 最终 |
|---|---|---|
| 候选检测 | 单 agent 探索（随机） | 代码 DAG（确定性） |
| Naming 修复 | agent 提案 | 代码预取 records → structured LLM |
| Overlap merge | agent 猜测 | 代码预取双方 records → structured LLM |
| Undersized | heuristic + agent | embedding 搜候选 → structured LLM |
| Split | agent（配对不明确） | agent per tag + SplitProposal |
| Governor | ReAct agent × N | structured LLM classifier × N |
| 原子性 | Plan→Validate→Commit 推断 | AuditOutput 结构保证 |

**LLMKnowledgeTagDeduplicator → 已删除**
职责已被 Maintainer（warm_start 时识别重叠）和 Governor（check_tag_overlap 工具）完全覆盖。

---

### 统一提案类型

```python
class TagChangeProposal(BaseModel):
    action: Literal["create", "merge", "deprecate"]
    # create：新建 tag
    name: str | None
    definition: str | None
    supporting_record_ids: list[str]
    # merge：将 source 合并入 target
    source_tag_name: str | None
    target_tag_name: str | None
    # deprecate：废弃某个 tag
    deprecated_tag_name: str | None
    reasoning: str
```

split = deprecate 旧 tag + create 若干新 tag，不需要单独 action 类型。

---

### Tag Constitution（注入 Maintainer 和 Governor 的 prompt）

```
一个 tag 应描述"一类反复出现的技术决策场景"：
- 覆盖 5-20 条 records 为健康粒度（<3 条应 merge 或 deprecate，>30 条应 split）
- 不应是工具名（akshare、playwright、consul）
- 不应是宽泛领域名（error_handling、persistence、database）
- 不应是一次性场景（tun_mode、waf_bypass）
- 应该能回答："开发者在什么情况下会重复遇到这类问题？"
```

---

### 工作流

**冷启动**（无 active tags）：
```
TagMaintainer（cold_start）
    → cluster_records() 工具获取所有 cluster
    → 批量分析，一次性提出所有 tag 提案
    ↓ TagGovernor（per 提案，accept/merge/reject）
    ↓ TagAssigner（per record）
    ↓ Coverage Check → 有未覆盖（含 noise records）→ Maintainer（warm_start）
    ↓ Persist
```

**热启动**（有 active tags，有新 records）：
```
TagAssigner：新 records + 历史 no_tag records 全部重跑
    ↓ Coverage Check：谁还是 uncovered？
    ↓ TagMaintainer（warm_start）：分析 uncovered pool
        ├── 能归入已有 tag → merge 提案
        └── 形成新模式 → create 提案
    ↓ TagGovernor
    ↓ TagAssigner 再跑一遍
    ↓ Persist
```

历史 no_tag records 先走 Assigner 是关键：新建的 tag 可能让之前无法归类的 records 现在能被分配，避免重复提案。

**动态审计**（`--stage audit-tags` 或定期触发）：
```
TagMaintainer（audit）
    → 用工具扫描 taxonomy 健康状态
    → 提出 merge/deprecate/split（= deprecate + create）提案
    ↓ TagGovernor
    ↓ 执行变更 + 受影响 records 重新分配
    ↓ Persist
```

---

### 聚类参数（已验证）

```python
umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1, metric="cosine")
hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric="euclidean")
# 结果：458 条 records → 51 clusters + 90 noise（19.65%）
# 过滤条件：unique_sessions > 1（单 session cluster 不提案）
```

抽检质量：cluster=44（LangGraph routing，coverage=1.0），cluster=48（LangGraph state，coverage=1.0），cluster=8（Git ops，coverage=1.0）。

---

### 实施说明

- 使用新 DB 文件（`outputs/knowledge-v2.db`）重新冷启动，不在现有 66 个 tag 上修补
- `scripts/spike_knowledge_cluster.py` 中的聚类逻辑提升为生产代码，作为 `cluster_records()` 工具的内部实现
- `KnowledgeConsolidationService` 主要改动：替换 Drafter → Maintainer，删除 Deduplicator，更新 LangGraph 路由
- 新增 `--stage audit-tags` CLI 命令触发 audit 模式

---

### 实测发现（2026-05-10）

#### DashScope embedding 相似度范围

DashScope text-embedding-v3 在知识 record 语料上，同类 records 之间的 cosine similarity 普遍在 **0.4–0.6**，不是直觉上的 0.7+。由此标定两个阈值：

- `reassign_hint` 阈值：0.65（低于此值的 record 不强制归入已有 tag）
- warm_start 贪心聚类阈值：0.55（低于此值的 record 对不归入同一 cluster）

如果未来更换 embedding 模型，需要重新在真实语料上测量同类相似度分布后再标定这两个阈值，不应沿用当前值。

#### Governor `get_tag_coverage_stats` 对 CREATE 提案无效

Constitution 的覆盖规则（"少于 3 条应 merge/deprecate"）对已有 tag 和新建提案语义不同：

- 已有 tag：查 DB 的实际分配数，`get_tag_coverage_stats` 有效
- CREATE 提案：tag 尚不存在，DB 返回 0，Governor 误判为"覆盖不足 → reject"

**修法**：Governor 按 `proposal.action` 路由到不同工具集和 prompt。CREATE 提案只给 `search_knowledge_records` + `check_tag_overlap`，覆盖数从 `supporting_record_ids` 读取；MERGE/DEPRECATE 才给 `get_tag_coverage_stats`。

#### Governor 并发 accept 可能产出语义重叠的 tag

`govern()` 用 `ThreadPoolExecutor(max_workers=10)` 并行调用 `_decide_one`，每个 agent 看到的是同一份 `active_tags` 快照，但看不到其他并发 agent 正在 accept 的提案。

**风险场景**：Maintainer 在一批里产出两个语义接近的 CREATE 提案（如 `langgraph_state_design` 和 `langgraph_state_contract`），两个 Governor agent 同时评估，各自的 `search_knowledge_records` 查不到对方即将 accept 的 tag（因为两个 tag 都还不存在），两者可能各自 accept，产出重叠 tag。

**实际风险低**：cold_start 提案来自 HDBSCAN cluster（相似 records 已聚合，不会产出语义重叠的提案）；warm_start 贪心聚类同理；audit heuristic 是确定性的。只有 audit ReAct agent 自由探索时才有可能对两个相似 gap 各提一个 create，风险不为零但概率低。

**如出现**：下次 audit 会检测到 overlap_ratio 高的 tag 对并产出 merge 提案，Governor 决策后合并，自动收敛。

**根治方式（待定）**：governance 串行化（`max_workers=1`）或在 `_taxonomy_governance` 收集所有决策后对 accept 的 create 提案做后验语义去重再批量写 DB。

#### warm_start 孤立 record 不提案是正确行为

贪心聚类过滤掉 size=1 的 cluster，孤立 record 落 `no_tag` 等待。这不是缺陷：当同语义域的新 records 进来后，孤立 record 和新 records 一起聚 cluster，自然产出新 tag 提案。实测验证：`k8s-001` 和 `syn-docker-002` 在落 `no_tag` 后，下一批加入 6 条语义邻居时成功聚成 cluster，分别提出 `kubernetes_resource_isolation` 和 `container_image_hardening` 并被 Governor accept。

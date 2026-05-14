# Knowledge Consolidation Spec

## Current Scope

The active pitfall consolidation pipeline produces durable canonical rules,
governed mechanism tags, and rule-tag assignments.

Generic session knowledge extraction is a separate upstream-adjacent pipeline.
It stores admitted `KnowledgeRecord` rows in `source_knowledge_records`, but
those records are not canonicalized, tagged, or included in this consolidation
graph.

The extraction pipeline can run an optional Evidence Agent before persistence.
The agent uses a bounded Plan-Execute-Judge workflow: one LLM call creates
1-4 deterministic `search_text` / `search_turns` actions, the executor reads
full candidate turns with `get_turn`, and a judge LLM admits, rejects, or asks
for one additional search pass. A final judge may only admit or reject, so the
agent has no recursive tool loop. A record is admitted only when the judge
returns `verdict=admit` and at least one concrete evidence turn. Records with
direct evidence are written with `evidence_turns` and `evidence_count`; records
without direct evidence, including empty-evidence admits, are soft rejected by
writing `evidence_count=0`. Soft rejected records are retained for audit and
future re-verification, but default embedding and retrieval paths exclude them.

Evidence Agent batch verification runs at session granularity. A session worker
parses the session XML once, reuses `get_turn` results from that parsed map, and
caches repeated text searches for records from the same session. Semantic
search uses the Chroma turn index built by `--embed-turns`; before verification,
the pipeline detects sessions missing from that index and embeds those session
turns so RAG coverage stays complete for the run. `search_turns` uses Chroma raw
distances and normalizes scores to `0..1` within each query result set, so
scores are only relative ordering signals for that query and are not cross-query
confidence values.

Canonical pitfalls and generic knowledge records can now be embedded by a
manual indexing command. Embedded canonical pitfalls can be used to retrieve
semantically related generic knowledge records, but this retrieval path is not
part of the consolidation graph.

The pattern layer is reserved for future work and is not part of the current
runtime, schema, prompts, or tests. Current runs must not synthesize abstract
patterns.

## Consolidation Graph

The implementation is a LangGraph workflow with cold-start and warm-start
routes.

Cold start has no active tags:

1. `canonicalize_sources`
2. `taxonomy_draft`
3. `taxonomy_governance`
4. `rule_classification`
5. `persist_assignments`

Warm start has active tags:

1. `canonicalize_sources`
2. `rule_classification`
3. `tag_coverage_check`
4. `taxonomy_draft`, if any canonical rules are uncovered
5. `taxonomy_governance`, if taxonomy drafting ran
6. `rule_classification_uncovered`, if governance ran
7. `persist_assignments`

If `tag_coverage_check` finds no uncovered canonical rules, later taxonomy and
uncovered-classification nodes are no-ops before `persist_assignments`.

`tag_coverage_check` uses `LLMTagCoverageChecker` to decide which canonical
rules cannot be covered by existing active tags and returns
`uncovered_canonical_ids`.

`rule_classification_uncovered` classifies only uncovered canonical rules not
already assigned by governance, then merges `governance_assignments` from
canonical-tag relationships assigned directly by governance decisions.

Routing is conditional:

- `_route_after_canonicalization` selects cold start or warm start from
  `is_warm_start`
- `_route_after_governance` returns to normal classification on cold start and
  to uncovered-rule classification on warm start
- `_route_after_rule_classification` persists cold-start classifications and
  sends warm-start classifications through `tag_coverage_check`

Each node has one responsibility. Taxonomy discovery and rule classification are
strictly separated.

## Canonicalization

Source pitfalls are stored as `source_pitfall_records`.

The canonicalization node is an LLM structured-output stage in real runs. It
receives all pending source pitfalls plus the active canonical rules, then
returns:

- stable canonical drafts for genuinely new rules
- one decision for every source record
- a relation value for each decision: `duplicate`, `overlap`, `parent_child`,
  or `distinct`

Every source record must be linked either to an existing canonical rule or to a
new canonical draft from the same response. The node must not silently drop
overlap or parent-child records.

Validation requires:

- every source record is decided exactly once
- every decision references an existing canonical id or a new draft temporary id
- every new draft cites at least one source record
- new drafts contain stable title, summary, preventive rule, category, and scope

Tests may inject a deterministic canonicalizer to avoid external LLM calls, but
production CLI runs use the LLM canonicalizer when model settings are present.

## Taxonomy Draft

`taxonomy_draft` receives:

- untagged canonical rules
- current active mechanism tags

It returns only `TaxonomyDraftResult.proposals`.

It must not classify rules. A proposal represents a reusable prevention control,
not a category, tool, failure symptom, incident title, or one-off situation.

Validation requires:

- unique normalized proposal names
- no duplicate of an active tag
- at least one supporting canonical id per proposal
- every supporting canonical id exists in the batch

## Taxonomy Governance

`taxonomy_governance` receives active tags, proposals, and supporting rules.

It must decide every proposal exactly once:

- `accept`: create a normalized active tag
- `merge`: map the proposal to an existing active tag or an accepted tag from
  the same response
- `reject`: discard unsupported, over-broad, duplicate, tool-only,
  symptom-only, or one-off proposals

If the LLM structured decision cannot be parsed after retries, governance
soft-rejects that proposal instead of failing the consolidation run. This keeps
the taxonomy conservative: failed decisions do not create or merge tags.

Governance can normalize accepted tag names and definitions. It does not assign
rules.

## Rule Classification

`rule_classification` receives untagged canonical rules and the governed active
tag pool.

It must return exactly one assignment per canonical id. It cannot create,
rename, merge, or propose tags. `tag_name` must match an active governed tag
name exactly after normalization.

If the tag pool cannot classify the batch, validation fails and no partial
assignments are written for that failed run.

## Persistence

The current schema includes:

- `source_pitfall_records`
- `source_knowledge_records`
- `canonical_knowledge`
- `knowledge_instance_links`
- `knowledge_relations`
- `mechanism_tags`
- `tag_proposals`
- `rule_tag_assignments`
- `consolidation_runs`
- `consolidation_failures`
- `agent_invocations`

The current schema does not create `abstract_patterns` or
`pattern_rule_links`.

Existing databases that still contain older pattern tables may remain on disk,
but the current code ignores them.

`source_knowledge_records` is intentionally parallel to the pitfall source
record path. It stores generic transferable session insights with evidence
turns from engineered XML sessions. Current consolidation nodes read
`source_pitfall_records` only. Knowledge `record_id` values include the session
id, relative source path, and extracted title so duplicate rollout files that
share a session id cannot overwrite each other's records.

Embedding columns:

- `canonical_knowledge.embedding`: JSON-encoded vector for active canonical
  pitfall retrieval and tag matching.
- `mechanism_tags.embedding`: JSON-encoded vector for tag matching.
- `source_knowledge_records.embedding`: JSON-encoded vector for verified
  generic knowledge retrieval.

Embedding writes are protected by the store lock. Saving an embedding for a
missing row raises `ValueError` instead of silently succeeding.

## Embedding And Retrieval

The manual embedding command is:

```bash
uv run python -m consolidate_agent --embed
```

The command embeds only verified generic knowledge records
(`source_knowledge_records.evidence_count > 0`). Soft rejected records remain in
the source table, but they are not embedded by default.

The command:

- loads `KnowledgeStore` for the selected `--knowledge-db-path`
- creates DashScope-compatible embeddings from settings
- embeds active canonical pitfalls through `KnowledgeVectorStore.embed_canonicals`
- embeds all generic knowledge records through
  `KnowledgeVectorStore.embed_knowledge_records`
- skips rows that already have embeddings
- prints counts for rows that were missing embeddings before the run

Generic knowledge record text is embedded as:

```text
Title: {record.title}
Insight: {record.insight}
Applicability: {record.applicability}
```

Pitfall-to-knowledge retrieval is exposed as:

```python
find_related_knowledge(store, embeddings, canonical_id, threshold=0.7, top_k=3)
```

The function reads the canonical pitfall embedding from
`canonical_knowledge.embedding`, compares it with stored
`source_knowledge_records.embedding` vectors by cosine similarity, filters by
threshold, excludes soft rejected records (`evidence_count=0`), sorts
descending, and returns up to `top_k` dict results containing:

- `record_id`
- `title`
- `insight`
- `applicability`
- `scope`
- `similarity_score`

If the canonical pitfall has no embedding, or if there are no knowledge
records, retrieval returns an empty list.

## Observability

LLM stages record `agent_invocations`.

This observability currently applies to consolidation stages. The generic
knowledge extraction pipeline logs progress and writes records, but it does not
yet persist per-call `agent_invocations`. During preprocessing it also tracks
session skip reason counts in `KnowledgeExtractionStats.skip_reasons` and logs
them in the `knowledge preprocess_sessions done` progress line. Current reason
keys are:

- `unreadable_or_empty`
- `sub_agent`
- `no_action`
- `too_small`
- `too_large`
- `already_processed`

Evidence Agent writes lightweight local traces to
`outputs/evidence-agent-failures.jsonl` when plan or judge structured output
returns `None` or raises, and when the final judge verdict rejects a record.
Structured-output traces include `record_id`, `session_id`, `title`, `insight`,
stage, error, and the raw provider response preview/hash plus `parsing_error`
when LangChain exposes them through `include_raw=True`. Judge failures also
include search actions plus a truncated search result preview, character count,
and SHA-1 hash. Reject traces use `stage=evidence_reject` and include
`judge_verdict`, `judge_reasoning`, evidence turns proposed by the judge, search
actions, and the search result preview/hash that the judge saw. Search result
and raw response previews may contain source session text, so the file is a
local debug artifact and should not be shared as sanitized telemetry.

When Plan validation fails only because the model produced more than 4
`searches`, Evidence Agent records the validation failure as a recovered trace
and continues with the first 4 search actions.

Plan and Judge structured-output parse failures are retried up to 3 attempts
with short error feedback and any exposed invalid tool-call arguments. Judge
retry is limited to format recovery: normal `reject` verdicts are not retried.
Judge `reasoning` must be a short natural-language summary and must not copy raw
JSON, code, regex, shell snippets, or backslash-heavy text into the structured
tool-call arguments. If Judge still cannot produce valid structured output after
retry, the record is soft rejected with empty evidence turns. Invalid Judge
arguments are not manually salvaged into an admit decision.

If a structured output fails validation, the retry prompt receives:

- a truncated validation or parsing error
- a truncated invalid tool-call args excerpt, when the provider exposes one
- the original stage context

Recovered retry output is recorded in the same trace file with `recovered=true`,
`recovery_method`, `attempt`, and `max_attempts`.

Run stats track:

- source records seen and linked
- canonical created and reused
- tags created
- tag proposals created
- rules tagged and classified
- taxonomy governance failures
- rule classification failures
- agent invocations and failures

## Future Pattern Layer

A pattern layer may be reconsidered only after mechanism tags are stable and
there is evidence that a higher abstraction adds value beyond tag profiles.

Future pattern work should be introduced as a separate graph extension with its
own schema migration, prompts, tests, and quality metrics.

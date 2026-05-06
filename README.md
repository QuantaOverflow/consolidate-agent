# Async Consolidate Agent

Offline-first knowledge consolidation tools for Codex session history.

The project currently has three runnable tracks:

- `pitfall extraction`: normalize Codex sessions, split large transcripts,
  extract pitfall candidates, admit reusable pitfall records, and write
  `outputs/candidates.json` plus source records in SQLite.
- `knowledge extraction`: convert complete sessions into compact XML, extract
  transferable knowledge records, and write them to `source_knowledge_records`.
- `consolidation`: canonicalize admitted pitfall source records, govern
  mechanism tags, classify canonical rules, and persist rule-tag assignments.
- `embedding index`: embed canonical pitfalls and transferable knowledge
  records for semantic retrieval.

## Development

Create the virtual environment and install dependencies:

```bash
uv venv
uv sync --extra dev
```

Run the CLI:

```bash
uv run consolidate-agent --help
```

Run pitfall extraction:

```bash
uv run python -m consolidate_agent \
  --sample-limit 20 \
  --output-dir ./outputs \
  --processed-index-path ./outputs/processed-index.json \
  --knowledge-db-path ./outputs/knowledge.db
```

Run generic session knowledge extraction:

```bash
uv run python -m consolidate_agent \
  --extract-knowledge \
  --sample-limit 20 \
  --knowledge-processed-index-path ./outputs/knowledge-processed-index.json \
  --knowledge-db-path ./outputs/knowledge.db \
  --max-session-chars 100000
```

Run pitfall extraction plus consolidation:

```bash
uv run python -m consolidate_agent \
  --sample-limit 50 \
  --output-dir /tmp/consolidate-real-sessions-50-output \
  --cursor-path /tmp/consolidate-real-sessions-50-output/cursor.json \
  --processed-index-path /tmp/consolidate-real-sessions-50-output/processed-index.json \
  --knowledge-db-path /tmp/knowledge-real-sessions-50.db \
  --run-consolidation \
  --report
```

Build embedding indexes:

```bash
uv run python -m consolidate_agent \
  --knowledge-db-path ./outputs/knowledge.db \
  --embed
```

`--embed` is a manual indexing command. It embeds active rows in
`canonical_knowledge` and rows in `source_knowledge_records`, skips records
that already have embeddings, and prints how many records were missing
embeddings before the run.

## Current Consolidation Mode

The consolidation stage now uses a LangGraph workflow with strict separation
between taxonomy discovery and rule classification.

Cold start, with no active tags:

1. canonicalize source pitfalls into durable canonical rules with an LLM structured-output stage
2. draft mechanism-tag taxonomy proposals from untagged canonical rules
3. govern proposals into accepted, merged, or rejected active tags
4. classify each untagged canonical rule against the governed tag pool
5. persist rule-tag assignments

Warm start, with active tags:

1. canonicalize source pitfalls into durable canonical rules with an LLM structured-output stage
2. classify untagged canonical rules against the active tag pool
3. check whether omitted canonical rules are covered by existing tags
4. draft and govern taxonomy only for uncovered canonical rules, if any
5. classify remaining uncovered canonical rules and persist assignments

Rule classification cannot create or rename tags. If the governed taxonomy is
insufficient, the batch fails instead of silently creating labels during
classification.

`LLMTagCoverageChecker` decides which omitted canonical rules cannot be covered
by existing active tags before the workflow drafts new taxonomy proposals.

The abstract pattern layer is intentionally out of the active extraction path.
It can be reintroduced later only if governed tags become stable enough to
support a higher-level abstraction that is not just a duplicate of tag profiles.

Generic session knowledge records are intentionally parallel to the pitfall
consolidation graph today. They can be embedded for retrieval and compared
against embedded canonical pitfalls with `find_related_knowledge`, but they
are not canonicalized into rules or mechanism tags.

# Async Consolidate Agent

Offline-first pitfall extraction pipeline for Codex session history.

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

Run a larger sample with consolidation and an observability report:

```bash
uv run python -m consolidate_agent.cli \
  --sample-limit 50 \
  --output-dir /tmp/consolidate-real-sessions-50-output \
  --cursor-path /tmp/consolidate-real-sessions-50-output/cursor.json \
  --processed-index-path /tmp/consolidate-real-sessions-50-output/processed-index.json \
  --knowledge-db-path /tmp/knowledge-real-sessions-50.db \
  --run-consolidation \
  --report
```

The abstract pattern layer is intentionally out of the active extraction path.
It can be reintroduced later only if governed tags become stable enough to
support a higher-level abstraction that is not just a duplicate of tag profiles.

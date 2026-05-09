# Async Consolidate Agent

Offline-first knowledge consolidation tools for Codex session history.

The project has transitioned to a unified knowledge extraction pipeline, consolidating both pitfalls and generic transferable insights into durable `KnowledgeRecord` objects.

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

### Knowledge Pipeline Stages

The CLI now operates via a `--stage` argument to run specific parts of the pipeline:

Run knowledge extraction (without verification):

```bash
uv run python -m consolidate_agent \
  --stage extract \
  --sample-limit 20 \
  --output-dir ./outputs \
  --knowledge-processed-index-path ./outputs/knowledge-processed-index.json \
  --knowledge-db-path ./outputs/knowledge.db
```

Verify extracted records with Evidence Agent:

```bash
uv run python -m consolidate_agent \
  --stage verify \
  --evidence-workers 10 \
  --knowledge-db-path ./outputs/knowledge.db \
  --evidence-failure-trace-path ./outputs/evidence-agent-failures.jsonl \
  --evidence-reject-trace-path ./outputs/evidence-agent-rejects.jsonl
```

Run tag extraction and deduplication:

```bash
uv run python -m consolidate_agent \
  --stage tag \
  --knowledge-db-path ./outputs/knowledge.db
```

Embed active knowledge records:

```bash
uv run python -m consolidate_agent \
  --stage embed \
  --knowledge-db-path ./outputs/knowledge.db
```

Export knowledge records to Obsidian:

```bash
uv run python -m consolidate_agent \
  --stage export \
  --knowledge-db-path ./outputs/knowledge.db \
  --export-dir ~/Documents/Obsidian/Knowledge
```

Run the complete pipeline end-to-end:

```bash
uv run python -m consolidate_agent \
  --stage full \
  --sample-limit 50 \
  --output-dir ./outputs \
  --knowledge-processed-index-path ./outputs/knowledge-processed-index.json \
  --knowledge-db-path ./outputs/knowledge.db \
  --export-dir ~/Documents/Obsidian/Knowledge
```

### Utilities

Build reusable turn embeddings for session fragments (uses Chroma):

```bash
uv run python -m consolidate_agent \
  --embed-turns \
  --knowledge-db-path ./outputs/knowledge.db
```

Search embedded knowledge records:

```bash
uv run python -m consolidate_agent \
  --query "python environment setup issues" \
  --top-k 5 \
  --knowledge-db-path ./outputs/knowledge.db
```

## Current Consolidation Mode

The consolidation stage uses a LangGraph workflow with strict separation between taxonomy discovery and rule classification.

Generic session knowledge records and pitfalls are now consolidated into a unified extraction pipeline. They are embedded for retrieval and compared against embedded canonical pitfalls with `search_knowledge`, and are canonicalized into rules or mechanism tags. Turn embeddings are stored using Langchain Chroma.

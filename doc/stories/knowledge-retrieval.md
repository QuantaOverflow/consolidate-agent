# Knowledge Retrieval — User Stories

## Background

`source_knowledge_records` stores transferable knowledge extracted from Codex sessions as KnowledgeRecord entries. These stories define the embedding and cross-type retrieval capabilities needed to connect pitfalls and knowledge.

Status: US-1 and US-2 are implemented as manual indexing and lookup paths.

---

## US-1: Build Vector Index For Knowledge Records And Pitfalls

**Implementation**

- CLI: `uv run python -m consolidate_agent --embed`
- Store columns: `canonical_knowledge.embedding`,
  `source_knowledge_records.embedding`
- Knowledge text embedded as:

```text
Title: {record.title}
Insight: {record.insight}
Applicability: {record.applicability}
```

- Idempotency: existing embeddings are skipped.
- Output statistics:

```text
Embedded canonicals: N
Embedded knowledge records: M
```

**Story statement**

As the system,  
I want knowledge records and canonical pitfalls to be embedded and stored as vectors in the database,  
So that semantic search across both types is possible for downstream retrieval.

**Acceptance Criteria**

```gherkin
Scenario: embed knowledge records
  Given source_knowledge_records contains N verified records with no embeddings
  When the embedding pipeline runs
  Then each verified record has an embedding vector stored in the database
  And soft rejected records with evidence_count=0 are not embedded
  And records already embedded are skipped without re-embedding

Scenario: embed canonical pitfalls
  Given canonical_knowledge contains M pitfall records with no embeddings
  When the embedding pipeline runs
  Then each pitfall has an embedding vector stored in the database
  And records already embedded are skipped without re-embedding

Scenario: embedding pipeline is idempotent
  Given all records already have embeddings
  When the embedding pipeline runs again
  Then no new embedding API calls are made
  And existing embeddings are unchanged
```

---

## US-2: Find Related Knowledge For A Pitfall

**Implementation**

- API: `find_related_knowledge(store, embeddings, canonical_id, threshold=0.7, top_k=3)`
- The lookup uses an existing canonical pitfall embedding from
  `canonical_knowledge.embedding`.
- It compares that vector with existing knowledge record embeddings from
  `source_knowledge_records.embedding`.
- It excludes soft rejected knowledge records where `evidence_count=0`.
- It returns dict results with `record_id`, `title`, `insight`,
  `applicability`, `scope`, and `similarity_score`.
- If the canonical pitfall has no embedding, or there are no knowledge
  records, it returns an empty list.

**Story statement**

As a developer handling a known pitfall,  
I want the system to find semantically related knowledge records,  
So that I know not only that a pitfall exists, but also whether there is a verified way to address it.

**Acceptance Criteria**

```gherkin
Scenario: pitfall has related knowledge above threshold
  Given a canonical pitfall about unhandled exceptions in tool calls
  And knowledge records exist with similarity scores of 0.85, 0.72, and 0.55 to that pitfall
  When related knowledge retrieval is triggered with similarity_threshold=0.7 and top_k=3
  Then records with scores 0.85 and 0.72 are returned
  And the record with score 0.55 is excluded
  And each result contains a similarity_score field

Scenario: no sufficiently related knowledge exists
  Given a canonical pitfall with no knowledge records above similarity_threshold=0.7
  When related knowledge retrieval is triggered
  Then an empty list is returned
  And no error is raised
```

---

## Notes

Removed/deferred scenarios:

- cwd-based retrieval remains deferred until there is a concrete injection
  path for new sessions.
- session review reports remain deferred. Session-to-pitfall matching by raw
  embedding is too coarse; session-end review should likely be agentic, with an
  agent reading the engineered session context and reasoning over known
  pitfalls and knowledge.

These scenarios are outside the current sprint scope.

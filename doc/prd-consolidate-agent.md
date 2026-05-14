# Async Consolidate Agent PRD

## 1. Document Status
- Status: Draft v1
- Last updated: 2026-05-10
- Owner: Shiwenjie
- Scope: Implementation updated

## 2. Product Goal
- Help a single user review past AI interactions, extract reusable knowledge, identify recurring problems, and accumulate durable personal working knowledge for future sessions and agents.
- Current implementation unifies pitfall extraction and generic knowledge extraction into a single unified knowledge extraction pipeline that produces durable `KnowledgeRecord` entries.

## 3. Target User
- Primary user: Shiwenjie only.
- First version is for personal use, not team sharing.
- The resulting knowledge may later be consumed by future agents as a long-term knowledge layer.

## 4. Problem Statement
- Historical AI conversations contain useful patterns, mistakes, and working tactics, but they are trapped inside raw transcripts.
- Manual review is expensive and inconsistent.
- The user needs a background system that continuously turns past interactions into structured, reusable knowledge.

## 5. Input Sources
- V1 input source: local Codex session files under `~/.codex/sessions/**/*.jsonl`.
- Included content:
  - `user` messages
  - `assistant` messages
  - `function_call`
  - `function_call_output`
  - selected high-signal `event_msg`
- Excluded by default:
  - `developer` instructions
  - environment boilerplate
  - low-signal raw logging noise

### Filtering Principle
- Do not filter by channel or message type alone.
- Filter by decision value and evidence value.
- Remove framework boilerplate and low-signal progress chatter.
- Keep any content that changes strategy, reveals a failed assumption, or provides evidence for a pitfall.

### High-Signal Event Policy
- Strongly include:
  - `exec_command_end`
  - tool-related events that expose failure reasons, runtime assumptions, or execution interruptions
- Weakly include:
  - file edit / patch completion events only when they help explain the pitfall outcome
- Exclude by default:
  - `web_search_end`
  - generic lifecycle/status events without clear pitfall evidence

## 6. Core Concepts
- Transcript: normalized conversation/session record.
- Insight: structured finding extracted from one or more transcripts.
- Knowledge: consolidated, reusable insight that is stable enough to reuse later.
- Consolidation: background process that merges insights into knowledge.

## 7. Core Workflows
- Manual trigger starts an incremental background pipeline via CLI stages (`extract`, `verify`, `tag`, `embed`, `export`).
- New Codex session transcripts are normalized into a filtered XML representation.
- The unified knowledge pipeline extracts transferable insights, decision rationale, debugging approaches, architecture trade-offs, and domain understanding, creating `KnowledgeRecord` entries.
- Evidence Agent verifies candidates against session XML, ensuring concrete evidence exists before admission.
- Sessions' turns are embedded using Langchain Chroma for semantic verification and future RAG.
- Accepted records update the long-term knowledge base.
- The system keeps a cursor/checkpoint so later runs process only newly added sessions.

## 7.1 Transformation Stages
1. `Session Context Engineering`
   - Input: raw Codex session JSONL events
   - Output: compact XML session representation
   - Responsibility:
     - preserve high-signal user, assistant, command, patch, web search, rollback, compaction, and sub-agent events
     - fold file reads into path-only markers
     - group retained events into ordered turns
     - compute compression statistics
   - Implementation preference:
     - deterministic, no LLM

2. `Knowledge Extraction`
   - Input: complete engineered session XML
   - Output: provisional `KnowledgeRecord`
   - Responsibility:
     - extract transferable cognition, pitfalls, and decision rationale
     - reject `session_specific` items
   - Implementation preference:
     - LLM-driven structured output

3. `Evidence Verification`
   - Input: provisional `KnowledgeRecord` and session XML
   - Output: admitted/rejected `KnowledgeRecord`
   - Responsibility:
     - Evidence Agent executes Plan-Execute-Judge workflow to find concrete turn evidence
     - Updates `evidence_turns` and `evidence_count`
   - Implementation preference:
     - Multi-turn Agent loop with text and semantic RAG search over Chroma `session_turns`.

4. `Tag Consolidation & Embedding`
   - Input: admitted `KnowledgeRecord`
   - Output: tagged and embedded records
   - Responsibility:
     - extract and deduplicate tags
     - embed canonical knowledge records for semantic retrieval

## 8. Output Artifacts
- Personal replayable session reviews.
- Structured knowledge extracted from prior AI interactions.
- A long-term personal knowledge layer that can be reused by future agents.
- Admitted generic knowledge records stored in SQLite with semantic embeddings in Chroma vector store.
- Obsidian Markdown export for human review and browsing.

## 8.1 Generic Knowledge Record Shape
```json
{
  "id": "knowledge_xxx",
  "session_id": "019d8b46-1564-7733-a75b-cb19d1a2cb70",
  "title": "Structured tool errors improve agent robustness",
  "insight": "Tool wrappers should return structured error records instead of surfacing raw exceptions to downstream orchestration.",
  "applicability": "Applies when designing tools that are called inside agentic pipelines or batch extraction flows.",
  "scope": "global",
  "evidence_turns": [1, 3],
  "evidence_count": 2,
  "processed_chars": 43000,
  "created_at": "2026-05-04T00:00:00Z",
  "updated_at": "2026-05-04T00:00:00Z"
}
```

### Generic Knowledge Principles
- `KnowledgeRecord` is extracted from a complete engineered session, not from transcript chunks.
- `scope == session_specific` items do not enter the long-term DB by default.
- `evidence_turns` must be non-empty and refer to engineered XML turn indexes.
- Generic knowledge records are canonicalized, tagged, and embedded for future retrieval.

## 9. Non-Goals
- Automatic background scheduling.
- Multi-user/team knowledge sharing.
- Proactive injection of extracted knowledge into future Codex sessions (V1).

## 10. Success Criteria
- V1 should reliably surface recurring failure modes, pitfalls, and transferable insights from historical AI interactions.
- Generic knowledge extraction should favor fewer high-quality transferable insights over broad session summaries.
- Every admitted generic knowledge record must be traceable to one or more engineered turn indexes (verified by Evidence Agent).
- V1 should favor precision over recall.
- Outputs should be usable both for human review (Obsidian export) and future agent consumption.

## 10.1 Admission Criteria
- A knowledge record may enter the long-term library only if it is:
  - reusable beyond the exact original moment
  - supported by clear transcript evidence verified by Evidence Agent
  - scoped as `global` or `project_specific`
- `session_specific` candidates are rejected.

## 11. Open Questions
- How should the current manual embedding/retrieval path be connected to session-start context injection or review reports?

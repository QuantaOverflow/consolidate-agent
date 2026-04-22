# Async Consolidate Agent PRD

## 1. Document Status
- Status: Draft v0
- Last updated: 2026-04-22
- Owner: Shiwenjie
- Scope: Requirement alignment in progress

## 2. Product Goal
- Help a single user review past AI interactions, extract reusable knowledge, identify recurring problems, and accumulate durable personal working knowledge for future sessions and agents.
- V1 priority: build a recurring problem and pitfall knowledge base from historical AI interactions.

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
- Manual trigger starts an incremental background analysis job.
- New Codex session transcripts are normalized into a filtered transcript representation.
- The system extracts pitfall candidates focused on `execution_strategy` and `tooling_environment`.
- Candidates are checked against admission criteria.
- Accepted pitfalls update the long-term pitfall library in both structured and human-readable forms.
- The system keeps a cursor/checkpoint so later runs process only newly added sessions.

## 7.1 Transformation Stages
1. `Normalization`
   - Input: raw Codex session JSONL events
   - Output: normalized `Transcript`
   - Responsibility:
     - filter boilerplate and low-signal noise
     - preserve user/assistant/tool/high-signal event evidence
     - build a single ordered event stream
   - Implementation preference:
     - deterministic first

2. `Extraction`
   - Input: normalized `Transcript`
   - Output: one or more `PitfallCandidate`
   - Responsibility:
     - extract only V1 pitfall categories
     - produce structured candidate records
     - attach evidence refs and confidence
   - Implementation preference:
     - LLM-driven but schema-constrained

3. `Admission + Consolidation`
   - Input: `PitfallCandidate`
   - Output: accepted/updated `PitfallRecord`
   - Responsibility:
     - enforce admission criteria
     - reject low-value or session-only candidates
     - deduplicate or merge with existing pitfall records
     - update structured records and readable pitfall cards
   - Implementation preference:
     - policy-driven with deterministic post-processing where possible

## 8. Output Artifacts
- Personal replayable session reviews.
- Structured knowledge extracted from prior AI interactions.
- A long-term personal knowledge layer that can be reused by future agents.
- V1 primary artifact: a recurring problem / pitfall library.
- V1 output format:
  - structured pitfall records for system use
  - readable pitfall cards for human review and browsing

## 8.1 Pitfall Schema v0
- `title`: short human-readable label for the pitfall.
- `trigger`: what condition or context tends to trigger it.
- `failure_mode`: what went wrong.
- `impact`: why it matters or what cost it caused.
- `preventive_rule`: what to do differently next time.
- `evidence`: references to supporting transcript segments.
- `scope` (required): whether the pitfall is global, project-specific, or session-specific.
- `confidence`: model confidence that this is a valid reusable pitfall.
- `tags`: optional topical labels for retrieval and grouping.

## 8.1.1 Pitfall Record Shape v0
```json
{
  "id": "pitfall_xxx",
  "title": "Default interpreter assumption breaks validation",
  "category": "tooling_environment",
  "trigger": "The workflow assumes `python` is available in PATH instead of using the project interpreter.",
  "failure_mode": "Validation fails for environment reasons before the actual task is verified.",
  "impact": "Adds a noisy debugging round and delays task progress.",
  "preventive_rule": "Use the project virtual environment interpreter or the user’s active interpreter for validation commands.",
  "scope": "global",
  "evidence": {
    "session_ids": ["019d8b46-1564-7733-a75b-cb19d1a2cb70"],
    "message_refs": [
      "assistant:2026-04-14T09:16:24.843Z",
      "tool_output:python command not found"
    ]
  },
  "confidence": 0.89,
  "tags": ["python", "env", "validation"],
  "created_at": "2026-04-22T00:00:00Z",
  "updated_at": "2026-04-22T00:00:00Z"
}
```

## 8.1.2 Supporting Internal Objects
- `Transcript`
  - normalized representation of one Codex session after filtering and compression
- `PitfallCandidate`
  - provisional extracted pitfall before admission into the long-term library

### Confidence Usage
- `confidence` should be retained in V1.
- `confidence` is a review and ranking aid, not a standalone admission rule.

### Transcript Shape v0
```json
{
  "session_id": "019d8b46-1564-7733-a75b-cb19d1a2cb70",
  "thread_name": "Fix report service startup failure",
  "source": "codex-cli",
  "cwd": "/Users/shiwenjie/Desktop/playground/vibe/web_agent",
  "started_at": "2026-04-14T09:15:26.524Z",
  "messages": [
    {
      "ref": "msg_001",
      "timestamp": "2026-04-14T09:15:26.525Z",
      "role": "user",
      "kind": "message",
      "text": "..."
    },
    {
      "ref": "msg_002",
      "timestamp": "2026-04-14T09:15:55.221Z",
      "role": "assistant",
      "kind": "message",
      "text": "..."
    },
    {
      "ref": "tool_003",
      "timestamp": "2026-04-14T09:16:24.843Z",
      "role": "tool",
      "kind": "tool_output",
      "tool_name": "exec_command",
      "text": "python: command not found"
    }
  ]
}
```

### Transcript Principles
- `Transcript` is a normalized extraction input, not a raw archive copy.
- `messages` should be a single unified event stream ordered by time.
- Every retained item must have a stable `ref` so later evidence can point back to it.
- `text` should be filtered/compressed content with decision value or evidence value.

### PitfallCandidate Shape v0
```json
{
  "candidate_id": "candidate_xxx",
  "session_id": "019d8b46-1564-7733-a75b-cb19d1a2cb70",
  "title": "Default interpreter assumption breaks validation",
  "category": "tooling_environment",
  "trigger": "The workflow assumes `python` is available in PATH.",
  "failure_mode": "Validation fails before the actual task is verified.",
  "impact": "Adds a noisy debugging round.",
  "preventive_rule": "Use the project virtual environment interpreter for validation.",
  "scope": "global",
  "evidence_refs": ["msg_002", "tool_003"],
  "confidence": 0.84,
  "admission_status": "pending"
}
```

### Candidate Principles
- `PitfallCandidate` is extracted from a single normalized transcript.
- `PitfallCandidate` is not yet a durable knowledge record.
- V1 `admission_status` values:
  - `pending`
  - `accepted`
  - `rejected`

## 8.2 V1 Pitfall Categories
- `execution_strategy`
  - Pitfalls caused by choosing the wrong execution path, sequencing, validation order, or change scope.
- `tooling_environment`
  - Pitfalls caused by incorrect assumptions about tools, interpreters, shell environment, runtime context, or validation path.
- Explicitly out of scope for V1:
  - `requirement_alignment`
  - `knowledge_judgment`

## 9. Non-Goals
- To be confirmed.

## 10. Success Criteria
- V1 should reliably surface recurring failure modes and pitfalls from historical AI interactions in a way that is reusable later.
- V1 should focus on extracting actionable pitfalls in `execution_strategy` and `tooling_environment`, not a broad summary of all possible knowledge types.
- Every admitted pitfall must be traceable to source evidence.
- V1 should favor precision over recall.
- Every admitted pitfall must contain an actionable preventive rule.
- Outputs should be usable both for human review and future agent consumption.

## 10.2 Update Model
- V1 trigger mode: manual trigger only.
- V1 processing mode: incremental processing only.
- V1 should process sessions created after the last successful cursor/checkpoint.
- V1 may auto-admit accepted pitfalls into the library, but the system should preserve a path for later human review.

## 10.1 Admission Criteria For Pitfall Library
- A pitfall may enter the long-term library only if it is:
  - reusable beyond the exact original moment
  - supported by clear transcript evidence
  - expressible as a concrete preventive rule
  - scoped as `global` or `project_specific`
- `session_specific` pitfall candidates should not enter the long-term library by default.
- Higher-impact pitfalls should be prioritized, but impact is a ranking factor rather than a hard admission gate in V1.

## 11. Open Questions
- What kinds of knowledge matter most in v1?
- What kinds of pitfalls should be prioritized in v1?
- What are the minimum required fields for a valid pitfall record?

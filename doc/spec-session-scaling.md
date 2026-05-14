# Session Scaling Spec

## Status
- Status: Draft v0
- Date: 2026-04-23
- Scope: Scale the offline pitfall extraction MVP from small single sessions to large and many Codex sessions without exceeding LLM context limits.

This spec describes the pitfall extraction path only. The generic knowledge
extraction path uses a different contract: raw session JSONL is first converted
to compact XML by `SessionContextEngineer`, then one complete bounded session
XML is sent to the LLM. Oversized engineered sessions are skipped by
`max_session_chars` rather than chunked.

## Problem
The current MVP can extract pitfall candidates from a small normalized Codex session, but it does not scale to the full local history.

Observed issue:
- A recent session normalized to 397 retained messages and about 169k characters.
- Sending that whole transcript to the LLM is slow and may exceed practical context limits.
- Running all sessions as one prompt is not viable.

The system needs to process all historical sessions without requiring any LLM call to see all raw transcripts at once.

## Design Decision
Do not feed all sessions, or very large sessions, into one LLM call.

Use a layered extraction pipeline:

```text
raw sessions
  -> per-session normalization
  -> transcript chunking when needed
  -> chunk-level pitfall candidates
  -> session-level candidate merge
  -> library admission and consolidation
```

The LLM should only operate on bounded inputs:
- one small transcript, or
- one transcript chunk, or
- a compact set of structured candidates.

Long-term pitfall knowledge should be consolidated from structured candidates,
not from raw full-history transcripts.

## Data Contracts

### TranscriptChunk
A bounded slice of a normalized transcript.

Required fields:
- `session_id`: original session id
- `chunk_id`: stable chunk id
- `chunk_index`: zero-based chunk index
- `total_chunks`: number of chunks for the session
- `messages`: ordered subset of `Transcript.messages`
- `char_count`: approximate serialized text size

Rules:
- Chunking must preserve message order.
- Evidence refs must remain the original transcript message refs.
- Chunks may overlap by a small number of messages to reduce boundary loss.

### ProcessedSessionState
Durable processing state for one session.

Required fields:
- `session_id`
- `path`
- `normalized_hash`
- `status`: `pending | processing | processed | failed`
- `chunk_count`
- `processed_at`
- `error`

### ProcessedIndex
Durable index of processed sessions.

Required fields:
- `sessions`: map from `session_id` to `ProcessedSessionState`
- `updated_at`

Purpose:
- avoid reprocessing completed sessions
- support retrying failed sessions
- avoid relying only on lexicographic `last_session_path`

## Contracts and Invariants

These rules must hold regardless of implementation details:

- The pipeline must never send all historical sessions to one LLM call.
- A transcript larger than the configured extraction budget must be split before LLM extraction.
- Chunking must preserve original `TranscriptMessage.ref` values.
- Candidate `evidence_refs` must refer to original transcript message refs, not chunk-local refs.
- `PitfallRecord` schema should remain unchanged in this iteration.
- `scope == session_specific` candidates must not enter the long-term pitfall library.
- LLM extraction should operate only on bounded transcript chunks or compact structured candidates.
- Normalization, chunking, admission, dedupe, rendering, and state updates should remain deterministic.

## Implementation Decisions

These decisions are intentionally fixed for this implementation slice:

- Add `TranscriptChunk` as an internal bounded extraction input.
- Add optional `chunk_id` to `PitfallCandidate`.
- Do not add `chunk_id` to `PitfallRecord.evidence` in this iteration; long-term evidence remains based on `session_ids` and original `message_refs`.
- Introduce `ProcessedIndex` as the primary durable processing state.
- Default processed index path should be `outputs/processed-index.json`.
- Add a CLI option `--processed-index-path`.
- Keep existing `--cursor-path` temporarily for compatibility, but new skip/retry logic should use `ProcessedIndex`.
- Empty LLM candidate output is a successful extraction result, not a failure.
- If any chunk extraction raises an exception, the owning session is marked `failed`.
- Failed sessions are eligible for retry on later runs.
- A session is marked `processed` only after session-level merge and all outputs are written successfully.

## Pipeline Behavior

### 1. Discovery
Current behavior discovers sessions by sorted file path and cursor.

New behavior:
- discover all candidate session files
- compute or load session metadata
- skip sessions with matching `normalized_hash` and `status == processed`
- include failed or changed sessions for retry

### 2. Normalization
Keep current deterministic normalization behavior.

Add normalized hash calculation based on the normalized transcript payload.

### 3. Chunking
Add `split_transcript(transcript, max_chunk_chars, overlap_messages)`.

Recommended defaults:
- `max_chunk_chars = 30000`
- `overlap_messages = 5`

Behavior:
- If transcript is under budget, create one chunk.
- If over budget, split by message order.
- Never split inside a retained message.
- Preserve original refs so candidates can cite source evidence.

### 4. Extraction
Extractor should accept chunk-like bounded inputs.

Candidate additions:
- include `chunk_id` when extracted from a chunk
- keep `session_id` as the owning session
- keep `evidence_refs` pointing to original transcript refs

### 5. Session-level Merge
After all chunks for one session are extracted:
- deduplicate candidates within the session
- merge evidence refs for near-identical candidates
- keep accepted/rejected candidate status separate from library admission

V1 can use deterministic matching first:
- `category + normalized title + normalized preventive_rule`

### 6. Library Admission
Use the existing admission criteria:
- must be reusable
- must have evidence refs
- must have preventive rule
- `scope != session_specific`

Library dedupe remains conservative and deterministic in this phase.

### 7. Processed Index Update
Only mark a session as `processed` after:
- all chunks were extracted or intentionally skipped
- session-level merge completed
- outputs were written successfully

Failed sessions should be marked `failed` with an error message and retried on a later run.

## State Semantics

### `pending`
- The session is known but has not started processing in the current run.

### `processing`
- The session is currently being normalized, chunked, extracted, merged, or written.
- A session should not remain in this state after a normal run completes.

### `processed`
- The session's normalized content hash matched the processed payload.
- All chunks were extracted or intentionally produced no candidates.
- Session-level merge completed.
- Output files and processed index were written successfully.

### `failed`
- At least one required step failed for the session.
- The error field should capture a concise failure reason.
- Failed sessions should be retried by default on later runs.

### Empty Result Semantics
- If the LLM returns zero candidates for all chunks without throwing, the session is still `processed`.
- Zero candidates means "no qualifying pitfall found", not a processing failure.

## Non-goals
This spec does not implement:
- async subagents
- generic knowledge extraction storage
- vector similarity clustering
- embedding-based deduplication
- automatic cron/background scheduling
- full raw transcript summarization
- proactive retrieval during future Codex sessions

## Acceptance Criteria
- A session larger than the LLM input budget is split into multiple chunks.
- Each chunk can be extracted independently.
- Candidate evidence refs still point back to original transcript message refs.
- Re-running the pipeline skips unchanged processed sessions.
- Failed sessions can be retried.
- Small sessions still produce the same shape of candidates and pitfall records as the current MVP.
- The pipeline never attempts to send all historical sessions to one LLM call.

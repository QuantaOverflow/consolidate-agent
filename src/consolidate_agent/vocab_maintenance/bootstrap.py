"""Two-step tag vocabulary extraction.

Step 1 — Distill: per-record theme descriptions (not tags).
Step 2 — Synthesize: cluster themes into a tag vocabulary in one shot.

Separating "identify the theme" from "name the tag" prevents the LLM from
inventing specific names while looking at concrete record details.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from .observability import invoke_with_retry
from consolidate_agent.consolidation._utils import _chat_model


# ── Step 1: Distill ────────────────────────────────────────────────────────────

class RecordTheme(BaseModel):
    record_idx: int = Field(description="1-based index within the batch")
    theme: str = Field(description="one sentence (<=25 words) describing the pattern/concept/lesson")


class DistillBatchOutput(BaseModel):
    themes: list[RecordTheme]


DISTILL_SYSTEM = """You read engineering knowledge records. For each record in the batch, write ONE concise sentence capturing the underlying pattern, concept, or lesson it teaches.

Rules:
- Describe the WHY/WHAT of the lesson, NOT the specific symptom or product
- Abstract level: pattern / concept / anti-pattern (e.g., "silent failure goes undetected without explicit rowcount validation", "state must be isolated when multiple agents share a checkpoint store")
- DO NOT mention specific product, library, framework, or platform names (no langgraph, mysql, playwright, astro, mcp, sqlite, codex, etc.)
- One sentence per record, max 25 words
- Stay faithful to the record's actual content"""


DISTILL_USER = """Batch of {batch_size} records:

{records}

For each record (by record_idx 1..{batch_size}), output the theme sentence."""


# ── Step 2: Synthesize ─────────────────────────────────────────────────────────

class TagDefinition(BaseModel):
    name: str = Field(description="snake_case, 1-2 words")
    definition: str = Field(description="one sentence")
    theme_indices: list[int] = Field(
        default_factory=list,
        description="0-based theme indices in the input list that belong to this tag",
    )


class SynthesizeOutput(BaseModel):
    vocab: list[TagDefinition]
    notes: str = Field(description="brief summary of vocab structure and any difficult cases")


SYNTHESIZE_SYSTEM = """Label each theme with the most SPECIFIC engineering pattern it represents. Group themes sharing the same specific pattern into a tag.

You are NOT being asked to find broad categories. You are being asked to surface fine-grained, named patterns that practitioners would recognize. Think "specific lesson", not "topic area".

NAMING RULES (absolute):
1. Tag: 1-2 words, snake_case
2. Tag = a specific recurring pattern, technique, or anti-pattern
3. NEVER use product/library/framework names: langgraph, playwright, mysql, sqlite, mcp, astro, codex, macos, pytest, asyncio, etc.
4. Same abstraction level across all tags
5. No singleton tags — each covers ≥ 2 themes
6. Every input theme belongs to exactly one tag

FORBIDDEN tag names (these are umbrella categories that swallow everything — refuse them outright):
- error_handling, error_management, exception_handling
- state_management, data_management, data_handling, data_integrity
- system_design, software_design, system_architecture
- code_quality, engineering_practices, best_practices, robust_systems
- testing, observability, security, performance (alone — too generic)
- ANY tag whose name is a textbook chapter title rather than a specific lesson

Why these are forbidden: tagging 100 themes as "error_handling" means readers learn nothing the input themes didn't already say. A useful vocabulary has 20-50+ distinct patterns at fine granularity.

──────── EXAMPLE ────────
Input themes (12 illustrative; your real input has hundreds):
[0] Validate file rowcounts after batch import to avoid silent corruption
[1] Concurrent agents writing same checkpoint without isolation
[2] Schema migration v1→v2 without backfill leaves dangling records
[3] Retry-once on LLM None masks underlying timeout instead of surfacing
[4] Map-reduce batch failure rolls back whole batch instead of partial commit
[5] Skip tests with @xfail without context — silent regression
[6] HITL pause uses interrupt() but resume value not validated for malformed input
[7] Embedding similarity threshold 0.5 creates false positives in tag attach
[8] Output token cap truncates JSON mid-string — parsing fails silently
[9] Two simultaneous saves overwrite each other without file lock
[10] Async generator leaks resources when consumer doesn't drain
[11] Config precedence: env > .env > cli arg — caller surprised by override

GOOD output — 8 distinct, named patterns:
- silent_failure: errors absorbed without surfacing (themes: 0, 3, 5, 8)
- state_isolation: concurrent access to shared mutable state without guards (themes: 1, 9)
- schema_evolution: migration paths between data shape versions (themes: 2)
- partial_apply: recovery committing successful work vs all-or-nothing (themes: 4)
- hitl_robustness: human pause/resume + input validation (themes: 6)
- threshold_tuning: similarity/confidence cutoff selection (themes: 7)
- resource_lifecycle: cleanup of async/external resources (themes: 10)
- config_precedence: visibility of layered configuration overrides (themes: 11)

REJECTED outputs and why:
- "error_handling" covering [0,3,4,5,8]:  ✗ — collapses 5 distinct lessons (rowcount validation, retry visibility, partial apply, test annotation, output truncation) into noise
- "state_management" covering [1,2,9]:    ✗ — collapses concurrency, migration, and locking into one mush
- "data_integrity" covering [0,2,9]:      ✗ — three different mechanisms (validation, migration, locking) sharing only that they "concern data"

Notice: 12 themes → 8 specific tags is correct. If you find yourself producing 2-5 tags for many themes, you are collapsing too hard and must split.

OUTPUT: for each tag, return name + definition + 0-based theme indices that belong to it."""


SYNTHESIZE_USER = """Total themes: {theme_count}

Themes (0-based indexed):
{themes}

Cluster these into a tag vocabulary following the rules above. Each theme belongs to exactly one tag. Avoid umbrella categories."""


# ── Utilities ──────────────────────────────────────────────────────────────────


def load_records(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT record_id, title, insight FROM source_knowledge_records ORDER BY created_at, record_id"
    ).fetchall()
    conn.close()
    return [
        {"record_id": r["record_id"], "title": r["title"], "insight": r["insight"]}
        for r in rows
    ]


def format_records_for_distill(records: list[dict]) -> str:
    lines = []
    for idx, r in enumerate(records, 1):
        insight = r["insight"][:280]
        lines.append(f"[{idx}] {r['title']}\n    {insight}")
    return "\n\n".join(lines)


def format_themes_for_synthesize(themes: list[str]) -> str:
    return "\n".join(f"[{i}] {t}" for i, t in enumerate(themes))


# ── Pipeline ───────────────────────────────────────────────────────────────────


def _distill_single_batch(
    batch_idx: int,
    batch_records: list[dict],
    distill_model,
    distill_prompt,
    total_batches: int,
    *,
    max_retries: int = 3,
) -> tuple[int, list[dict], float, str | None]:
    """Run one distill batch with retry. Thread-safe.

    Retries up to `max_retries` times with exponential backoff on exception.
    On persistent failure: returns placeholder entries (theme="" +
    distill_failed=True) so the records aren't silently dropped — preserves
    the themes ↔ records 1:1 invariant.

    Returns (batch_idx, themes_list, elapsed, error_str_or_None).
    """
    t0 = time.perf_counter()
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            messages = distill_prompt.invoke({
                "batch_size": len(batch_records),
                "records": format_records_for_distill(batch_records),
            })
            result: DistillBatchOutput = distill_model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 — log + retry
            last_err = exc
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
            continue

        elapsed = time.perf_counter() - t0
        themes_by_idx = {t.record_idx: t.theme for t in result.themes}
        batch_out = [
            {
                "record_id": rec["record_id"],
                "title": rec["title"],
                "theme": themes_by_idx.get(rec_idx, "").strip(),
                "distill_failed": False,
            }
            for rec_idx, rec in enumerate(batch_records, 1)
        ]
        return batch_idx, batch_out, elapsed, None

    # All retries failed — emit placeholder entries (preserve 1:1, mark failure).
    elapsed = time.perf_counter() - t0
    placeholder = [
        {
            "record_id": rec["record_id"],
            "title": rec["title"],
            "theme": "",
            "distill_failed": True,
        }
        for rec in batch_records
    ]
    return batch_idx, placeholder, elapsed, str(last_err)[:200]


def distill_step(
    model_factory,
    records: list[dict],
    batch_size: int,
    log_file=None,
    *,
    concurrency: int = 10,
) -> list[dict]:
    """Distill records into themes via parallel LLM batches.

    Returns list of {record_id, title, theme}, preserving input record order.

    Args:
        log_file: optional .write/.flush sink. If provided, per-batch metric
            lines are written (under a lock for thread safety).
        concurrency: ThreadPoolExecutor workers. 1 = serial.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    distill_model = model_factory().with_structured_output(DistillBatchOutput)
    distill_prompt = ChatPromptTemplate.from_messages([
        ("system", DISTILL_SYSTEM),
        ("user", DISTILL_USER),
    ])

    total_batches = (len(records) + batch_size - 1) // batch_size
    batches = []
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(records))
        batches.append((batch_idx, records[start:end]))

    log_lock = threading.Lock()

    def _emit_batch_result(batch_idx, batch_out, elapsed, err):
        with log_lock:
            if err:
                print(f"  distill batch {batch_idx + 1}/{total_batches} FAILED: {err}", flush=True)
                if log_file is not None:
                    log_file.write(json.dumps({"phase": "distill", "batch": batch_idx + 1, "error": err}) + "\n")
                    log_file.flush()
            else:
                print(f"  distill batch {batch_idx + 1}/{total_batches}  records={len(batch_out)}  "
                      f"themes={len([t for t in batch_out if t['theme']])}  ({elapsed:.1f}s)", flush=True)
                if log_file is not None:
                    log_file.write(json.dumps({
                        "phase": "distill",
                        "batch": batch_idx + 1,
                        "records": len(batch_out),
                        "themes_produced": len([t for t in batch_out if t["theme"]]),
                        "elapsed_sec": round(elapsed, 2),
                    }, ensure_ascii=False) + "\n")
                    log_file.flush()

    results_by_idx: dict[int, list[dict]] = {}
    if concurrency <= 1:
        # Serial fallback (useful for debugging / strict order)
        for idx, recs in batches:
            r_idx, batch_out, elapsed, err = _distill_single_batch(
                idx, recs, distill_model, distill_prompt, total_batches,
            )
            results_by_idx[r_idx] = batch_out
            _emit_batch_result(r_idx, batch_out, elapsed, err)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(_distill_single_batch, idx, recs, distill_model, distill_prompt, total_batches): idx
                for idx, recs in batches
            }
            for fut in as_completed(futures):
                r_idx, batch_out, elapsed, err = fut.result()
                results_by_idx[r_idx] = batch_out
                _emit_batch_result(r_idx, batch_out, elapsed, err)

    return [r for i in sorted(results_by_idx) for r in results_by_idx[i]]


def _default_target_count_range(n_themes: int) -> tuple[int, int]:
    """Heuristic: aim for ~12-25 themes per cluster (independent of dataset size).

    target_lo = max(10, n // 25)   # below this → tags too broad
    target_hi = max(20, n // 12)   # above this → over-fragmented
    For 696 themes: (27, 58). For 200: (10, 16). For 5000: (200, 416).
    """
    return max(10, n_themes // 25), max(20, n_themes // 12)


def synthesize_step(
    model_factory,
    themes: list[dict],
    log_file,
    *,
    system_prompt: str = SYNTHESIZE_SYSTEM,
) -> dict:
    """Returns synthesize raw output (vocab + notes).

    Prompt no longer asks LLM to hit a specific vocab size — count scaling
    with theme count caused LLMs to ignore guidance and over-collapse.
    Granularity is instead enforced by few-shot examples + explicit umbrella
    rejection in SYNTHESIZE_SYSTEM.
    """
    synthesize_model = model_factory().with_structured_output(SynthesizeOutput)
    synthesize_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", SYNTHESIZE_USER),
    ])

    theme_texts = [t["theme"] for t in themes]

    t0 = time.perf_counter()
    messages = synthesize_prompt.invoke({
        "theme_count": len(theme_texts),
        "themes": format_themes_for_synthesize(theme_texts),
    })
    result: SynthesizeOutput | None = invoke_with_retry(
        synthesize_model, messages, retries=3, caller="synthesize",
    )
    if result is None:
        raise RuntimeError(
            "synthesize: LLM returned None or validation error after 3 retries; "
            "check run jsonl for llm.* events"
        )
    elapsed = time.perf_counter() - t0

    log_file.write(json.dumps({
        "phase": "synthesize",
        "themes_in": len(theme_texts),
        "tags_out": len(result.vocab),
        "elapsed_sec": round(elapsed, 2),
        "notes": result.notes,
    }, ensure_ascii=False) + "\n")
    log_file.flush()

    print(f"  synthesize  themes={len(theme_texts)}  →  vocab={len(result.vocab)}  ({elapsed:.1f}s)", flush=True)

    return {
        "vocab": [t.model_dump() for t in result.vocab],
        "notes": result.notes,
    }


# ── Step 2 (map-reduce variant) ────────────────────────────────────────────────


class ConsolidatedTag(BaseModel):
    name: str = Field(description="snake_case, 1-2 words")
    definition: str = Field(description="one sentence")


class ConsolidateOutput(BaseModel):
    vocab: list[ConsolidatedTag]
    notes: str = Field(description="brief note on merge decisions / leftover concerns")


CONSOLIDATE_SYSTEM = """You're given a list of candidate tag names + definitions produced from independent batches. Many candidates are near-synonyms (same underlying lesson stated differently). Produce a clean consolidated vocabulary that captures the distinct patterns.

You're NOT being asked to partition or index candidates. You only output the final tag list. Assignment of original candidates to final tags is done by a separate downstream step.

RULES:
1. Each output tag: 1-2 words, snake_case
2. NEVER use product/library/framework names (langgraph, mysql, sqlite, mcp, codex, etc.)
3. Each output tag captures a SPECIFIC pattern (a lesson a practitioner would recognize)
4. Same abstraction level across all tags

FORBIDDEN tag names (umbrella categories — refuse them):
- error_handling, error_management, exception_handling
- state_management, data_management, data_handling, data_integrity
- system_design, software_design, system_architecture
- code_quality, engineering_practices, best_practices, robust_systems
- testing, observability, security, performance (alone — too generic)

GRANULARITY HEURISTIC:
- If the input has ~250 candidates from independent batches of ~70 themes each, the underlying vocabulary is typically 30-60 distinct patterns.
- Don't collapse to 5-10 superclusters. The candidates were already deduplicated within each batch — patterns that survive across batches are GENUINE distinct concepts.
- When you see "early_termination" / "premature_stop" / "early_exit" from different batches — they merge into one final tag. But "silent_failure" / "data_truncation" / "state_isolation" each stay separate.

OUTPUT: just the final vocabulary as (name, definition) entries. NO index lists, NO merged_from. Downstream does that."""


CONSOLIDATE_USER = """Total candidates: {n_candidates}

Candidates:
{candidates}

Produce the consolidated final vocabulary. Just name + definition for each final tag."""


def synthesize_mapreduce(
    map_model_factory,
    reduce_model_factory,
    themes: list[dict],
    log_file,
    *,
    batch_size: int = 70,
    concurrency: int = 5,
    map_system_prompt: str = SYNTHESIZE_SYSTEM,
    reduce_system_prompt: str = CONSOLIDATE_SYSTEM,
) -> dict:
    """Map-reduce synthesize.

    Map: split themes into batches (default 70 each), run SYNTHESIZE_SYSTEM
    on each — each batch produces 10-15 specific tags. LLM cannot collapse
    to 2 umbrella categories at 70-theme scale.

    Reduce: collect all candidates (~100-150), run CONSOLIDATE_SYSTEM to
    merge synonyms across batches, keeping distinct patterns separate.

    Returns same shape as synthesize_step: {vocab: [...], notes: str}.
    Each vocab entry has {name, definition, theme_indices} where indices
    are global (0..len(themes)-1), aggregated through the merged_from chain.
    """
    theme_texts = [t["theme"] for t in themes]
    n = len(theme_texts)

    # ── Map ──
    batches = [(i, theme_texts[i:i + batch_size]) for i in range(0, n, batch_size)]
    print(f"  [synth.map] {len(batches)} batches × ~{batch_size} themes", flush=True)
    log_file.write(json.dumps({"phase": "synth_map.start",
                               "batches": len(batches), "batch_size": batch_size}) + "\n")
    log_file.flush()

    map_model = map_model_factory().with_structured_output(SynthesizeOutput)
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", map_system_prompt),
        ("user", SYNTHESIZE_USER),
    ])

    def _map_one(batch_idx: int, start: int, sub_themes: list[str]):
        messages = map_prompt.invoke({
            "theme_count": len(sub_themes),
            "themes": format_themes_for_synthesize(sub_themes),
        })
        t0 = time.perf_counter()
        result = invoke_with_retry(map_model, messages, retries=3,
                                    caller=f"synth_map_{batch_idx}")
        elapsed = time.perf_counter() - t0
        if result is None:
            print(f"  [synth.map] batch {batch_idx}: FAILED after retries", flush=True)
            return batch_idx, [], elapsed
        out = []
        for tag in result.vocab:
            global_idxs = sorted({start + i for i in tag.theme_indices
                                  if 0 <= i < len(sub_themes)})
            out.append({
                "name": tag.name,
                "definition": tag.definition,
                "theme_indices": global_idxs,
                "batch_idx": batch_idx,
            })
        print(f"  [synth.map] batch {batch_idx + 1}/{len(batches)}: "
              f"{len(out)} tags ({elapsed:.1f}s)", flush=True)
        return batch_idx, out, elapsed

    from concurrent.futures import ThreadPoolExecutor, as_completed

    candidates: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_map_one, i, start, sub): i
                   for i, (start, sub) in enumerate(batches)}
        for fut in as_completed(futures):
            _, batch_candidates, _ = fut.result()
            candidates.extend(batch_candidates)

    log_file.write(json.dumps({"phase": "synth_map.done",
                               "candidates": len(candidates)}) + "\n")
    log_file.flush()
    print(f"  [synth.map] total candidates: {len(candidates)}", flush=True)

    if not candidates:
        raise RuntimeError("synthesize_mapreduce: all map batches failed; no candidates")

    # ── Reduce step 1: LLM consolidates names only (no indexing) ──
    print(f"  [synth.reduce.llm] consolidating {len(candidates)} candidates "
          f"(LLM produces vocab only — no partition)", flush=True)
    log_file.write(json.dumps({"phase": "synth_reduce_llm.start",
                               "candidates": len(candidates)}) + "\n")
    log_file.flush()

    reduce_model = reduce_model_factory().with_structured_output(ConsolidateOutput)
    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", reduce_system_prompt),
        ("user", CONSOLIDATE_USER),
    ])
    cand_block = "\n".join(
        f"- {c['name']}: {c['definition']} "
        f"(batch {c['batch_idx']}, supports {len(c['theme_indices'])} themes)"
        for c in candidates
    )
    t0 = time.perf_counter()
    messages = reduce_prompt.invoke({
        "n_candidates": len(candidates),
        "candidates": cand_block,
    })
    result = invoke_with_retry(reduce_model, messages, retries=3, caller="synth_reduce_llm")
    llm_elapsed = time.perf_counter() - t0
    if result is None:
        raise RuntimeError("synthesize_mapreduce: reduce LLM returned None after retries")

    print(f"  [synth.reduce.llm] {len(result.vocab)} final tags ({llm_elapsed:.1f}s)",
          flush=True)
    log_file.write(json.dumps({"phase": "synth_reduce_llm.done",
                               "final_tags": len(result.vocab),
                               "elapsed_sec": round(llm_elapsed, 2),
                               "notes": result.notes}) + "\n")
    log_file.flush()

    # ── Reduce step 2: embedding-based assignment (deterministic) ──
    # For each original candidate, find the closest final tag by cosine
    # similarity of (name + definition) embeddings. Aggregate theme_indices.
    from .similarity import _embed, cosine

    print(f"  [synth.reduce.embed] assigning {len(candidates)} candidates "
          f"to {len(result.vocab)} final tags via embeddings", flush=True)
    t1 = time.perf_counter()

    def _tag_text(name: str, definition: str) -> str:
        return f"{name}: {definition}"

    final_vocab: list[dict] = [
        {"name": t.name, "definition": t.definition, "theme_indices": set()}
        for t in result.vocab
    ]
    final_embeds = [_embed(_tag_text(t.name, t.definition)) for t in result.vocab]

    assignment_log: list[dict] = []
    for cand_idx, c in enumerate(candidates):
        c_vec = _embed(_tag_text(c["name"], c["definition"]))
        sims = [cosine(c_vec, fe) for fe in final_embeds]
        best = max(range(len(sims)), key=lambda i: sims[i])
        final_vocab[best]["theme_indices"].update(c["theme_indices"])
        assignment_log.append({
            "candidate": c["name"], "candidate_idx": cand_idx,
            "assigned_to": result.vocab[best].name,
            "similarity": round(sims[best], 4),
        })

    # Convert sets → sorted lists for JSON-friendliness
    for v in final_vocab:
        v["theme_indices"] = sorted(v["theme_indices"])

    embed_elapsed = time.perf_counter() - t1
    print(f"  [synth.reduce.embed] {len(candidates)} assignments done "
          f"({embed_elapsed:.1f}s)", flush=True)
    log_file.write(json.dumps({"phase": "synth_reduce_embed.done",
                               "final_tags": len(final_vocab),
                               "candidates_assigned": len(assignment_log),
                               "min_similarity": round(min(a["similarity"] for a in assignment_log), 4),
                               "elapsed_sec": round(embed_elapsed, 2)}) + "\n")
    log_file.flush()

    # Inline a sample of low-similarity assignments for forensics
    suspect = sorted(assignment_log, key=lambda a: a["similarity"])[:10]
    log_file.write(json.dumps({"phase": "synth_reduce_embed.low_similarity_sample",
                               "samples": suspect}) + "\n")
    log_file.flush()

    return {"vocab": final_vocab, "notes": result.notes}


# ── Step 2 (cluster-first variant, BERTopic-style) ─────────────────────────────


class ClusterNameOutput(BaseModel):
    name: str = Field(description="snake_case, 1-2 words; specific pattern, not umbrella")
    definition: str = Field(description="one sentence describing the lesson/pattern")


CLUSTER_NAME_SYSTEM = """Given a cluster of theme descriptions sharing one underlying engineering pattern, name that pattern.

OUTPUT: one tag with name (1-2 words snake_case) + definition (one sentence).

RULES:
1. Tag = a specific recurring pattern, technique, or anti-pattern that a practitioner would recognize
2. NEVER use product/library names (langgraph, mysql, sqlite, mcp, codex, etc.)
3. Capture what these themes have IN COMMON — not the broadest category they belong to

FORBIDDEN umbrella names (refuse them):
- error_handling, error_management, exception_handling
- state_management, data_management, data_handling, data_integrity
- system_design, software_design, system_architecture
- code_quality, engineering_practices, best_practices, robust_systems
- testing, observability, security, performance (alone)

EXAMPLE:
Themes:
  - Concurrent agents writing same checkpoint without isolation
  - Two simultaneous saves overwrite each other without file lock
  - Multi-process state mutation creates lost updates
GOOD: state_isolation: concurrent access to shared mutable state without isolation guards
BAD:  state_management (too broad), concurrency (too broad), data_integrity (umbrella)"""


CLUSTER_NAME_USER = """Cluster of {n} themes sharing one pattern:

{themes}

Output one tag name + definition that names what these have in common."""


def synthesize_via_clustering(
    model_factory,
    themes: list[dict],
    log_file,
    *,
    n_topics: int = 50,
    concurrency: int = 10,
    random_seed: int = 42,
) -> dict:
    """BERTopic-style synthesize: embed → KMeans cluster → LLM names each cluster.

    Skip the "LLM finds patterns in 696 items" task entirely. Embedding does
    the grouping (deterministic, scales), LLM only names each group of
    ~14 thematically-similar items (its strength).

    Trade-off vs map-reduce:
      - Pro: no umbrella collapse, no partition bookkeeping, faster, cheaper
      - Con: cluster boundaries are embedding-driven (no LLM creativity at the
             cluster level — only at naming); n_topics must be chosen up front
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np
    from sklearn.cluster import KMeans

    from .similarity import _embed

    theme_texts = [t["theme"] for t in themes]
    n = len(theme_texts)

    # ── Stage 1: embed all themes ──
    print(f"  [cluster.embed] embedding {n} themes", flush=True)
    log_file.write(json.dumps({"phase": "cluster_embed.start", "n_themes": n}) + "\n")
    log_file.flush()
    t0 = time.perf_counter()
    # _embed is LRU-cached; concurrent embed calls are I/O bound
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        vecs = list(ex.map(_embed, theme_texts))
    embed_elapsed = time.perf_counter() - t0
    print(f"  [cluster.embed] done ({embed_elapsed:.1f}s)", flush=True)

    # ── Stage 2: KMeans cluster ──
    X = np.array(vecs)
    print(f"  [cluster.kmeans] partitioning {n} themes into {n_topics} clusters", flush=True)
    t1 = time.perf_counter()
    km = KMeans(n_clusters=n_topics, random_state=random_seed, n_init=10)
    labels = km.fit_predict(X)
    kmeans_elapsed = time.perf_counter() - t1
    # cluster_id → list of (global_theme_idx, theme_text)
    clusters: dict[int, list[tuple[int, str]]] = {}
    for i, l in enumerate(labels):
        clusters.setdefault(int(l), []).append((i, theme_texts[i]))
    sizes = sorted(len(c) for c in clusters.values())
    print(f"  [cluster.kmeans] done ({kmeans_elapsed:.1f}s). "
          f"cluster sizes: min={sizes[0]}, median={sizes[len(sizes)//2]}, max={sizes[-1]}",
          flush=True)
    log_file.write(json.dumps({"phase": "cluster_kmeans.done",
                               "n_clusters": n_topics,
                               "min_size": sizes[0],
                               "max_size": sizes[-1],
                               "median_size": sizes[len(sizes)//2],
                               "elapsed_sec": round(kmeans_elapsed, 2)}) + "\n")
    log_file.flush()

    # ── Stage 3: LLM names each cluster concurrently ──
    name_model = model_factory().with_structured_output(ClusterNameOutput)
    name_prompt = ChatPromptTemplate.from_messages([
        ("system", CLUSTER_NAME_SYSTEM),
        ("user", CLUSTER_NAME_USER),
    ])

    def _name_cluster(cluster_id: int, members: list[tuple[int, str]]):
        block = "\n".join(f"- {text}" for _, text in members)
        messages = name_prompt.invoke({"n": len(members), "themes": block})
        t = time.perf_counter()
        result = invoke_with_retry(name_model, messages, retries=3,
                                    caller=f"cluster_name_{cluster_id}")
        elapsed = time.perf_counter() - t
        if result is None:
            return cluster_id, None, elapsed
        return cluster_id, result, elapsed

    print(f"  [cluster.name] LLM naming {n_topics} clusters (concurrency={concurrency})",
          flush=True)
    t2 = time.perf_counter()
    cluster_names: dict[int, ClusterNameOutput] = {}
    failed = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_name_cluster, cid, members): cid
                   for cid, members in clusters.items()}
        for fut in as_completed(futures):
            cid, res, _ = fut.result()
            if res is None:
                failed.append(cid)
                print(f"  [cluster.name] cluster {cid} FAILED (all retries)", flush=True)
            else:
                cluster_names[cid] = res
    name_elapsed = time.perf_counter() - t2
    print(f"  [cluster.name] {len(cluster_names)}/{n_topics} named "
          f"({name_elapsed:.1f}s, {len(failed)} failures)", flush=True)
    log_file.write(json.dumps({"phase": "cluster_name.done",
                               "named": len(cluster_names),
                               "failed_clusters": failed,
                               "elapsed_sec": round(name_elapsed, 2)}) + "\n")
    log_file.flush()

    # ── Assemble final vocab ──
    final_vocab = []
    for cid, members in clusters.items():
        if cid not in cluster_names:
            continue  # skip failed clusters; their themes will appear as missing in reverse_check
        named = cluster_names[cid]
        final_vocab.append({
            "name": named.name,
            "definition": named.definition,
            "theme_indices": sorted(i for i, _ in members),
        })

    notes = (f"BERTopic-style: embedded {n} themes, KMeans k={n_topics} "
             f"(sizes min/med/max={sizes[0]}/{sizes[len(sizes)//2]}/{sizes[-1]}), "
             f"{len(cluster_names)}/{n_topics} clusters named, {len(failed)} failures")
    return {"vocab": final_vocab, "notes": notes}


def run(db_path: Path, output_dir: Path, batch_size: int, skip_distill: bool) -> None:
    settings = Settings()

    def model_factory():
        return _chat_model(settings)

    output_dir.mkdir(parents=True, exist_ok=True)
    themes_path = output_dir / "tag_extraction_v2_themes.json"
    vocab_path = output_dir / "tag_extraction_v2_vocab.json"
    log_path = output_dir / "tag_extraction_v2_log.jsonl"

    log_file = log_path.open("a" if skip_distill else "w", encoding="utf-8")

    if skip_distill and themes_path.exists():
        themes = json.loads(themes_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(themes)} cached themes from {themes_path}", flush=True)
    else:
        records = load_records(db_path)
        print(f"Loaded {len(records)} records from {db_path}", flush=True)

        # Step 1: Distill
        print(f"\n=== Step 1: Distill themes ({len(records)} records, batch={batch_size}) ===\n", flush=True)
        themes = distill_step(model_factory, records, batch_size, log_file)
        themes_path.write_text(
            json.dumps(themes, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  ✅ {len(themes)} themes → {themes_path}", flush=True)

    # Step 2: Synthesize
    print(f"\n=== Step 2: Synthesize vocab from themes ===\n", flush=True)
    syn_result = synthesize_step(model_factory, themes, log_file)
    vocab_payload = {
        "vocab": syn_result["vocab"],
        "notes": syn_result["notes"],
        "theme_count": len(themes),
    }
    vocab_path.write_text(
        json.dumps(vocab_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  ✅ {len(syn_result['vocab'])} tags → {vocab_path}", flush=True)

    log_file.close()
    print(f"\nLog → {log_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--output-dir", default="outputs", type=Path)
    parser.add_argument("--batch-size", default=30, type=int)
    parser.add_argument("--skip-distill", action="store_true", help="reuse cached themes.json, only run synthesize")
    args = parser.parse_args()

    run(db_path=args.db, output_dir=args.output_dir, batch_size=args.batch_size, skip_distill=args.skip_distill)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


class SynthesizeOutput(BaseModel):
    vocab: list[TagDefinition]
    notes: str = Field(description="brief summary of vocab structure and any difficult cases")


SYNTHESIZE_SYSTEM = """You're given theme descriptions extracted from engineering knowledge records. Cluster these themes into a tag vocabulary.

STRICT NAMING RULES (these are absolute):
1. Each tag name: 1 or 2 words, snake_case (NO 3+ word tags like `early_termination_logic` — use `early_termination`)
2. Tag = abstract pattern/concept/anti-pattern. Examples of GOOD names:
   - silent_failure, state_isolation, schema_evolution, early_termination,
   - resilience_patterns, fallback_strategy, naming_conflict, env_propagation
3. FORBIDDEN: product/library/platform/framework names anywhere in the tag
   - NEVER: langgraph, playwright, mysql, sqlite, mcp, astro, codex, macos, pytest, asyncio, telegram, consul, akshare, npm, vscode
4. Same abstraction level across all tags (pattern-level, not specific-scenario-level)
5. No singleton tags — every tag must group at least 2 themes
6. Merge synonyms aggressively (e.g. "early_termination" and "premature_stop" → pick one)

Goals:
- Coverage: every theme must belong to exactly one tag
- Cluster size targets: each tag should average 10-25 themes. Tags covering fewer than 5 themes should merge into broader categories; tags covering more than 40 themes should split into finer-grained patterns. Aim for a balanced size distribution, not a few mega-clusters.
- Total count: aim for {target_lo}-{target_hi} tags for this corpus of {theme_count} themes. Producing far fewer means each tag is overly broad; producing far more means the vocabulary is over-fragmented. Treat this as a hard guidance, not a preference for 'cleaner over more'.

If a theme doesn't fit any current cluster, you may create a new cluster for it (but think carefully — can it merge with an existing cluster?). Don't create a cluster of one.

Output: for each tag, give name + definition + the 0-based theme indices that belong to it."""


SYNTHESIZE_USER = """Total themes: {theme_count}

Themes (one per line, 0-based indexed):
{themes}

Cluster these into a clean tag vocabulary following the rules. Each theme must belong to exactly one tag."""


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
    target_count_range: tuple[int, int] | None = None,
) -> dict:
    """Returns synthesize raw output (vocab + notes).

    `system_prompt` defaults to the module-level SYNTHESIZE_SYSTEM constant.
    `target_count_range` overrides the auto-derived (lo, hi) range; if None,
    computed from len(themes) via _default_target_count_range.
    """
    synthesize_model = model_factory().with_structured_output(SynthesizeOutput)
    synthesize_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", SYNTHESIZE_USER),
    ])

    theme_texts = [t["theme"] for t in themes]
    target_lo, target_hi = target_count_range or _default_target_count_range(len(theme_texts))

    t0 = time.perf_counter()
    messages = synthesize_prompt.invoke({
        "theme_count": len(theme_texts),
        "themes": format_themes_for_synthesize(theme_texts),
        "target_lo": target_lo,
        "target_hi": target_hi,
    })
    result: SynthesizeOutput = synthesize_model.invoke(messages)
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

"""Incremental tag vocabulary extraction via LLM.

Reads source_knowledge_records, shuffles them, processes in batches.
Each batch: LLM sees cumulative tag list + the batch records, proposes new tags.
Output: a flat tag vocabulary (no record-tag assignments).
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model


class TagDefinition(BaseModel):
    name: str = Field(description="snake_case name, <= 3 words, describes a scenario/pattern/concept")
    definition: str = Field(description="one sentence definition")


class BatchOutput(BaseModel):
    new_tags: list[TagDefinition] = Field(description="new tags to add. empty if existing tags cover this batch.")
    rationale: str = Field(description="one sentence: what new themes this batch introduced")


SYSTEM_PROMPT = """You are building a knowledge tag vocabulary from a stream of engineering knowledge records.

Each turn you receive:
  - the CUMULATIVE TAG LIST built so far (with definitions)
  - a NEW BATCH of records (title + insight)

Your task: identify any new themes/scenarios/patterns this batch introduces that are NOT already covered by the existing tags. Propose new tags for them.

Tag naming rules (strict):
  1. Describes a scenario, pattern, or concept (e.g. `silent_failure_detection`, `state_consistency`, `version_compatibility`)
  2. NEVER use product, library, platform, or framework names (NOT `langgraph`, `mysql`, `playwright`, `astro`, `pytest`, `sqlite`, `mcp`, `vscode`, `macos`)
  3. snake_case, at most 3 words
  4. If existing tags already cover the batch's themes, return an empty new_tags list

When deciding whether to add a new tag:
  - If the theme already exists with a different name → DO NOT add (reuse mentally)
  - If the theme is a subset of an existing tag → DO NOT add
  - Only add when a genuinely new concept/pattern appears that needs a new label

Return concise definitions (one sentence each)."""


USER_PROMPT = """## Cumulative tag list ({tag_count} tags)

{tag_list}

## New batch ({batch_size} records)

{records}

Identify new themes this batch introduces (if any) and propose new tags following the naming rules."""


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


def format_tag_list(tags: list[TagDefinition]) -> str:
    if not tags:
        return "(empty - this is the first batch)"
    return "\n".join(f"- {t.name}: {t.definition}" for t in tags)


def format_records(records: list[dict]) -> str:
    lines = []
    for idx, r in enumerate(records, 1):
        insight = r["insight"][:300]
        lines.append(f"{idx}. [{r['record_id'][:12]}] {r['title']}\n   {insight}")
    return "\n\n".join(lines)


def run(
    db_path: Path,
    output_dir: Path,
    batch_size: int,
    seed: int,
    max_batches: int | None,
) -> None:
    settings = Settings()
    model = _chat_model(settings).with_structured_output(BatchOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    records = load_records(db_path)
    print(f"Loaded {len(records)} records from {db_path}", flush=True)

    rng = random.Random(seed)
    rng.shuffle(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "tag_extraction_vocab.json"
    log_path = output_dir / "tag_extraction_log.jsonl"

    cumulative_tags: list[TagDefinition] = []
    tag_names_seen: set[str] = set()

    total_batches = (len(records) + batch_size - 1) // batch_size
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    log_file = log_path.open("w", encoding="utf-8")

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(records))
        batch_records = records[start:end]

        t0 = time.perf_counter()
        try:
            messages = prompt.invoke({
                "tag_count": len(cumulative_tags),
                "tag_list": format_tag_list(cumulative_tags),
                "batch_size": len(batch_records),
                "records": format_records(batch_records),
            })
            result: BatchOutput = model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - want batch-level failure isolation
            print(f"  batch {batch_idx + 1}/{total_batches} FAILED: {exc}", flush=True)
            log_entry = {
                "batch": batch_idx + 1,
                "records_count": len(batch_records),
                "error": str(exc),
            }
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            log_file.flush()
            continue

        elapsed = time.perf_counter() - t0

        # dedupe by name
        net_new: list[TagDefinition] = []
        for t in result.new_tags:
            normalized = t.name.strip().lower()
            if normalized in tag_names_seen:
                continue
            tag_names_seen.add(normalized)
            net_new.append(TagDefinition(name=normalized, definition=t.definition.strip()))

        cumulative_tags.extend(net_new)

        log_entry = {
            "batch": batch_idx + 1,
            "records_count": len(batch_records),
            "new_tag_count": len(net_new),
            "cumulative_tag_count": len(cumulative_tags),
            "elapsed_sec": round(elapsed, 2),
            "rationale": result.rationale,
            "new_tags": [t.model_dump() for t in net_new],
        }
        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        log_file.flush()

        new_names = ", ".join(t.name for t in net_new) if net_new else "(none)"
        print(
            f"  batch {batch_idx + 1}/{total_batches}  records={len(batch_records)}  "
            f"+{len(net_new)} new  total={len(cumulative_tags)}  ({elapsed:.1f}s)  → {new_names}",
            flush=True,
        )

        # save vocab after each batch (resumable progress)
        vocab_path.write_text(
            json.dumps([t.model_dump() for t in cumulative_tags], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    log_file.close()

    print(f"\n✅ Done. Final vocab: {len(cumulative_tags)} tags → {vocab_path}", flush=True)
    print(f"   Per-batch log → {log_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--output-dir", default="outputs", type=Path)
    parser.add_argument("--batch-size", default=30, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-batches", default=None, type=int, help="limit for smoke testing")
    args = parser.parse_args()

    run(
        db_path=args.db,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        max_batches=args.max_batches,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

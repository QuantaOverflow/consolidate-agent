"""Propose new tags from missing records.

Input: assignments file (records where missing=true)
       + current vocab (so LLM can avoid re-creating existing concepts)

Pipeline:
  1. Load missing records (with title+insight from db)
  2. Distill: each record → theme (no tag yet)
  3. Synthesize with vocab context: themes → vocab patch
     - new_tags: genuinely missing concepts
     - reassignments: themes that actually fit an existing tag (LLM over-flagged in reverse check)
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

# Reuse distill primitives from bootstrap pipeline
from ..bootstrap import (
    DISTILL_SYSTEM,
    DISTILL_USER,
    DistillBatchOutput,
    format_records_for_distill,
)


# ── Synthesize-with-vocab schema ───────────────────────────────────────────────

class NewTagCandidate(BaseModel):
    name: str = Field(description="snake_case, 1-2 words, abstract pattern/concept")
    definition: str = Field(description="one sentence")
    support_count: int = Field(description="how many themes this tag covers (must be >= 2)")


class SynthesisWithVocabOutput(BaseModel):
    new_tags: list[NewTagCandidate] = Field(description="genuinely missing concepts (each must cover >=2 themes)")
    notes: str = Field(description="brief analysis. mention themes reassigned to existing vocab tags.")


SYNTHESIZE_VOCAB_SYSTEM = """You're analyzing themes from records that were flagged as "no fit" by an earlier tagging pass. Some are genuinely new concepts; some were over-flagged (they actually fit an existing tag the previous pass missed).

Existing vocab is provided. Your task:

For each theme, decide:
   - Does it fit any EXISTING vocab tag? → just don't include it in new_tags, mention reassignment in notes
   - Is it a genuinely new concept needing a new tag? → group with similar themes into a new_tag

New tag rules (same strict naming as the existing vocab):
   - snake_case, 1-2 words
   - abstract pattern/concept (not product/library names)
   - Each new tag must have >=2 theme support (no singletons)
   - Don't duplicate existing concepts under new names

Be conservative: when in doubt, do NOT create new tag. Better to leave a theme reassignable than to create singletons.

Output:
  - new_tags: only genuinely missing concepts with 2+ supporting themes
  - notes: brief analysis. If many themes were reassigned to existing vocab, list them here (theme summary → existing tag)."""


SYNTHESIZE_VOCAB_USER = """## Existing vocab ({vocab_count} tags)

{vocab}

## Themes from "missing" records ({theme_count} themes)

{themes}

For each theme decide: reassign to existing OR group into new tag. Be conservative."""


def load_vocab(vocab_path: Path) -> list[dict]:
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    return data["vocab"]


def load_missing_records(assignments_path: Path, db_path: Path) -> list[dict]:
    """Load records flagged as missing=true, fetching title+insight from db."""
    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    missing_ids = [a["record_id"] for a in assignments if a.get("missing")]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(missing_ids))
    rows = conn.execute(
        f"SELECT record_id, title, insight FROM source_knowledge_records WHERE record_id IN ({placeholders})",
        missing_ids,
    ).fetchall()
    conn.close()

    return [{"record_id": r["record_id"], "title": r["title"], "insight": r["insight"]} for r in rows]


def format_vocab(vocab: list[dict]) -> str:
    return "\n".join(f"- {t['name']}: {t['definition']}" for t in vocab)


def format_themes(themes: list[str]) -> str:
    return "\n".join(f"[{i}] {t}" for i, t in enumerate(themes))


def distill_records(model_factory, records: list[dict], batch_size: int) -> list[dict]:
    """Distill records into themes (no tags)."""
    distill_model = model_factory().with_structured_output(DistillBatchOutput)
    distill_prompt = ChatPromptTemplate.from_messages([
        ("system", DISTILL_SYSTEM),
        ("user", DISTILL_USER),
    ])

    total_batches = (len(records) + batch_size - 1) // batch_size
    all_themes: list[dict] = []

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(records))
        batch_records = records[start:end]

        t0 = time.perf_counter()
        messages = distill_prompt.invoke({
            "batch_size": len(batch_records),
            "records": format_records_for_distill(batch_records),
        })
        result: DistillBatchOutput = distill_model.invoke(messages)
        elapsed = time.perf_counter() - t0

        themes_by_idx = {t.record_idx: t.theme for t in result.themes}
        for rec_idx, rec in enumerate(batch_records, 1):
            all_themes.append({
                "record_id": rec["record_id"],
                "title": rec["title"],
                "theme": themes_by_idx.get(rec_idx, "").strip(),
            })

        print(f"  distill batch {batch_idx + 1}/{total_batches}  records={len(batch_records)}  ({elapsed:.1f}s)", flush=True)

    return all_themes


def synthesize_with_vocab(model_factory, themes: list[dict], vocab: list[dict]) -> SynthesisWithVocabOutput:
    syn_model = model_factory().with_structured_output(SynthesisWithVocabOutput)
    syn_prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIZE_VOCAB_SYSTEM),
        ("user", SYNTHESIZE_VOCAB_USER),
    ])

    theme_texts = [t["theme"] for t in themes]

    t0 = time.perf_counter()
    messages = syn_prompt.invoke({
        "vocab_count": len(vocab),
        "vocab": format_vocab(vocab),
        "theme_count": len(theme_texts),
        "themes": format_themes(theme_texts),
    })
    result = syn_model.invoke(messages)
    elapsed = time.perf_counter() - t0

    print(f"  synthesize  themes={len(theme_texts)}  →  new_tags={len(result.new_tags) if result else 0}  ({elapsed:.1f}s)", flush=True)

    if result is None:
        raise ValueError("synthesize returned None")
    return result


def propose_new_fn(
    vocab: list[dict],
    assignments: list[dict],
    focus: str = "",
    *,
    db_path: Path | None = None,
    batch_size: int = 30,
):
    """In-memory propose_new for agent loop.

    Pipeline: missing records → distill → synthesize-with-vocab → NewTagProposal list.
    focus is appended to synthesize prompt to guide candidate generation.
    """
    from ..apply import NewTagProposal as _NewTagProposal

    settings = Settings()
    def model_factory():
        return _chat_model(settings)

    # Load missing records' titles + insights from db
    missing_ids = [a["record_id"] for a in assignments if a.get("missing")]
    if not missing_ids or db_path is None:
        return []
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(missing_ids))
    rows = conn.execute(
        f"SELECT record_id, title, insight FROM source_knowledge_records WHERE record_id IN ({placeholders})",
        missing_ids,
    ).fetchall()
    conn.close()
    missing = [{"record_id": r["record_id"], "title": r["title"], "insight": r["insight"]} for r in rows]

    if not missing:
        return []

    # Step 1: distill (in-memory, no cache)
    themes = distill_records(model_factory, missing, batch_size)

    # Step 2: synthesize with focus hint
    syn_model = model_factory().with_structured_output(SynthesisWithVocabOutput)
    sys_prompt_with_focus = SYNTHESIZE_VOCAB_SYSTEM
    if focus:
        sys_prompt_with_focus = SYNTHESIZE_VOCAB_SYSTEM + f"\n\nFOCUS HINT (agent guidance): {focus}"
    syn_prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt_with_focus),
        ("user", SYNTHESIZE_VOCAB_USER),
    ])
    theme_texts = [t["theme"] for t in themes]
    messages = syn_prompt.invoke({
        "vocab_count": len(vocab),
        "vocab": format_vocab(vocab),
        "theme_count": len(theme_texts),
        "themes": format_themes(theme_texts),
    })
    result = syn_model.invoke(messages)
    if result is None:
        result = syn_model.invoke(messages)  # retry
    if result is None:
        return []

    # validate + dedupe against existing vocab
    vocab_names = {t["name"] for t in vocab}
    proposals: list = []
    seen: set[str] = set()
    for cand in result.new_tags:
        if not cand.name or cand.name in vocab_names or cand.name in seen:
            continue
        if cand.support_count < 2:
            continue
        seen.add(cand.name)
        proposals.append(_NewTagProposal(name=cand.name, definition=cand.definition))

    return proposals


def run(
    assignments_path: Path,
    db_path: Path,
    vocab_path: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    settings = Settings()

    def model_factory():
        return _chat_model(settings)

    vocab = load_vocab(vocab_path)
    missing = load_missing_records(assignments_path, db_path)
    print(f"Missing records: {len(missing)}", flush=True)
    print(f"Vocab: {len(vocab)} tags", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Distill missing records (or load cached)
    themes_path = output_dir / "propose_new_themes.json"
    if themes_path.exists():
        themes = json.loads(themes_path.read_text(encoding="utf-8"))
        print(f"\nLoaded {len(themes)} cached themes from {themes_path}\n", flush=True)
    else:
        print(f"\n=== Step 1: Distill missing records ===\n", flush=True)
        themes = distill_records(model_factory, missing, batch_size)
        themes_path.write_text(json.dumps(themes, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  ✅ {len(themes)} themes → {themes_path}", flush=True)

    # Step 2: Synthesize with vocab context
    print(f"\n=== Step 2: Synthesize against vocab ===\n", flush=True)
    syn = synthesize_with_vocab(model_factory, themes, vocab)

    output_path = output_dir / "propose_new_vocab_patch.json"
    output_payload = {
        "vocab_size": len(vocab),
        "missing_count": len(missing),
        "new_tags": [t.model_dump() for t in syn.new_tags],
        "notes": syn.notes,
    }
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  ✅ vocab patch → {output_path}", flush=True)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Missing themes analyzed: {len(themes)}")
    print(f"  → Proposed new tags:  {len(syn.new_tags)}")
    covered = sum(t.support_count for t in syn.new_tags)
    print(f"  → Themes covered by new tags: {covered}")
    print(f"  → Themes reassigned to existing (see notes): {len(themes) - covered}")

    print(f"\n=== Proposed new tags ===")
    for t in syn.new_tags:
        print(f"  • {t.name}  (support: {t.support_count} themes)")
        print(f"    def: {t.definition}")

    print(f"\n=== Notes ===\n{syn.notes}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", default="outputs/full_assignment/reverse_check_assignments.json", type=Path)
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--vocab", default="outputs/tag_extraction_v2_vocab_v1.1.json", type=Path)
    parser.add_argument("--output-dir", default="outputs/full_assignment", type=Path)
    parser.add_argument("--batch-size", default=30, type=int)
    args = parser.parse_args()

    run(
        assignments_path=args.assignments,
        db_path=args.db,
        vocab_path=args.vocab,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

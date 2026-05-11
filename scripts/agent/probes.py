"""Probe tools — deterministic data access for the diagnostic agent.

Each probe returns structured info the LLM can use to verify or refute
raw diagnostics signals. No LLM calls.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_record_details(db_path: Path, record_ids: list[str]) -> dict[str, dict]:
    if not record_ids:
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(record_ids))
    rows = conn.execute(
        f"SELECT record_id, title, insight FROM source_knowledge_records WHERE record_id IN ({placeholders})",
        record_ids,
    ).fetchall()
    conn.close()
    return {r["record_id"]: {"title": r["title"], "insight": r["insight"]} for r in rows}


def inspect_missing_records(assignments: list[dict], db_path: Path, n: int = 10) -> list[dict]:
    """Return n missing records with title + insight excerpt + LLM's missing_concept guess."""
    missing = [a for a in assignments if a.get("missing")][:n]
    ids = [a["record_id"] for a in missing]
    details = _load_record_details(db_path, ids)
    return [
        {
            "record_id": a["record_id"],
            "title": details.get(a["record_id"], {}).get("title", a.get("title", "")),
            "insight": details.get(a["record_id"], {}).get("insight", "")[:240],
            "llm_missing_concept": a.get("missing_concept", ""),
        }
        for a in missing
    ]


def inspect_cooccur_pair(
    assignments: list[dict], db_path: Path, tag_a: str, tag_b: str, n: int = 5
) -> dict:
    """Return records that co-occur on both tags, with title + insight."""
    shared = []
    for a in assignments:
        if a.get("missing"):
            continue
        names = {t["name"] for t in a.get("selected_tags", [])}
        if tag_a in names and tag_b in names:
            shared.append(a["record_id"])
    sample = shared[:n]
    details = _load_record_details(db_path, sample)
    return {
        "pair": [tag_a, tag_b],
        "total_cooccur": len(shared),
        "sample_records": [
            {
                "record_id": rid,
                "title": details.get(rid, {}).get("title", ""),
                "insight": details.get(rid, {}).get("insight", "")[:240],
            }
            for rid in sample
        ],
    }


def inspect_tag(
    vocab: list[dict], assignments: list[dict], db_path: Path, name: str, n: int = 8
) -> dict:
    """Return tag def + sample records + co-tag distribution."""
    tag_def = next((t for t in vocab if t["name"] == name), None)
    record_ids = [
        a["record_id"] for a in assignments
        if not a.get("missing") and any(t["name"] == name for t in a.get("selected_tags", []))
    ]
    co_tags: dict[str, int] = defaultdict(int)
    for a in assignments:
        if a.get("missing"):
            continue
        names = {t["name"] for t in a.get("selected_tags", [])}
        if name in names:
            for n2 in names - {name}:
                co_tags[n2] += 1
    top_co = sorted(co_tags.items(), key=lambda x: -x[1])[:5]
    sample = record_ids[:n]
    details = _load_record_details(db_path, sample)
    return {
        "name": name,
        "definition": tag_def["definition"] if tag_def else "(tag not in vocab)",
        "usage_count": len(record_ids),
        "top_co_occurring_tags": [{"tag": t, "count": c} for t, c in top_co],
        "sample_records": [
            {
                "record_id": rid,
                "title": details.get(rid, {}).get("title", ""),
                "insight": details.get(rid, {}).get("insight", "")[:200],
            }
            for rid in sample
        ],
    }


def compare_tag_records(
    vocab: list[dict], assignments: list[dict], db_path: Path, tag_a: str, tag_b: str
) -> dict:
    """Compare two tags' record sets — overlap, unique counts, sample of each side."""
    a_records: set[str] = set()
    b_records: set[str] = set()
    for a in assignments:
        if a.get("missing"):
            continue
        names = {t["name"] for t in a.get("selected_tags", [])}
        if tag_a in names:
            a_records.add(a["record_id"])
        if tag_b in names:
            b_records.add(a["record_id"])

    overlap = a_records & b_records
    only_a = a_records - b_records
    only_b = b_records - a_records

    sample_overlap = _load_record_details(db_path, list(overlap)[:3])
    sample_only_a = _load_record_details(db_path, list(only_a)[:3])
    sample_only_b = _load_record_details(db_path, list(only_b)[:3])

    def_a = next((t["definition"] for t in vocab if t["name"] == tag_a), "(not in vocab)")
    def_b = next((t["definition"] for t in vocab if t["name"] == tag_b), "(not in vocab)")

    return {
        "tag_a": {"name": tag_a, "definition": def_a, "count": len(a_records)},
        "tag_b": {"name": tag_b, "definition": def_b, "count": len(b_records)},
        "overlap_count": len(overlap),
        "only_a_count": len(only_a),
        "only_b_count": len(only_b),
        "overlap_ratio_of_smaller": len(overlap) / min(len(a_records), len(b_records)) if a_records and b_records else 0,
        "sample_overlap": [
            {"title": d["title"], "insight": d["insight"][:160]}
            for d in sample_overlap.values()
        ],
        "sample_only_a": [
            {"title": d["title"], "insight": d["insight"][:160]}
            for d in sample_only_a.values()
        ],
        "sample_only_b": [
            {"title": d["title"], "insight": d["insight"][:160]}
            for d in sample_only_b.values()
        ],
    }


# ── Dispatch table ───────────────────────────────────────────────────────────


PROBES = {
    "inspect_missing_records": inspect_missing_records,
    "inspect_cooccur_pair": inspect_cooccur_pair,
    "inspect_tag": inspect_tag,
    "compare_tag_records": compare_tag_records,
}


def run_probe(probe_call: str, vocab: list[dict], assignments: list[dict], db_path: Path) -> dict:
    """Parse probe call string like 'inspect_tag(name=foo)' and dispatch."""
    import re
    m = re.match(r"(\w+)\((.*)\)", probe_call.strip())
    if not m:
        return {"error": f"unparseable probe call: {probe_call}"}
    func_name = m.group(1)
    args_str = m.group(2).strip()
    if func_name not in PROBES:
        return {"error": f"unknown probe: {func_name}"}

    # parse args (simple kw=val parsing)
    kwargs: dict[str, Any] = {}
    if args_str:
        for part in re.split(r",\s*(?=[a-z_]+\s*=)", args_str):
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"\'')
                if v.isdigit():
                    kwargs[k] = int(v)
                else:
                    kwargs[k] = v

    func = PROBES[func_name]
    # inject vocab/assignments/db_path based on signature
    if func_name == "inspect_missing_records":
        return func(assignments, db_path, **kwargs)
    elif func_name == "inspect_cooccur_pair":
        return func(assignments, db_path, **kwargs)
    elif func_name == "inspect_tag":
        return func(vocab, assignments, db_path, **kwargs)
    elif func_name == "compare_tag_records":
        return func(vocab, assignments, db_path, **kwargs)
    return {"error": "unreachable"}

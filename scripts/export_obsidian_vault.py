#!/usr/bin/env python3
import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

UNSAFE_CHARS = {":", "/", "\\"}

GRAPH_JSON = {
    "collapse-filter": True,
    "search": "",
    "showTags": False,
    "showAttachments": False,
    "hideUnresolved": False,
    "showOrphans": True,
    "collapse-color-groups": False,
    "colorGroups": [
        {"query": "path:tags/", "color": {"a": 1, "rgb": 10184099}},
        {"query": "path:records/orphan/", "color": {"a": 1, "rgb": 15171644}},
        {"query": "path:records/", "color": {"a": 1, "rgb": 3447003}},
    ],
    "collapse-display": True,
    "showArrow": False,
    "textFadeMultiplier": 0,
    "nodeSizeMultiplier": 1,
    "lineSizeMultiplier": 1,
    "collapse-forces": True,
    "centerStrength": 0.518,
    "repelStrength": 10,
    "linkStrength": 1,
    "linkDistance": 250,
    "scale": 1,
    "close": False,
}


def safe_filename(name: str) -> str:
    result = name
    for ch in UNSAFE_CHARS:
        if ch in result:
            print(f"WARNING: unsafe char '{ch}' in name '{name}', replacing with '_'")
            result = result.replace(ch, "_")
    return result


def safe_yaml_str(s: str) -> str:
    """Return a JSON-style quoted string safe for YAML frontmatter."""
    return json.dumps(s.replace("\n", " "), ensure_ascii=False)


def safe_wikilink_display(title: str) -> str:
    """Replace chars that break wikilink display text."""
    for ch in ("|", "[", "]"):
        title = title.replace(ch, "-")
    return title


def load_knowledge_db(db_path: Path) -> dict[str, dict]:
    """Load record bodies from knowledge.db. Returns empty dict if file missing."""
    if not db_path.exists():
        print(f"WARNING: knowledge-db not found at {db_path}, skipping body enrich")
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT record_id, insight, applicability, scope, evidence_count, created_at"
            " FROM source_knowledge_records"
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    result = {}
    for record_id, insight, applicability, scope, evidence_count, created_at in rows:
        result[record_id] = {
            "insight": insight or "",
            "applicability": applicability or "",
            "scope": scope or "",
            "evidence_count": evidence_count,
            "created_at": created_at or "",
        }
    print(f"loaded {len(result)} record bodies from knowledge.db")
    return result


def write_tag_file(tags_dir: Path, tag: dict, record_refs: list[tuple[str, str]]) -> None:
    name = safe_filename(tag["name"])
    definition = tag["definition"]
    sorted_refs = sorted(record_refs, key=lambda x: x[1])
    n = len(sorted_refs)

    lines = [
        "---",
        f"type: tag",
        f"definition: {safe_yaml_str(definition)}",
        f"records_count: {n}",
        "---",
        "",
        f"# {name}",
        "",
        f"> {definition}",
        "",
        f"## Records ({n})",
        "",
    ]
    for rec_id, rec_title in sorted_refs:
        display = safe_wikilink_display(rec_title)
        lines.append(f"- [[{rec_id}|{display}]]")

    (tags_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def write_record_file(
    records_dir: Path,
    orphan_dir: Path,
    assignment: dict,
    body_map: dict[str, dict],
) -> None:
    rec_id = safe_filename(assignment["record_id"])
    title = assignment["title"]
    missing = assignment.get("missing", False)

    body = body_map.get(assignment["record_id"])
    if body is None and body_map:
        print(f"WARNING: record_id {assignment['record_id']} not found in knowledge.db, writing without body")

    insight = body["insight"].strip() if body else ""
    applicability = body["applicability"].strip() if body else ""
    scope = body["scope"] if body else ""
    evidence_count = body["evidence_count"] if body else 0
    created_at = body["created_at"] if body else ""

    insight_text = insight if insight else "*(empty)*"
    applicability_text = applicability if applicability else "*(empty)*"

    if missing:
        missing_concept = assignment.get("missing_concept") or "(none specified)"
        reason = (assignment.get("reason") or "")[:500]
        lines = [
            "---",
            f"type: record",
            f"title: {safe_yaml_str(title)}",
            f"scope: {scope}",
            f"evidence_count: {evidence_count}",
            f'created_at: "{created_at}"',
            f"tags_count: 0",
            f"status: orphan",
            "---",
            "",
            f"# {title}",
            "",
            f"> ⚠️ Orphan — no matching vocab tag.",
            "",
            "## Insight",
            "",
            insight_text,
            "",
            "## Applicability",
            "",
            applicability_text,
            "",
            f"**Missing concept**: {missing_concept}",
            "",
            f"**Reason**: {reason}",
        ]
        (orphan_dir / f"{rec_id}.md").write_text("\n".join(lines), encoding="utf-8")
    else:
        selected_tags = sorted(assignment["selected_tags"], key=lambda t: t["name"])
        n = len(selected_tags)
        lines = [
            "---",
            f"type: record",
            f"title: {safe_yaml_str(title)}",
            f"scope: {scope}",
            f"evidence_count: {evidence_count}",
            f'created_at: "{created_at}"',
            f"tags_count: {n}",
            "---",
            "",
            f"# {title}",
            "",
            "## Insight",
            "",
            insight_text,
            "",
            "## Applicability",
            "",
            applicability_text,
            "",
            f"## Tags ({n})",
            "",
        ]
        for t in selected_tags:
            tag_name = safe_filename(t["name"])
            confidence = t.get("confidence", "")
            lines.append(f"- [[{tag_name}]] (confidence: {confidence})")

        (records_dir / f"{rec_id}.md").write_text("\n".join(lines), encoding="utf-8")


def write_graph_config(out_dir: Path) -> None:
    obsidian_dir = out_dir / ".obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    (obsidian_dir / "graph.json").write_text(
        json.dumps(GRAPH_JSON, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_readme(out_dir: Path, network_path: str, vocab: list, assignments: list) -> None:
    tagged = [a for a in assignments if not a.get("missing")]
    orphan = [a for a in assignments if a.get("missing")]
    total_edges = sum(len(a["selected_tags"]) for a in assignments)
    mean_tags = total_edges / len(tagged) if tagged else 0.0
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        "# Tag Network Vault",
        "",
        f"Generated from `{network_path}` at {ts}.",
        "",
        "## Stats",
        "",
        f"- Tags: {len(vocab)}",
        f"- Records: {len(assignments)}",
        f"  - Tagged: {len(tagged)}",
        f"  - Orphan: {len(orphan)}",
        f"- Total tag-record edges: {total_edges}",
        f"- Mean tags per record (tagged only): {mean_tags:.2f}",
        "",
        "## How to view in Obsidian",
        "",
        "1. Open Obsidian → \"Open folder as vault\" → select this directory",
        "2. Graph view (Cmd-G / Ctrl-G) — colors are pre-configured:",
        "   - **Purple**: tags",
        "   - **Blue**: tagged records",
        "   - **Red**: orphan records (in records/orphan/)",
        "3. If colors don't show, open Graph settings → 颜色组 (Color groups) → toggle on",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export network.json to Obsidian vault")
    parser.add_argument("--network", default="outputs/network.json")
    parser.add_argument("--knowledge-db", default="outputs/knowledge.db")
    parser.add_argument("--out", default="outputs/obsidian_vault")
    parser.add_argument("--force", action="store_true", help="Skip confirmation when clearing output dir")
    args = parser.parse_args()

    network_path = Path(args.network)
    out_dir = Path(args.out)
    db_path = Path(args.knowledge_db)

    if not network_path.exists():
        print(f"ERROR: network file not found: {network_path}", file=sys.stderr)
        sys.exit(1)

    if out_dir.exists():
        if not args.force:
            ans = input(f"Output dir '{out_dir}' exists. Clear and overwrite? [y/N] ").strip().lower()
            if ans != "y":
                print("Aborted.")
                sys.exit(0)
        shutil.rmtree(out_dir)

    tags_dir = out_dir / "tags"
    records_dir = out_dir / "records"
    orphan_dir = records_dir / "orphan"
    tags_dir.mkdir(parents=True)
    records_dir.mkdir(parents=True)
    orphan_dir.mkdir(parents=True)

    body_map = load_knowledge_db(db_path)

    with open(network_path, encoding="utf-8") as f:
        data = json.load(f)

    vocab: list = data["vocab"]
    assignments: list = data["assignments"]

    # Build index: tag_name -> list of (record_id, title)
    tag_to_records: dict[str, list[tuple[str, str]]] = {v["name"]: [] for v in vocab}
    for assignment in assignments:
        if not assignment.get("missing"):
            for t in assignment["selected_tags"]:
                tag_name = t["name"]
                if tag_name in tag_to_records:
                    tag_to_records[tag_name].append((assignment["record_id"], assignment["title"]))

    # Write tag files
    for tag in vocab:
        write_tag_file(tags_dir, tag, tag_to_records[tag["name"]])

    # Write record files
    for i, assignment in enumerate(assignments):
        write_record_file(records_dir, orphan_dir, assignment, body_map)
        if (i + 1) % 100 == 0:
            print(f"wrote {i + 1} records...")

    write_graph_config(out_dir)
    write_readme(out_dir, args.network, vocab, assignments)

    tagged_count = sum(1 for a in assignments if not a.get("missing"))
    orphan_count = sum(1 for a in assignments if a.get("missing"))
    total_files = len(vocab) + len(assignments) + 1

    print(f"Wrote vault to {out_dir}/")
    print(f"  tags:    {len(vocab)}")
    print(f"  records: {len(assignments)} ({tagged_count} tagged, {orphan_count} orphan)")
    print(f"  README:  1")
    print(f"Total: {total_files} files")


if __name__ == "__main__":
    main()

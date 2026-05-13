#!/usr/bin/env python3
"""Pre-filter candidate records for golden 80 set, grouped by 9 cells.

9 cells = 3 difficulty × 3 lesson_type, plus 4 hard subcells.
Outputs JSONL with candidates per cell + selection rationale.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

NETWORK = Path("outputs/network.json")
DB = Path("outputs/knowledge.db")
OUT = Path("tests/fixtures/golden_candidates.jsonl")


# ── Cell detection heuristics ────────────────────────────────────────────────

EASY_ANTI = re.compile(r"\b(must not|never|avoid|fragile|error-prone|insufficient|cannot|fails to|breaks?|wrong|incorrect|brittle)\b", re.I)
EASY_DISCOVERY = re.compile(r"\b(silently|unexpectedly|surprisingly|actually|turns out|discovered|behavior is|does not (warn|error|notify))\b", re.I)
EASY_BESTPRACTICE = re.compile(r"\b(should|prefer|recommend|correct way|proper|canonical|best practice|use .+ instead)\b", re.I)

MATTER_KEYWORDS = {
    "llm": re.compile(r"\b(llm|prompt|model|token|embedding|chat|dashscope|openai|claude|qwen|chatmodel|tool call|toolcall)\b", re.I),
    "langgraph": re.compile(r"\b(langgraph|stategraph|checkpointer|memorysaver|sqlitesaver|reducer)\b", re.I),
    "db": re.compile(r"\b(sqlite|database|sql|insert|select|schema|migration|column|table|row|index|knowledgestore)\b", re.I),
    "test": re.compile(r"\b(test|pytest|fixture|mock|assert|unittest|coverage|tdd|integration test|unit test)\b", re.I),
    "git": re.compile(r"\b(git|commit|branch|merge|rebase|stash|diff|gitignore|.gitignore|pull request|fast-forward)\b", re.I),
    "fs": re.compile(r"\b(file|path|directory|filesystem|filename|symlink|symbolic link|pathlib|os\.path)\b", re.I),
    "http": re.compile(r"\b(http|fastapi|endpoint|route|request|response|401|403|404|middleware|status code)\b", re.I),
    "async": re.compile(r"\b(async|await|asyncio|thread|threading|concurrency|race|deadlock|to_thread)\b", re.I),
    "cli": re.compile(r"\b(cli|argparse|argv|script|command-line|terminal|stdout|stderr|shell|bash|typer)\b", re.I),
    "frontend": re.compile(r"\b(browser|dom|html|css|javascript|playwright|chrome|cdp|frontend|page\.)\b", re.I),
    "config": re.compile(r"\b(env|environment variable|config|dotenv|\.env|settings|api key|secret)\b", re.I),
    "deps": re.compile(r"\b(pip|uv|poetry|npm|package|dependency|venv|virtualenv|pyproject)\b", re.I),
    "json": re.compile(r"\b(json|yaml|toml|pickle|serialize|deserialize|marshal|json\.dumps|json\.loads)\b", re.I),
    "log": re.compile(r"\b(log|logging|logger|trace|alert|metric|observability|stack trace)\b", re.I),
    "datetime": re.compile(r"\b(date|datetime|timestamp|timezone|iso 8601|epoch|utc)\b", re.I),
    "deploy": re.compile(r"\b(docker|container|deploy|k8s|kubernetes|build|ci|cd|github action|workflow)\b", re.I),
}

ACTIVITY_KEYWORDS = {
    "design": re.compile(r"\b(design|architecture|schema|interface|contract|abstraction)\b", re.I),
    "debug": re.compile(r"\b(debug|debugging|investigate|trace|diagnose|root cause)\b", re.I),
    "refactor": re.compile(r"\b(refactor|refactoring|extract|decouple|rename|reorganize)\b", re.I),
    "test": re.compile(r"\b(test|testing|fixture|mock|assert)\b", re.I),
    "document": re.compile(r"\b(document|documentation|readme|guide|contributor)\b", re.I),
    "configure": re.compile(r"\b(configure|configuring|setup|setting|config)\b", re.I),
    "migrate": re.compile(r"\b(migrate|migration|upgrade|deprecate|breaking change)\b", re.I),
    "deploy": re.compile(r"\b(deploy|deployment|release|ci/cd|workflow)\b", re.I),
}

PATTERN_KEYWORDS = {
    "explicit": re.compile(r"\b(explicit|explicitly|contract|interface|formal|deliberate)\b", re.I),
    "silent_fail": re.compile(r"\b(silent|silently|swallow|hidden|unobserved|without (any )?(error|message|warning|notification))\b", re.I),
    "validation": re.compile(r"\b(validate|validation|assert|check|guard|precondition|sanitize)\b", re.I),
    "fallback": re.compile(r"\b(fallback|graceful|degradation|default|retry|backup)\b", re.I),
    "ssot": re.compile(r"\b(single source|canonical|authoritative|duplicate|drift|centralized)\b", re.I),
    "separation": re.compile(r"\b(separation of concerns|decouple|isolation|boundary|orthogonal)\b", re.I),
    "leak": re.compile(r"\b(leak|coupling|tightly coupled|side effect|spillover|bleed through)\b", re.I),
}


def main():
    net = json.loads(NETWORK.read_text())
    assigns = {a["record_id"]: a for a in net["assignments"]}

    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT record_id, title, insight, applicability, scope FROM source_knowledge_records WHERE insight IS NOT NULL"
    ).fetchall()
    print(f"loaded {len(rows)} records", file=sys.stderr)

    # Identify v1 orphans
    v1_orphans = {a["record_id"] for a in net["assignments"] if a.get("missing")}
    print(f"v1 orphans: {len(v1_orphans)}", file=sys.stderr)

    candidates_per_cell: dict[str, list] = {}

    for rid, title, insight, app, scope in rows:
        text = f"{title} {insight} {app or ''}"

        # Count matter / activity / pattern hits
        matter_hits = [k for k, p in MATTER_KEYWORDS.items() if p.search(text)]
        activity_hits = [k for k, p in ACTIVITY_KEYWORDS.items() if p.search(text)]
        pattern_hits = [k for k, p in PATTERN_KEYWORDS.items() if p.search(text)]

        # Lesson type signals
        anti = bool(EASY_ANTI.search(text))
        disc = bool(EASY_DISCOVERY.search(text))
        best = bool(EASY_BESTPRACTICE.search(text))
        polarity_hits = sum([anti, disc, best])

        candidate = {
            "record_id": rid,
            "title": title,
            "insight": insight[:300],
            "applicability": (app or "")[:200],
            "scope": scope,
            "matter_hits": matter_hits,
            "activity_hits": activity_hits,
            "pattern_hits": pattern_hits,
            "lesson_anti": anti,
            "lesson_discovery": disc,
            "lesson_best": best,
        }

        # ── Cell assignment ──
        is_orphan = rid in v1_orphans

        # Easy: only one lesson_type fires + ≤ 2 matter + simple
        if polarity_hits == 1 and len(matter_hits) <= 2 and len(insight) < 400:
            if anti:
                candidates_per_cell.setdefault("easy_anti_pattern", []).append(candidate)
            elif disc:
                candidates_per_cell.setdefault("easy_discovery", []).append(candidate)
            elif best:
                candidates_per_cell.setdefault("easy_best_practice", []).append(candidate)

        # Medium: 2 matter (cross-matter) or 2+ activity (activity ambiguous) or 2 patterns (pattern boundary)
        if len(matter_hits) >= 2 and len(matter_hits) <= 3 and polarity_hits == 1:
            candidates_per_cell.setdefault("medium_cross_matter", []).append(candidate)

        if len(activity_hits) >= 2 and len(activity_hits) <= 3:
            candidates_per_cell.setdefault("medium_activity_ambiguous", []).append(candidate)

        if len(pattern_hits) >= 2 and len(pattern_hits) <= 3:
            candidates_per_cell.setdefault("medium_pattern_boundary", []).append(candidate)

        # Hard: orphans
        if is_orphan:
            candidates_per_cell.setdefault("hard_v1_orphan", []).append(candidate)

        # Hard: multi-matter (3+)
        if len(matter_hits) >= 3:
            candidates_per_cell.setdefault("hard_multi_matter", []).append(candidate)

        # Hard: lesson type ambiguous (2+ polarity signals)
        if polarity_hits >= 2:
            candidates_per_cell.setdefault("hard_lesson_type_ambiguous", []).append(candidate)

    print("\n=== Candidate pool sizes ===", file=sys.stderr)
    for cell, cands in sorted(candidates_per_cell.items()):
        print(f"  {cell:<32} {len(cands)}", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for cell, cands in candidates_per_cell.items():
            for c in cands:
                c["cell"] = cell
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    total = sum(len(c) for c in candidates_per_cell.values())
    print(f"\nwrote {total} candidate entries to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regenerate Activity and Pattern facets for vocab_v2.1 using data-driven clustering.

Three-stage pipeline:
  Stage 1 – LLM extracts activity_phrase + pattern_phrase from each record.
  Stage 2 – KMeans(8) on activity phrases → LLM names Activity facet tags.
  Stage 3 – KMeans(6) on pattern phrases → LLM names Pattern facet tags.
  Stage 4 – Synthesise vocab_v2.1.json (Matter / lesson_type preserved from v2).
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model
from consolidate_agent.vocab_maintenance.observability import RunLogger, invoke_with_retry
from consolidate_agent.vocab_maintenance.similarity import _embed

# ── Paths ────────────────────────────────────────────────────────────────────

DB_PATH = ROOT / "outputs" / "knowledge.db"
VOCAB_V2_PATH = ROOT / "docs" / "plans" / "vocab_v2.json"
VOCAB_V21_PATH = ROOT / "docs" / "plans" / "vocab_v2.1.json"
ASPECTS_JSONL = ROOT / "outputs" / "record_aspects.jsonl"
ACTIVITY_CLUSTERS_JSON = ROOT / "outputs" / "activity_clusters.json"
PATTERN_CLUSTERS_JSON = ROOT / "outputs" / "pattern_clusters.json"
LOG_PATH = ROOT / "outputs" / "runs" / "regenerate_facets.jsonl"

# ── Clustering constants ─────────────────────────────────────────────────────

ACTIVITY_N_CLUSTERS = 8
PATTERN_N_CLUSTERS = 6
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
CLUSTER_REPRESENTATIVE_MIN = 8
CLUSTER_REPRESENTATIVE_MAX = 12
EMBED_CONCURRENCY = 5
LLM_CONCURRENCY = 10
ASPECT_BATCH_SIZE = 8
ASPECT_MAX_RETRIES = 3
CLUSTER_MIN_SIZE = 3  # clusters smaller than this are merged into nearest

# ── Pydantic models ──────────────────────────────────────────────────────────


class RecordAspect(BaseModel):
    record_idx: int
    activity_phrase: str = Field(
        description=(
            "5-15 word, tech-agnostic phrase about WHAT KIND OF ENGINEERING WORK the lesson "
            "surfaced during. Avoid product/library names. "
            "E.g. 'investigating a silent state-update bug', "
            "'designing migration coexistence layer', "
            "'configuring environment variable propagation in subprocess'."
        )
    )
    pattern_phrase: str = Field(
        description=(
            "5-15 word phrase about WHAT ABSTRACT PATTERN/PRINCIPLE the lesson illustrates, "
            "NOT what the record is technically about. Avoid technology names. "
            "E.g. 'failure mode hidden by silent default fallback', "
            "'data integrity broken by missing boundary validation', "
            "'explicit interface contract preventing implicit drift'."
        )
    )


class AspectBatchOutput(BaseModel):
    aspects: list[RecordAspect]


class FacetTagOutput(BaseModel):
    name: str = Field(description="snake_case 1-2 words; must be gerund/action word for Activity, abstract pattern for Pattern")
    definition: str = Field(description="definition sentence starting with the tag name word, includes concrete signals and boundary")


# ── Prompts ──────────────────────────────────────────────────────────────────

ASPECT_SYSTEM = """\
You read engineering knowledge records. For each record extract two independent phrases:

1. activity_phrase: describes the KIND OF ENGINEERING WORK happening when this lesson surfaced.
   - Focus on the engineering activity: debugging / building / migrating / testing / configuring / designing
   - Must be tech-agnostic — no product names (no langgraph, sqlite, git, fastapi, pytest, etc.)
   - 5-15 words, phrase form (not a sentence)
   - E.g. "investigating a silent state-update bug", "designing migration coexistence layer"

2. pattern_phrase: describes the ABSTRACT PATTERN or PRINCIPLE illustrated by the lesson.
   - Cross-technology — must apply beyond the specific tool mentioned
   - No product/library names
   - 5-15 words, phrase form
   - E.g. "failure mode hidden by silent default fallback", "implicit assumption broken at abstraction boundary"

Both phrases must be independent: activity = what you were doing, pattern = what lesson it reveals."""

ASPECT_USER = """\
Batch of {batch_size} records (record_idx is 1-based within this batch):

{records}

For each record output activity_phrase and pattern_phrase."""


ACTIVITY_CLUSTER_SYSTEM = """\
You are naming a cluster of engineering activity phrases that share a common type of engineering work.

Goal: produce ONE tag that names the activity category.

RULES:
1. name: 1-2 words, snake_case, MUST be a gerund or verbal noun (e.g., debugging, refactoring, configuring, migrating, testing, deploying)
2. definition format EXACTLY:
   "[Gerund] is <concise description>. Concrete signals: <A>, <B>, <C>. Boundary: not <neighbor_activity> (use <other_tag> instead)."
   — The definition MUST start with the tag name word (e.g., if name=debugging, definition starts with "Debugging is...")
3. FORBIDDEN umbrella names: engineering, development, programming, coding, implementation, work, activity
4. DO NOT use product names (langgraph, pytest, sqlite, git, etc.)
5. The name must describe a specific kind of engineering work, not a lifecycle phase

Output format:
- name: snake_case 1-2 words, e.g. "debugging" or "migrating"
- definition: "Debugging is X. Concrete signals: A, B, C. Boundary: not refactoring (use refactoring instead)." """

ACTIVITY_CLUSTER_USER = """\
Cluster of {n} activity phrases representing one type of engineering work:

{phrases}

Name this activity category with a gerund tag + definition."""


PATTERN_CLUSTER_SYSTEM = """\
You are naming a cluster of pattern phrases that share a common abstract engineering lesson.

Goal: produce ONE tag that names the recurring principle or anti-pattern.

RULES:
1. name: 1-3 words, snake_case, describing an abstract principle (e.g., silent_failure, early_validation, single_source_of_truth)
2. definition format EXACTLY:
   "[Name] is <description>. Concrete signals: <A>, <B>, <C>. Boundary: not <different_pattern> (use <other> instead)."
   — The definition MUST start with the tag name's first word (e.g., if name=silent_failure, definition starts with "Silent failure is...")
3. FORBIDDEN names: explicit_contract, separation_of_concerns, abstraction (these are over-broad meta-concepts)
4. FORBIDDEN: product or technology names in the tag name
5. The name must describe a pattern a practitioner would look for across multiple tech stacks

Output format:
- name: snake_case 1-3 words, e.g. "silent_failure" or "early_validation"
- definition: "Silent failure is X. Concrete signals: A, B, C. Boundary: not Y (use Z instead)." """

PATTERN_CLUSTER_USER = """\
Cluster of {n} pattern phrases representing one abstract engineering principle:

{phrases}

Name this pattern with a snake_case tag + definition."""


# ── Stage 1: LLM aspect extraction ──────────────────────────────────────────


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


def format_records_for_aspect(records: list[dict]) -> str:
    lines = []
    for idx, r in enumerate(records, 1):
        insight = r["insight"][:300]
        lines.append(f"[{idx}] {r['title']}\n    {insight}")
    return "\n\n".join(lines)


def extract_aspects(
    records: list[dict],
    model_factory,
    logger: RunLogger,
) -> list[dict]:
    aspect_model = model_factory().with_structured_output(AspectBatchOutput)
    aspect_prompt = ChatPromptTemplate.from_messages([
        ("system", ASPECT_SYSTEM),
        ("user", ASPECT_USER),
    ])

    # Split into batches
    batches: list[list[dict]] = []
    for i in range(0, len(records), ASPECT_BATCH_SIZE):
        batches.append(records[i : i + ASPECT_BATCH_SIZE])

    print(f"  [aspect] {len(records)} records in {len(batches)} batches of ~{ASPECT_BATCH_SIZE}", flush=True)

    results: list[dict | None] = [None] * len(batches)

    def _run_batch(batch_idx: int, batch: list[dict]) -> tuple[int, list[dict] | None]:
        messages = aspect_prompt.invoke({
            "batch_size": len(batch),
            "records": format_records_for_aspect(batch),
        })
        output: AspectBatchOutput | None = invoke_with_retry(
            aspect_model,
            messages,
            retries=ASPECT_MAX_RETRIES,
            caller=f"aspect_batch_{batch_idx}",
            logger=logger,
        )
        if output is None:
            return batch_idx, None
        aspects_by_idx = {a.record_idx: a for a in output.aspects}
        batch_results = []
        for local_idx, rec in enumerate(batch, 1):
            aspect = aspects_by_idx.get(local_idx)
            if aspect is None:
                # Fallback: use title as phrase rather than silently drop
                import logging
                logging.warning(
                    "aspect_batch_%d: missing record_idx=%d, using title fallback",
                    batch_idx, local_idx,
                )
                batch_results.append({
                    "record_id": rec["record_id"],
                    "activity_phrase": rec["title"][:80],
                    "pattern_phrase": rec["title"][:80],
                })
            else:
                batch_results.append({
                    "record_id": rec["record_id"],
                    "activity_phrase": aspect.activity_phrase,
                    "pattern_phrase": aspect.pattern_phrase,
                })
        return batch_idx, batch_results

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as ex:
        futures = {ex.submit(_run_batch, i, b): i for i, b in enumerate(batches)}
        done = 0
        for fut in as_completed(futures):
            bidx, batch_res = fut.result()
            results[bidx] = batch_res
            done += 1
            if done % 10 == 0 or done == len(batches):
                print(f"  [aspect] {done}/{len(batches)} batches done", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  [aspect] all batches done ({elapsed:.1f}s)", flush=True)

    flat: list[dict] = []
    for batch_result in results:
        if batch_result is not None:
            flat.extend(batch_result)
    return flat


# ── Stage 2/3: Embed + Cluster + Name ────────────────────────────────────────


def embed_phrases(phrases: list[str]) -> np.ndarray:
    print(f"  [embed] embedding {len(phrases)} phrases (concurrency={EMBED_CONCURRENCY})", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as ex:
        vecs = list(ex.map(_embed, phrases))
    elapsed = time.perf_counter() - t0
    print(f"  [embed] done ({elapsed:.1f}s)", flush=True)
    return np.array(vecs)


def kmeans_cluster(
    X: np.ndarray,
    n_clusters: int,
    phrases: list[str],
    record_ids: list[str],
) -> dict[int, list[tuple[int, str, str]]]:
    """Run KMeans; merge clusters smaller than CLUSTER_MIN_SIZE into nearest centroid.

    Returns cluster_id -> [(global_idx, phrase, record_id), ...]
    """
    print(f"  [kmeans] n_clusters={n_clusters}, n_samples={len(phrases)}", flush=True)
    t0 = time.perf_counter()
    km = KMeans(n_clusters=n_clusters, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels = km.fit_predict(X)
    elapsed = time.perf_counter() - t0

    clusters: dict[int, list[tuple[int, str, str]]] = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append((i, phrases[i], record_ids[i]))

    sizes = sorted(len(v) for v in clusters.values())
    print(
        f"  [kmeans] done ({elapsed:.1f}s) sizes: min={sizes[0]}, "
        f"median={sizes[len(sizes)//2]}, max={sizes[-1]}",
        flush=True,
    )

    # Check for degenerate all-in-one-cluster outcome
    if sizes[-1] == len(phrases):
        raise RuntimeError(
            f"KMeans degenerate: all {len(phrases)} items in one cluster. "
            "Cannot produce meaningful tags. Stopping."
        )

    # Merge tiny clusters
    small_ids = [cid for cid, members in clusters.items() if len(members) < CLUSTER_MIN_SIZE]
    if small_ids:
        print(f"  [kmeans] merging {len(small_ids)} small clusters (<{CLUSTER_MIN_SIZE}) into nearest", flush=True)
        centroids = km.cluster_centers_
        for small_cid in small_ids:
            members = clusters.pop(small_cid)
            # Find nearest other cluster centroid
            small_centroid = centroids[small_cid]
            best_cid = min(
                (cid for cid in clusters),
                key=lambda cid: float(np.linalg.norm(centroids[cid] - small_centroid)),
            )
            clusters[best_cid].extend(members)
        print(f"  [kmeans] after merge: {len(clusters)} clusters", flush=True)

    return clusters


def pick_representatives(
    cluster_members: list[tuple[int, str, str]],
    X: np.ndarray,
    centroid: np.ndarray,
) -> list[tuple[int, str, str]]:
    """Pick 8-12 phrases closest to centroid."""
    n = min(CLUSTER_REPRESENTATIVE_MAX, max(CLUSTER_REPRESENTATIVE_MIN, len(cluster_members)))
    distances = [(np.linalg.norm(X[idx] - centroid), idx, phrase, rid) for idx, phrase, rid in cluster_members]
    distances.sort(key=lambda x: x[0])
    return [(idx, phrase, rid) for _, idx, phrase, rid in distances[:n]]


def name_clusters(
    clusters: dict[int, list[tuple[int, str, str]]],
    X: np.ndarray,
    km: KMeans,
    model_factory,
    logger: RunLogger,
    *,
    facet: str,  # "activity" or "pattern"
) -> dict[int, FacetTagOutput]:
    if facet == "activity":
        system_prompt = ACTIVITY_CLUSTER_SYSTEM
        user_prompt = ACTIVITY_CLUSTER_USER
    else:
        system_prompt = PATTERN_CLUSTER_SYSTEM
        user_prompt = PATTERN_CLUSTER_USER

    name_model = model_factory().with_structured_output(FacetTagOutput)
    name_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    def _name_cluster(cid: int) -> tuple[int, FacetTagOutput | None]:
        members = clusters[cid]
        centroid = km.cluster_centers_[cid] if cid < len(km.cluster_centers_) else np.mean(
            [X[idx] for idx, _, _ in members], axis=0
        )
        reps = pick_representatives(members, X, centroid)
        phrase_block = "\n".join(f"- {phrase}" for _, phrase, _ in reps)
        messages = name_prompt.invoke({"n": len(reps), "phrases": phrase_block})
        result = invoke_with_retry(
            name_model,
            messages,
            retries=ASPECT_MAX_RETRIES,
            caller=f"{facet}_cluster_{cid}",
            logger=logger,
        )
        return cid, result

    print(f"  [name] naming {len(clusters)} {facet} clusters (concurrency={LLM_CONCURRENCY})", flush=True)
    t0 = time.perf_counter()
    named: dict[int, FacetTagOutput] = {}
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as ex:
        futures = {ex.submit(_name_cluster, cid): cid for cid in clusters}
        for fut in as_completed(futures):
            cid, result = fut.result()
            if result is None:
                failed.append(cid)
                print(f"  [name] cluster {cid} FAILED (all retries exhausted)", flush=True)
            else:
                named[cid] = result
    elapsed = time.perf_counter() - t0
    print(f"  [name] {len(named)}/{len(clusters)} named ({elapsed:.1f}s, {len(failed)} failures)", flush=True)
    if failed:
        print(f"  [name] WARN: failed cluster ids: {failed}", flush=True)
    return named


RENAME_SYSTEM = """\
A cluster was named "{duplicate_name}" but that name collides with another cluster in the same facet.
Give it a NEW, distinct snake_case name (1-3 words) that accurately describes these phrases.
Keep the same definition structure but replace the name and update the first sentence to start with the new name word.
Return: name (snake_case) + definition (starting with new name word)."""

RENAME_USER = """\
Phrases in this cluster:
{phrases}

Current (duplicate) name: {duplicate_name}
Other names already taken: {taken}

Provide a new distinct name + updated definition."""

DEFLEN_TARGET_WORDS = 35  # trim definitions above this length


def _trim_definition(definition: str, max_words: int = DEFLEN_TARGET_WORDS) -> str:
    """Trim definition to at most max_words by cutting complete sentences."""
    words = definition.split()
    if len(words) <= max_words:
        return definition
    # Try cutting at sentence boundaries
    sentences = definition.replace(". Concrete signals:", "\x00Concrete signals:").replace(
        ". Boundary:", "\x00Boundary:"
    ).split("\x00")
    result = ""
    for sent in sentences:
        candidate = (result + " " + sent).strip() if result else sent
        if len(candidate.split()) <= max_words:
            result = candidate
        else:
            break
    if result:
        return result.rstrip(". ") + "."
    # Last resort: hard truncate at max_words
    return " ".join(words[:max_words]).rstrip(",") + "."


def resolve_duplicate_names(
    named: dict[int, FacetTagOutput],
    clusters: dict[int, list[tuple[int, str, str]]],
    X: np.ndarray,
    km: KMeans,
    model_factory,
    logger: RunLogger,
    *,
    facet: str,
) -> dict[int, FacetTagOutput]:
    """Detect duplicate tag names across clusters and re-prompt LLM for the duplicates."""
    from collections import Counter

    name_to_cids: dict[str, list[int]] = {}
    for cid, tag in named.items():
        name_to_cids.setdefault(tag.name, []).append(cid)

    duplicates = {name: cids for name, cids in name_to_cids.items() if len(cids) > 1}
    if not duplicates:
        return named

    print(f"  [dedup] found duplicates: {list(duplicates.keys())}", flush=True)

    rename_model = model_factory().with_structured_output(FacetTagOutput)
    rename_prompt = ChatPromptTemplate.from_messages([
        ("system", RENAME_SYSTEM),
        ("user", RENAME_USER),
    ])

    result = dict(named)
    for dup_name, cids in duplicates.items():
        # Keep the first (lowest cid) occurrence unchanged; rename the rest
        for cid_to_rename in cids[1:]:
            taken = sorted({t.name for ocid, t in result.items() if ocid != cid_to_rename})
            members = clusters[cid_to_rename]
            centroid = km.cluster_centers_[cid_to_rename] if cid_to_rename < len(km.cluster_centers_) else np.mean(
                [X[idx] for idx, _, _ in members], axis=0
            )
            reps = pick_representatives(members, X, centroid)
            phrase_block = "\n".join(f"- {p}" for _, p, _ in reps)
            messages = rename_prompt.invoke({
                "duplicate_name": dup_name,
                "phrases": phrase_block,
                "taken": ", ".join(taken),
            })
            new_tag = invoke_with_retry(
                rename_model,
                messages,
                retries=3,
                caller=f"{facet}_rename_{cid_to_rename}",
                logger=logger,
            )
            if new_tag is None:
                # Fallback: append cluster id to make unique
                old = result[cid_to_rename]
                result[cid_to_rename] = FacetTagOutput(
                    name=f"{old.name}_{cid_to_rename}",
                    definition=old.definition,
                )
                print(
                    f"  [dedup] WARN: rename LLM failed for cluster {cid_to_rename}, "
                    f"appended id as suffix: {result[cid_to_rename].name}",
                    flush=True,
                )
            else:
                result[cid_to_rename] = new_tag
                print(f"  [dedup] cluster {cid_to_rename}: {dup_name} -> {new_tag.name}", flush=True)

    return result


def build_cluster_audit(
    clusters: dict[int, list[tuple[int, str, str]]],
    X: np.ndarray,
    km: KMeans,
    named: dict[int, FacetTagOutput],
) -> list[dict]:
    audit = []
    for cid, members in sorted(clusters.items()):
        centroid = km.cluster_centers_[cid] if cid < len(km.cluster_centers_) else np.mean(
            [X[idx] for idx, _, _ in members], axis=0
        )
        reps = pick_representatives(members, X, centroid)
        tag = named.get(cid)
        audit.append({
            "cluster_id": cid,
            "size": len(members),
            "tag_name": tag.name if tag else None,
            "tag_definition": tag.definition if tag else None,
            "representative_phrases": [phrase for _, phrase, _ in reps],
            "representative_record_ids": [rid for _, _, rid in reps],
            "all_record_ids": [rid for _, _, rid in members],
        })
    return audit


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run() -> None:
    settings = Settings()

    def model_factory():
        return _chat_model(settings)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(LOG_PATH)

    t_total = time.perf_counter()

    # ── Stage 1: Load records ────────────────────────────────────────────────
    print("\n=== Stage 1: LLM aspect extraction ===\n", flush=True)
    records = load_records(DB_PATH)
    print(f"  Loaded {len(records)} records from {DB_PATH}", flush=True)

    if ASPECTS_JSONL.exists():
        print(f"  Found existing {ASPECTS_JSONL}, loading...", flush=True)
        aspects: list[dict] = []
        with ASPECTS_JSONL.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    aspects.append(json.loads(line))
        print(f"  Loaded {len(aspects)} cached aspects", flush=True)
        if len(aspects) != len(records):
            print(
                f"  WARN: cached aspects count ({len(aspects)}) != records ({len(records)}), re-extracting",
                flush=True,
            )
            aspects = extract_aspects(records, model_factory, logger)
            ASPECTS_JSONL.write_text(
                "\n".join(json.dumps(a, ensure_ascii=False) for a in aspects) + "\n",
                encoding="utf-8",
            )
    else:
        aspects = extract_aspects(records, model_factory, logger)
        ASPECTS_JSONL.write_text(
            "\n".join(json.dumps(a, ensure_ascii=False) for a in aspects) + "\n",
            encoding="utf-8",
        )

    print(f"  {len(aspects)} aspects -> {ASPECTS_JSONL}", flush=True)

    # Build index record_id -> aspect
    aspect_by_id = {a["record_id"]: a for a in aspects}

    activity_phrases: list[str] = []
    pattern_phrases: list[str] = []
    aspect_record_ids: list[str] = []
    for rec in records:
        asp = aspect_by_id.get(rec["record_id"])
        if asp is None:
            activity_phrases.append(rec["title"])
            pattern_phrases.append(rec["title"])
        else:
            activity_phrases.append(asp["activity_phrase"])
            pattern_phrases.append(asp["pattern_phrase"])
        aspect_record_ids.append(rec["record_id"])

    # ── Stage 2: Activity facet ──────────────────────────────────────────────
    print("\n=== Stage 2: Activity facet clustering ===\n", flush=True)

    X_activity = embed_phrases(activity_phrases)
    km_activity = KMeans(
        n_clusters=ACTIVITY_N_CLUSTERS,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    km_activity.fit(X_activity)
    labels_activity = km_activity.labels_

    act_raw_clusters: dict[int, list[tuple[int, str, str]]] = {}
    for i, lbl in enumerate(labels_activity):
        act_raw_clusters.setdefault(int(lbl), []).append((i, activity_phrases[i], aspect_record_ids[i]))

    sizes_a = sorted(len(v) for v in act_raw_clusters.values())
    print(
        f"  [kmeans] done sizes: min={sizes_a[0]}, median={sizes_a[len(sizes_a)//2]}, max={sizes_a[-1]}",
        flush=True,
    )
    if sizes_a[-1] == len(activity_phrases):
        raise RuntimeError("KMeans degenerate for Activity: all in one cluster.")

    # Merge small clusters
    small_act = [cid for cid, members in act_raw_clusters.items() if len(members) < CLUSTER_MIN_SIZE]
    if small_act:
        print(f"  Merging {len(small_act)} small activity clusters", flush=True)
        for small_cid in small_act:
            members = act_raw_clusters.pop(small_cid)
            best_cid = min(
                (cid for cid in act_raw_clusters),
                key=lambda cid: float(np.linalg.norm(km_activity.cluster_centers_[cid] - km_activity.cluster_centers_[small_cid])),
            )
            act_raw_clusters[best_cid].extend(members)

    activity_named = name_clusters(
        act_raw_clusters, X_activity, km_activity, model_factory, logger, facet="activity"
    )
    activity_named = resolve_duplicate_names(
        activity_named, act_raw_clusters, X_activity, km_activity, model_factory, logger, facet="activity"
    )
    # Trim long definitions
    for cid, tag in activity_named.items():
        trimmed = _trim_definition(tag.definition)
        if trimmed != tag.definition:
            activity_named[cid] = FacetTagOutput(name=tag.name, definition=trimmed)
    activity_audit = build_cluster_audit(act_raw_clusters, X_activity, km_activity, activity_named)
    ACTIVITY_CLUSTERS_JSON.write_text(
        json.dumps(activity_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Activity clusters -> {ACTIVITY_CLUSTERS_JSON}", flush=True)

    # ── Stage 3: Pattern facet ───────────────────────────────────────────────
    print("\n=== Stage 3: Pattern facet clustering ===\n", flush=True)

    X_pattern = embed_phrases(pattern_phrases)
    km_pattern = KMeans(
        n_clusters=PATTERN_N_CLUSTERS,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    km_pattern.fit(X_pattern)
    labels_pattern = km_pattern.labels_

    pat_raw_clusters: dict[int, list[tuple[int, str, str]]] = {}
    for i, lbl in enumerate(labels_pattern):
        pat_raw_clusters.setdefault(int(lbl), []).append((i, pattern_phrases[i], aspect_record_ids[i]))

    sizes_p = sorted(len(v) for v in pat_raw_clusters.values())
    print(
        f"  [kmeans] done sizes: min={sizes_p[0]}, median={sizes_p[len(sizes_p)//2]}, max={sizes_p[-1]}",
        flush=True,
    )
    if sizes_p[-1] == len(pattern_phrases):
        raise RuntimeError("KMeans degenerate for Pattern: all in one cluster.")

    small_pat = [cid for cid, members in pat_raw_clusters.items() if len(members) < CLUSTER_MIN_SIZE]
    if small_pat:
        print(f"  Merging {len(small_pat)} small pattern clusters", flush=True)
        for small_cid in small_pat:
            members = pat_raw_clusters.pop(small_cid)
            best_cid = min(
                (cid for cid in pat_raw_clusters),
                key=lambda cid: float(np.linalg.norm(km_pattern.cluster_centers_[cid] - km_pattern.cluster_centers_[small_cid])),
            )
            pat_raw_clusters[best_cid].extend(members)

    pattern_named = name_clusters(
        pat_raw_clusters, X_pattern, km_pattern, model_factory, logger, facet="pattern"
    )
    pattern_named = resolve_duplicate_names(
        pattern_named, pat_raw_clusters, X_pattern, km_pattern, model_factory, logger, facet="pattern"
    )
    # Trim long definitions
    for cid, tag in pattern_named.items():
        trimmed = _trim_definition(tag.definition)
        if trimmed != tag.definition:
            pattern_named[cid] = FacetTagOutput(name=tag.name, definition=trimmed)
    pattern_audit = build_cluster_audit(pat_raw_clusters, X_pattern, km_pattern, pattern_named)
    PATTERN_CLUSTERS_JSON.write_text(
        json.dumps(pattern_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Pattern clusters -> {PATTERN_CLUSTERS_JSON}", flush=True)

    # ── Stage 4: Synthesise vocab_v2.1.json ─────────────────────────────────
    print("\n=== Stage 4: Synthesise vocab_v2.1.json ===\n", flush=True)

    v2 = json.loads(VOCAB_V2_PATH.read_text(encoding="utf-8"))

    activity_tags = [
        {"name": named.name, "definition": named.definition}
        for cid, named in sorted(activity_named.items())
        if named is not None
    ]
    pattern_tags = [
        {"name": named.name, "definition": named.definition}
        for cid, named in sorted(pattern_named.items())
        if named is not None
    ]

    v21 = {
        "version": "v2.1-data-driven-activity-pattern",
        "drafted_at": "2026-05-13",
        "status": "generated — Activity and Pattern facets are data-driven (KMeans+LLM), Matter and lesson_type preserved from v2.0",
        "scheme": v2["scheme"],
        "notes": (
            "Activity facet: embed→KMeans(8)→LLM named, data-driven from 696 record activity phrases. "
            "Pattern facet: embed→KMeans(6)→LLM named, data-driven from 696 record pattern phrases. "
            "Matter facet and lesson_type preserved from vocab_v2.0."
        ),
        "facets": {
            "matter": v2["facets"]["matter"],
            "activity": {
                "description": v2["facets"]["activity"]["description"],
                "tags": activity_tags,
            },
            "pattern": {
                "description": v2["facets"]["pattern"]["description"],
                "tags": pattern_tags,
            },
        },
        "lesson_type": v2["lesson_type"],
    }

    VOCAB_V21_PATH.write_text(
        json.dumps(v21, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  vocab_v2.1.json -> {VOCAB_V21_PATH}", flush=True)
    print(f"  Activity: {len(activity_tags)} tags", flush=True)
    print(f"  Pattern:  {len(pattern_tags)} tags", flush=True)

    # ── Run validate_vocab_structure.py ──────────────────────────────────────
    print("\n=== Validate vocab_v2.1.json ===\n", flush=True)
    validate_script = ROOT / "scripts" / "validate_vocab_structure.py"
    proc = subprocess.run(
        [sys.executable, str(validate_script), str(VOCAB_V21_PATH)],
        capture_output=False,
    )
    if proc.returncode == 2:
        print("\n  WARN: validate_vocab_structure reported FAIL (see output above)", flush=True)
    elif proc.returncode == 1:
        print("\n  WARN: validate_vocab_structure reported WARN (see output above)", flush=True)

    total_elapsed = time.perf_counter() - t_total
    print(f"\nTotal elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)

    logger.close()


if __name__ == "__main__":
    run()

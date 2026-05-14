"""Propose a tag split: decompose a semantically overloaded tag into 2-3 sub-tags.

Triggered when a tag exhibits low intra-cluster coherence (records assigned to
it are semantically diverse). Workflow:
  1. Load stored record embeddings from db (no LLM embed calls).
  2. LLM inspects a sample of records and decides n_groups (2 or 3).
  3. KMeans clusters the records into n_groups.
  4. LLM names each cluster, constrained to be more specific than the parent tag.
  5. Returns SplitTagProposal (or [] if LLM declines / invariants would fail).
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Literal

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model

from ..apply import SplitTagProposal
from ..bootstrap import CLUSTER_NAME_SYSTEM
from ..observability import get_default_logger, invoke_with_retry
from ..probes import _load_record_details


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_RECORDS_TO_SPLIT = 4   # absolute minimum records a tag needs before splitting


# ── Pydantic output models ────────────────────────────────────────────────────


class SplitDecision(BaseModel):
    n_groups: Literal[2, 3] = Field(
        description="number of groups to split into: 2 or 3"
    )
    reasoning: str = Field(description="brief justification for this number")


class SubTagProposal(BaseModel):
    name: str = Field(
        description="snake_case 1-3 words, more specific than parent tag"
    )
    definition: str = Field(
        description=(
            "one sentence, include name keyword, add boundary vs other sub-tags"
        )
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

_SPLIT_DECISION_SYSTEM = """You are analyzing a knowledge tag to decide whether its records form 2 or 3 distinct semantic sub-groups.

Given:
- The current tag name + definition
- A sample of records assigned to this tag (title + insight excerpt)

Decide: do these records naturally cluster into 2 groups or 3 groups?

Guidelines:
- If you see high diversity (e.g., an http_api tag covering auth patterns, streaming responses, AND routing strategies) → 3 groups
- If you see moderate diversity with two clear themes → 2 groups
- When in doubt, prefer 2 groups (safer, less fragmentation)
- Output ONLY the number and a brief reasoning."""

_SPLIT_DECISION_USER = """Tag: {tag_name}
Definition: {tag_def}

Sample records ({n_sample} records):
{record_sample}

How many semantic sub-groups do these records form? Answer 2 or 3."""


_CLUSTER_NAME_USER = """Parent tag: {parent_tag}
Parent definition: {parent_def}
Other sub-tags being created: {other_sub_tags}

Cluster of {n} records sharing one specific sub-pattern of the parent tag:

{themes}

Name this sub-pattern. The name MUST be more specific than "{parent_tag}".
Distinguished from: {other_sub_tags}"""


# ── DB helpers ────────────────────────────────────────────────────────────────


def _load_record_embeddings(
    db_path: Path, record_ids: list[str]
) -> dict[str, tuple[float, ...]]:
    if not record_ids:
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(record_ids))
    rows = conn.execute(
        f"SELECT record_id, embedding FROM source_knowledge_records WHERE record_id IN ({placeholders})",
        record_ids,
    ).fetchall()
    conn.close()
    result: dict[str, tuple[float, ...]] = {}
    for row in rows:
        raw = row["embedding"]
        if raw is None:
            continue
        if isinstance(raw, (bytes, str)):
            vec = json.loads(raw)
        else:
            vec = raw
        result[row["record_id"]] = tuple(vec)
    return result


# ── Core function ─────────────────────────────────────────────────────────────


def propose_split_fn(
    vocab: list[dict],
    assignments: list[dict],
    focus: str,
    *,
    db_path: Path,
    random_seed: int = 42,
    n_sample_for_decision: int = 10,
    n_representative: int = 6,
) -> list[SplitTagProposal]:
    """Propose splitting `focus` tag into 2-3 more specific sub-tags.

    Returns [SplitTagProposal] on success, [] if skipped or declined.
    """
    logger = get_default_logger()

    tag = next((t for t in vocab if t["name"] == focus), None)
    if tag is None:
        logger.event("propose_split.tag_not_found", focus=focus[:200])
        return []

    # gather record ids assigned to this tag
    tag_record_ids = [
        a["record_id"]
        for a in assignments
        if not a.get("missing")
        and any(t["name"] == focus for t in a.get("selected_tags", []))
    ]
    if len(tag_record_ids) < MIN_RECORDS_TO_SPLIT:
        logger.event(
            "propose_split.too_few_records",
            tag=focus,
            count=len(tag_record_ids),
            min_required=MIN_RECORDS_TO_SPLIT,
        )
        return []

    # load stored embeddings
    emb_by_id = _load_record_embeddings(db_path, tag_record_ids)
    # keep only records that have embeddings
    valid_ids = [rid for rid in tag_record_ids if rid in emb_by_id]
    if len(valid_ids) < MIN_RECORDS_TO_SPLIT:
        logger.event(
            "propose_split.too_few_embeddings",
            tag=focus,
            valid=len(valid_ids),
        )
        return []

    # load record details (title + insight) for prompt construction
    details = _load_record_details(db_path, valid_ids)

    settings = Settings()

    # ── Step 1: LLM decides n_groups ─────────────────────────────────────────
    rng = random.Random(random_seed)
    sample_ids = rng.sample(valid_ids, min(n_sample_for_decision, len(valid_ids)))
    record_sample_lines = []
    for rid in sample_ids:
        d = details.get(rid, {})
        title = d.get("title", rid)
        insight = d.get("insight", "")[:150]
        record_sample_lines.append(f"- [{rid}] {title}: {insight}")
    record_sample_text = "\n".join(record_sample_lines)

    decision_model = _chat_model(settings).with_structured_output(SplitDecision)
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", _SPLIT_DECISION_SYSTEM),
        ("user", _SPLIT_DECISION_USER),
    ])
    decision_msg = decision_prompt.invoke({
        "tag_name": focus,
        "tag_def": tag["definition"],
        "n_sample": len(sample_ids),
        "record_sample": record_sample_text,
    })
    decision: SplitDecision | None = invoke_with_retry(
        decision_model, decision_msg, retries=3,
        caller=f"propose_split.decision.{focus}", logger=logger,
    )
    if decision is None:
        logger.event("propose_split.decision_failed", tag=focus)
        return []

    n_groups = decision.n_groups
    logger.event(
        "propose_split.decision",
        tag=focus,
        n_groups=n_groups,
        reasoning=decision.reasoning[:200],
    )

    # ── Step 2: KMeans cluster ────────────────────────────────────────────────
    vecs = np.array([list(emb_by_id[rid]) for rid in valid_ids])
    km = KMeans(n_clusters=n_groups, random_state=random_seed, n_init=10)
    labels = km.fit_predict(vecs)
    centroids = km.cluster_centers_

    # cluster_id -> list of (record_id, distance_to_centroid)
    clusters: dict[int, list[tuple[str, float]]] = {i: [] for i in range(n_groups)}
    for idx, (rid, label) in enumerate(zip(valid_ids, labels)):
        diff = vecs[idx] - centroids[label]
        dist = float(np.dot(diff, diff) ** 0.5)
        clusters[int(label)].append((rid, dist))
    # sort each cluster by distance ascending (closest to centroid first)
    for cid in clusters:
        clusters[cid].sort(key=lambda x: x[1])

    # ── Step 3: LLM names each cluster ───────────────────────────────────────
    name_model = _chat_model(settings).with_structured_output(SubTagProposal)
    name_prompt = ChatPromptTemplate.from_messages([
        ("system", CLUSTER_NAME_SYSTEM),
        ("user", _CLUSTER_NAME_USER),
    ])

    named_clusters: list[dict] = []  # {name, definition, record_ids: list[str]}
    for cid in range(n_groups):
        members = clusters[cid]
        rep_ids = [rid for rid, _ in members[:n_representative]]
        other_names = [nc["name"] for nc in named_clusters]
        other_sub_str = ", ".join(other_names) if other_names else "(none yet)"

        theme_lines = []
        for rid in rep_ids:
            d = details.get(rid, {})
            title = d.get("title", rid)
            insight = d.get("insight", "")[:150]
            theme_lines.append(f"- {title}: {insight}")
        themes_text = "\n".join(theme_lines)

        name_msg = name_prompt.invoke({
            "parent_tag": focus,
            "parent_def": tag["definition"],
            "other_sub_tags": other_sub_str,
            "n": len(rep_ids),
            "themes": themes_text,
        })
        sub_proposal: SubTagProposal | None = invoke_with_retry(
            name_model, name_msg, retries=3,
            caller=f"propose_split.name.{focus}.cluster{cid}", logger=logger,
        )
        if sub_proposal is None:
            logger.event("propose_split.naming_failed", tag=focus, cluster=cid)
            return []

        named_clusters.append({
            "name": sub_proposal.name,
            "definition": sub_proposal.definition,
            "record_ids": tuple(rid for rid, _ in members),
        })
        logger.event(
            "propose_split.cluster_named",
            tag=focus,
            cluster=cid,
            sub_tag=sub_proposal.name,
            size=len(members),
        )

    # ── Step 4: construct SplitTagProposal ───────────────────────────────────
    proposal = SplitTagProposal(
        tag=focus,
        sub_tags=tuple(named_clusters),
    )
    logger.event(
        "propose_split.ok",
        tag=focus,
        sub_tags=[nc["name"] for nc in named_clusters],
        total_records=len(valid_ids),
    )
    return [proposal]

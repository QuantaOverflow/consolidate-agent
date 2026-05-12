"""Compute pairwise definition similarity for vocab tags.

Used to surface suspected near-synonym pairs to the diagnostic agent.
Pure deterministic — embedding cache keyed by definition text.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from functools import lru_cache
from pathlib import Path

from dashscope import TextEmbedding


@lru_cache(maxsize=4096)
def _embed(text: str, model: str = "text-embedding-v3") -> tuple[float, ...]:
    """Get embedding via DashScope. Cached per text.

    Retries with exponential backoff + jitter on transient throttling
    (`Throttling.RateQuota`). DashScope embedding QPS is ~50/account
    for v3, so concurrency × call-rate easily exceeds it — back off
    rather than crash the whole synthesize.
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        resp = TextEmbedding.call(model=model, input=text, api_key=api_key)
        if resp.status_code == 200:
            return tuple(resp.output["embeddings"][0]["embedding"])
        code = getattr(resp, "code", "")
        if "Throttling" in str(code) and attempt < max_attempts:
            # 0.5, 1, 2, 4 seconds + jitter
            time.sleep((0.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5))
            continue
        raise RuntimeError(f"embedding failed: {code} {resp.message}")
    raise RuntimeError(f"embedding failed after {max_attempts} attempts: throttled")


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def find_records_similar_to_tag(
    themes: dict[str, str],
    assignments: list[dict],
    tag_definition: str,
    *,
    threshold: float = 0.6,
    top_n: int = 30,
    max_existing_tags: int = 3,
) -> list[tuple[float, dict]]:
    """Find already-assigned records semantically close to a tag definition.

    Used by ingest's "additive re-check" — when propose_new adds a new tag,
    this finds existing (non-missing) records whose themes match the new tag,
    so the new tag can be additively attached to them.

    Returns [(similarity, assignment_dict), ...] sorted by similarity desc,
    capped at top_n. Records that are missing, have full tag slots, or have
    no theme cached are skipped.
    """
    if not tag_definition or not themes:
        return []
    tag_emb = _embed(tag_definition)
    scored: list[tuple[float, dict]] = []
    for a in assignments:
        if a.get("missing"):
            continue
        if len(a.get("selected_tags", [])) >= max_existing_tags:
            continue
        rid = a["record_id"]
        theme = themes.get(rid, "")
        if not theme:
            continue
        sim = cosine(tag_emb, _embed(theme))
        if sim >= threshold:
            scored.append((sim, a))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_n]


def find_similar_pairs(vocab: list[dict], threshold: float = 0.80, top_n: int = 5) -> list[dict]:
    """Compute pairwise def cosine, return pairs above threshold sorted desc.

    Returns at most top_n pairs to keep prompt size sane.
    """
    embeddings = {t["name"]: _embed(t["definition"]) for t in vocab}
    names = [t["name"] for t in vocab]

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = cosine(embeddings[names[i]], embeddings[names[j]])
            if sim >= threshold:
                pairs.append({
                    "tag_a": names[i],
                    "tag_b": names[j],
                    "similarity": round(sim, 3),
                })

    pairs.sort(key=lambda x: -x["similarity"])
    return pairs[:top_n]


def main():
    """CLI for inspection."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--threshold", default=0.80, type=float)
    parser.add_argument("--top-n", default=10, type=int)
    args = parser.parse_args()

    vocab = json.loads(args.vocab.read_text())["vocab"]
    pairs = find_similar_pairs(vocab, threshold=args.threshold, top_n=args.top_n)
    print(f"Top {len(pairs)} similar pairs (threshold {args.threshold}):")
    for p in pairs:
        print(f"  {p['similarity']:.3f}  {p['tag_a']} ↔ {p['tag_b']}")


if __name__ == "__main__":
    main()

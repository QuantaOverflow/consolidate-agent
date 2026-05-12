"""Compute pairwise definition similarity for vocab tags.

Used to surface suspected near-synonym pairs to the diagnostic agent.
Pure deterministic — embedding cache keyed by definition text.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

from dashscope import TextEmbedding


@lru_cache(maxsize=1024)
def _embed(text: str, model: str = "text-embedding-v3") -> tuple[float, ...]:
    """Get embedding via DashScope. Cached per text."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")
    resp = TextEmbedding.call(model=model, input=text, api_key=api_key)
    if resp.status_code != 200:
        raise RuntimeError(f"embedding failed: {resp.code} {resp.message}")
    vec = resp.output["embeddings"][0]["embedding"]
    return tuple(vec)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


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

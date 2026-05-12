"""Multi-dim vocab health metrics for the maintenance agent.

Five orthogonal dimensions, each rescaled to roughly [0, 1] with "higher is
healthier" semantics. Replaces the hit_rate single-metric gate as the visible
health state. Computed once per iter; embedding calls share the global LRU
cache in similarity._embed with compute_fit_signals.

  coverage     — 1 - missing_rate; do tags reach records?
  coherence    — mean cosine(record, tag_def) over (record, selected_tag) pairs
  distinctness — 1 - mean pairwise cosine of tag definitions
  granularity  — normalized Shannon entropy of tag usage distribution
  multi_axis   — fraction of non-missing records with >= 2 tags

Each dim is exposed as a pure function so it is individually testable.
measure_health() ties them together.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .probes import _compute_record_tag_embeddings
from .similarity import _embed, cosine


_DIM_NAMES = ("coverage", "coherence", "distinctness", "granularity", "multi_axis")


@dataclass(frozen=True)
class HealthMetrics:
    coverage: float
    coherence: float
    distinctness: float
    granularity: float
    multi_axis: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}

    def delta(self, prior: "HealthMetrics") -> dict[str, float]:
        """Per-dim (self - prior). Used for iter_deltas history rendering."""
        return {k: round(getattr(self, k) - getattr(prior, k), 4) for k in _DIM_NAMES}


def measure_coverage(assignments: list[dict]) -> float:
    """1 - missing_rate. Empty input -> 0.0."""
    if not assignments:
        return 0.0
    missing = sum(1 for a in assignments if a.get("missing"))
    return 1.0 - missing / len(assignments)


def measure_coherence(
    vocab: list[dict],
    assignments: list[dict],
    db_path: Path,
    *,
    tag_embs: dict | None = None,
    rec_embs: dict | None = None,
    max_records: int = 300,
) -> float:
    """Weighted mean over (record, selected_tag) pairs of cosine(rec, tag_def).

    Naturally weighted by tag usage and multi-tag count. Returns 0.0 when no
    valid pairs exist. Callers may pre-supply tag_embs/rec_embs to skip the
    embedding pass (e.g. when also computing fit_signals in the same iter).
    """
    non_missing = [a for a in assignments if not a.get("missing") and a.get("selected_tags")]
    if not non_missing or not vocab:
        return 0.0
    if tag_embs is None or rec_embs is None:
        tag_embs, rec_embs, _ = _compute_record_tag_embeddings(
            vocab, assignments, db_path, max_records=max_records,
        )

    sims: list[float] = []
    for a in non_missing[:max_records]:
        rec_emb = rec_embs.get(a["record_id"])
        if rec_emb is None:
            continue
        for t in a["selected_tags"]:
            te = tag_embs.get(t["name"])
            if te is not None:
                sims.append(cosine(rec_emb, te))
    if not sims:
        return 0.0
    return sum(sims) / len(sims)


def measure_distinctness(
    vocab: list[dict],
    *,
    tag_embs: dict | None = None,
) -> float:
    """1 - mean pairwise cosine of tag definitions.

    N=0 or N=1: returns 1.0 (no redundancy possible). tag_embs optional for
    embedding reuse.
    """
    if len(vocab) < 2:
        return 1.0
    if tag_embs is None:
        embs = [_embed(t["definition"]) for t in vocab]
    else:
        embs = [tag_embs[t["name"]] for t in vocab if t["name"] in tag_embs]
        if len(embs) < 2:
            return 1.0

    sims: list[float] = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sims.append(cosine(embs[i], embs[j]))
    return 1.0 - sum(sims) / len(sims)


def measure_granularity(assignments: list[dict]) -> float:
    """Normalized Shannon entropy of tag usage distribution.

    Uniform across N used tags -> 1.0; single tag dominant -> 0;
    0 or 1 used tags -> 0.0.
    """
    usage: Counter[str] = Counter()
    for a in assignments:
        if a.get("missing"):
            continue
        for t in a.get("selected_tags", []):
            usage[t["name"]] += 1
    n_used = len(usage)
    if n_used <= 1:
        return 0.0
    total = sum(usage.values())
    entropy = 0.0
    for c in usage.values():
        p = c / total
        entropy -= p * math.log(p)
    return entropy / math.log(n_used)


def measure_multi_axis(assignments: list[dict]) -> float:
    """Fraction of non-missing records with >= 2 selected_tags.

    Aligned with the multi-tag reverse_check policy. Healthy ~0.10-0.30 on
    multi-domain corpora. No non-missing records -> 0.0.
    """
    non_missing = [a for a in assignments if not a.get("missing")]
    if not non_missing:
        return 0.0
    multi = sum(1 for a in non_missing if len(a.get("selected_tags", [])) >= 2)
    return multi / len(non_missing)


def measure_health(
    vocab: list[dict],
    assignments: list[dict],
    db_path: Path,
    *,
    max_records: int = 300,
) -> HealthMetrics:
    """All 5 dims with shared embedding pass.

    Coverage/granularity/multi_axis are pure aggregates; coherence/distinctness
    use embeddings. One concurrent pass via _compute_record_tag_embeddings,
    then both embedding-based dims reuse the same dicts.
    """
    if vocab and assignments:
        tag_embs, rec_embs, _ = _compute_record_tag_embeddings(
            vocab, assignments, db_path, max_records=max_records,
        )
    else:
        tag_embs, rec_embs = {}, {}
    return HealthMetrics(
        coverage=measure_coverage(assignments),
        coherence=measure_coherence(
            vocab, assignments, db_path,
            tag_embs=tag_embs, rec_embs=rec_embs, max_records=max_records,
        ),
        distinctness=measure_distinctness(vocab, tag_embs=tag_embs),
        granularity=measure_granularity(assignments),
        multi_axis=measure_multi_axis(assignments),
    )

"""TagRecordNetwork: canonical state of vocab + record-tag assignments.

Single atomic unit for the tag-record bipartite graph. Avoids the
vocab/assignments drift problem by treating them as one snapshot —
load, mutate via apply, save, always together.

Streaming ingest (handling new batches of records) is a follow-up
abstraction layered on top of this — not included here.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .apply import (
    apply_proposal as _apply_proposal,
    check_invariants,
)


SCHEMA_VERSION = 1


def _vocab_hash(vocab: list[dict]) -> str:
    """Stable 16-char hash of vocab content. Independent of list ordering."""
    canonical = json.dumps(
        sorted(
            [{"name": t["name"], "definition": t["definition"]} for t in vocab],
            key=lambda x: x["name"],
        ),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _filter_dangling_refs(assignments: list[dict], vocab: list[dict]) -> int:
    """In-place: drop tag refs not in vocab. Mark records with all refs dropped as missing.
    Returns count of dropped refs."""
    vocab_names = {t["name"] for t in vocab}
    dropped = 0
    for a in assignments:
        original = a.get("selected_tags", []) or []
        filtered = [t for t in original if t["name"] in vocab_names]
        if len(filtered) < len(original):
            dropped += len(original) - len(filtered)
        a["selected_tags"] = filtered
        if not filtered and not a.get("missing"):
            a["missing"] = True
            a["missing_concept"] = a.get("missing_concept") or "all selected_tags absent from current vocab"
    return dropped


@dataclass
class TagRecordNetwork:
    """Canonical vocab + assignments snapshot.

    All mutations go through `apply()` so vocab and assignments stay in sync.
    `metadata` carries vocab_hash for drift detection across save/load cycles.
    """
    vocab: list[dict]
    assignments: list[dict]
    metadata: dict = field(default_factory=dict)

    # ── Lifecycle ────────────────────────────────────────────────────────

    @classmethod
    def bootstrap(
        cls,
        db_path: Path,
        *,
        batch_size: int = 30,
        concurrency: int = 10,
    ) -> "TagRecordNetwork":
        """Build a fresh network from raw records.

        Phase A: records → distill themes → synthesize vocab → reverse_check
                 → initial assignments → Network.

        Long-running (LLM-heavy, ~5-10 min on ~700 records).
        """
        # Lazy imports to avoid LLM-dep load when only using load/save/apply
        from consolidate_agent.config import Settings
        from consolidate_agent.consolidation._utils import _chat_model

        from .bootstrap import distill_step, synthesize_step
        from .measure import load_records, reverse_check_subset

        settings = Settings()

        def model_factory():
            return _chat_model(settings)

        records = load_records(db_path)
        print(f"[bootstrap] loaded {len(records)} records from {db_path.name}", flush=True)

        log_sink = io.StringIO()  # discard structured log (caller can re-run with log_path if needed)

        # Step 1: per-record themes
        themes = distill_step(model_factory, records, batch_size, log_sink)
        # Step 2: cluster themes into vocab
        syn_result = synthesize_step(model_factory, themes, log_sink)
        vocab = syn_result["vocab"]
        print(f"[bootstrap] vocab: {len(vocab)} tags", flush=True)

        # Step 3: reverse_check on all records → initial assignments
        print(f"[bootstrap] reverse_check on {len(records)} records (concurrency={concurrency})", flush=True)
        assignments = reverse_check_subset(records, vocab, batch_size=10, concurrency=concurrency)

        # Guard: LLM occasionally outputs tag names outside vocab (~3%)
        dropped = _filter_dangling_refs(assignments, vocab)
        if dropped:
            print(f"[bootstrap] filtered {dropped} vocab-external tag refs", flush=True)

        now = time.time()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "vocab_hash": _vocab_hash(vocab),
            "created_at": now,
            "last_modified": now,
            "bootstrap_source": str(db_path),
            "synthesize_notes": syn_result.get("notes", ""),
        }

        network = cls(vocab=vocab, assignments=assignments, metadata=metadata)
        network.validate()
        return network

    @classmethod
    def load(cls, path: Path) -> "TagRecordNetwork":
        """Load network from a single JSON snapshot file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        sv = meta.get("schema_version")
        if sv != SCHEMA_VERSION:
            raise ValueError(
                f"network schema mismatch: file is v{sv}, code is v{SCHEMA_VERSION}. "
                "Re-bootstrap or migrate."
            )
        network = cls(
            vocab=data["vocab"],
            assignments=data["assignments"],
            metadata=meta,
        )
        network.validate()
        return network

    @classmethod
    def from_legacy_files(
        cls,
        vocab_path: Path,
        assignments_path: Path,
    ) -> "TagRecordNetwork":
        """One-time migration helper: load split vocab/assignments files.

        Auto-drops dangling tag refs (legacy assignments may reference tags
        absent from the current vocab — the historic cache-drift case).
        """
        vocab_data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        vocab = vocab_data["vocab"] if isinstance(vocab_data, dict) and "vocab" in vocab_data else vocab_data
        assignments = json.loads(Path(assignments_path).read_text(encoding="utf-8"))

        dropped = _filter_dangling_refs(assignments, vocab)
        if dropped:
            print(f"[from_legacy_files] dropped {dropped} dangling tag refs", flush=True)

        now = time.time()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "vocab_hash": _vocab_hash(vocab),
            "created_at": now,
            "last_modified": now,
            "imported_from": f"{Path(vocab_path).name} + {Path(assignments_path).name}",
            "import_dangling_refs_dropped": dropped,
        }
        network = cls(vocab=vocab, assignments=assignments, metadata=metadata)
        network.validate()
        return network

    def save(self, path: Path) -> None:
        """Atomically save network to a single JSON snapshot file."""
        self.metadata["last_modified"] = time.time()
        self.metadata["vocab_hash"] = _vocab_hash(self.vocab)
        self.metadata.setdefault("schema_version", SCHEMA_VERSION)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "vocab": self.vocab,
                    "assignments": self.assignments,
                    "metadata": self.metadata,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)

    # ── Derived state ────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        """Pure aggregation of assignments → hit_rate, cooccur, usage, etc. No LLM."""
        from .measure import build_diagnostics
        return build_diagnostics(self.vocab, self.assignments)

    def hit_rate(self) -> float:
        d = self.diagnostics()
        n = d.get("sample_size", 0)
        return d.get("total_assigned", 0) / n if n else 0.0

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> None:
        """Coherence checks. Raises on violation."""
        # Vocab uniqueness
        names = [t["name"] for t in self.vocab]
        if len(names) != len(set(names)):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate tag names in vocab: {dups}")

        # Assignments ↔ vocab integrity
        check_invariants(self.vocab, self.assignments)

    # ── Mutation ─────────────────────────────────────────────────────────

    def apply(self, proposal: Any) -> None:
        """Apply a proposal in-place. vocab + assignments updated atomically."""
        new_vocab, new_assignments = _apply_proposal(self.vocab, self.assignments, proposal)
        self.vocab = new_vocab
        self.assignments = new_assignments
        self.metadata["last_modified"] = time.time()
        self.metadata["vocab_hash"] = _vocab_hash(self.vocab)

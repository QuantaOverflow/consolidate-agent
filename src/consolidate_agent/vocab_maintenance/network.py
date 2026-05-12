"""TagRecordNetwork: canonical state of vocab + record-tag assignments.

Single atomic unit for the tag-record bipartite graph. Avoids the
vocab/assignments drift problem by treating them as one snapshot —
load, mutate via apply, save, always together.

Also: `ingest_batch` for streaming new record batches (growth path),
`flush_pending` for explicit pool re-check (migration / cleanup).
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .apply import (
    InvalidProposal,
    NewTagProposal,
    OrphanError,
    apply_proposal as _apply_proposal,
    check_invariants,
)
from .observability import get_default_logger


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
    themes: dict[str, str] = field(default_factory=dict)  # record_id → distilled theme (optional cache)

    # ── Lifecycle ────────────────────────────────────────────────────────

    @classmethod
    def bootstrap(
        cls,
        db_path: Path,
        *,
        batch_size: int = 30,
        concurrency: int = 10,
        on_vocab_review: Callable[[list[dict]], str] | None = None,
        thread_id: str | None = None,
        checkpoint_db: Path | None = None,
    ) -> "TagRecordNetwork":
        """Build a fresh network from raw records via LangGraph state machine.

        Phase A pipeline:
          records → distill themes → synthesize vocab → [HITL gate] → reverse_check → Network

        Args:
            db_path: source SQLite of records.
            batch_size: distill batch size.
            concurrency: ThreadPool workers for reverse_check.
            on_vocab_review: optional callback called after each synthesize attempt.
                Must return "accept" / "regenerate" / "abort". If None, auto-accept
                (unattended mode).
            thread_id: LangGraph thread id for checkpointing. Auto-generated if None.
            checkpoint_db: SQLite path for checkpoints (default outputs/checkpoints.db).
                Persists run state so crashed bootstrap can resume via bootstrap_resume.

        Long-running (LLM-heavy, ~5-10 min on ~700 records).

        Raises:
            ValueError: if user chose "abort" at the review gate.
        """
        from langgraph.types import Command

        from .graphs.bootstrap import build_bootstrap_graph
        from .graphs.checkpointer import sqlite_checkpointer

        if checkpoint_db is None:
            checkpoint_db = Path("outputs/checkpoints.db")
        if thread_id is None:
            thread_id = f"bootstrap-{int(time.time())}"

        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "db_path": str(db_path),
            "batch_size": batch_size,
            "concurrency": concurrency,
            "auto_accept": on_vocab_review is None,
        }

        with sqlite_checkpointer(checkpoint_db) as cp:
            graph = build_bootstrap_graph(cp)
            print(f"[bootstrap] thread_id={thread_id}, checkpoint={checkpoint_db.name}", flush=True)

            result = graph.invoke(initial_state, config)
            while "__interrupt__" in result:
                ipt = result["__interrupt__"][0]
                payload = ipt.value
                if payload.get("stage") == "vocab_review":
                    decision = on_vocab_review(payload["vocab"]) if on_vocab_review else "accept"
                else:
                    decision = "accept"
                result = graph.invoke(Command(resume=decision), config)

            if result.get("abort_reason"):
                raise ValueError(f"bootstrap aborted: {result['abort_reason']}")

            vocab = result["vocab"]
            assignments = result["assignments"]
            # Extract themes from graph state (list[{record_id, title, theme}]) → dict[record_id, str]
            theme_records = result.get("themes", [])
            themes_dict = {t["record_id"]: t["theme"] for t in theme_records if t.get("theme")}

        if result.get("fake_tag_drops", 0):
            print(f"[bootstrap] filtered {result['fake_tag_drops']} vocab-external tag refs", flush=True)

        now = time.time()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "vocab_hash": _vocab_hash(vocab),
            "created_at": now,
            "last_modified": now,
            "bootstrap_source": str(db_path),
            "synthesize_notes": result.get("synthesize_notes", ""),
            "bootstrap_thread_id": thread_id,
            "synthesize_attempts": result.get("synthesize_attempts", 1),
        }
        network = cls(vocab=vocab, assignments=assignments, themes=themes_dict, metadata=metadata)
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
            themes=data.get("themes", {}),  # optional; defaults to {} for older snapshots
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
                    "themes": self.themes,
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

    # ── Streaming ingest ─────────────────────────────────────────────────

    def ingest_batch(
        self,
        new_records: list[dict],
        *,
        db_path: Path | None = None,
        propose_threshold: int = 30,
        concurrency: int = 10,
        reverse_check_fn: Callable | None = None,
        propose_new_fn: Callable | None = None,
        records_fetcher: Callable[[list[str]], list[dict]] | None = None,
    ) -> "IngestResult":
        """Process a batch of new records.

        Flow:
          1. Dedup: skip records whose record_id is already in assignments.
          2. Classify: LLM reverse_check fresh records against current vocab.
          3. Maybe-trigger: if pending pool size ≥ propose_threshold,
             run propose_new + apply new tags + re-check the whole pool.

        Args:
            new_records: list of raw records (need record_id, title, insight).
            db_path: used by default propose_new_fn / records_fetcher.
            propose_threshold: pool size that triggers propose_new (default 30).
            concurrency: LLM ThreadPoolExecutor workers.
            reverse_check_fn / propose_new_fn / records_fetcher: test hooks;
                production uses defaults.

        Returns: IngestResult with all the counts.

        Raises:
            ValueError if vocab is empty (call bootstrap() first).
        """
        if not self.vocab:
            raise ValueError("vocab is empty; run bootstrap() or load() first before ingest_batch")

        # Step 1: dedup
        existing_ids = {a["record_id"] for a in self.assignments}
        fresh = [r for r in new_records if r["record_id"] not in existing_ids]
        skipped = len(new_records) - len(fresh)

        # Early return: nothing fresh to process AND no implicit re-check semantics
        # (use flush_pending() if you want to re-evaluate the pool without new records)
        if not fresh:
            return IngestResult(
                ingested=0,
                skipped_duplicates=skipped,
                newly_assigned=0,
                newly_missing=0,
                new_tags_added=[],
                absorbed_from_pool=0,
                total_pending_after=sum(1 for a in self.assignments if a.get("missing")),
            )

        # Step 2: classify fresh records via LLM
        _reverse_check = reverse_check_fn or _default_reverse_check
        new_assignments = _reverse_check(fresh, self.vocab, concurrency)
        _filter_dangling_refs(new_assignments, self.vocab)

        self.assignments.extend(new_assignments)
        newly_assigned = sum(1 for a in new_assignments if not a.get("missing"))
        newly_missing = sum(1 for a in new_assignments if a.get("missing"))

        # Step 3 & 4: maybe trigger propose_new + re-check pool
        new_tags_added, absorbed = self._maybe_propose_and_recheck(
            propose_threshold=propose_threshold,
            concurrency=concurrency,
            db_path=db_path,
            propose_new_fn=propose_new_fn,
            records_fetcher=records_fetcher,
            reverse_check_fn=reverse_check_fn,
        )

        self.metadata["last_modified"] = time.time()
        self.metadata["vocab_hash"] = _vocab_hash(self.vocab)

        return IngestResult(
            ingested=len(fresh),
            skipped_duplicates=skipped,
            newly_assigned=newly_assigned,
            newly_missing=newly_missing,
            new_tags_added=new_tags_added,
            absorbed_from_pool=absorbed,
            total_pending_after=sum(1 for a in self.assignments if a.get("missing")),
        )

    def flush_pending(
        self,
        *,
        db_path: Path | None = None,
        propose_threshold: int = 30,
        concurrency: int = 10,
        reverse_check_fn: Callable | None = None,
        propose_new_fn: Callable | None = None,
        records_fetcher: Callable[[list[str]], list[dict]] | None = None,
    ) -> "IngestResult":
        """Force a propose_new check on the accumulated pending pool.

        Useful right after `from_legacy_files()` or when pool has grown to
        threshold but no fresh batch has come in to trigger ingest_batch.

        Same logic as ingest_batch's Step 3-4, no fresh records.
        """
        if not self.vocab:
            raise ValueError("vocab is empty; run bootstrap() or load() first before flush_pending")

        new_tags_added, absorbed = self._maybe_propose_and_recheck(
            propose_threshold=propose_threshold,
            concurrency=concurrency,
            db_path=db_path,
            propose_new_fn=propose_new_fn,
            records_fetcher=records_fetcher,
            reverse_check_fn=reverse_check_fn,
        )

        self.metadata["last_modified"] = time.time()
        self.metadata["vocab_hash"] = _vocab_hash(self.vocab)

        return IngestResult(
            ingested=0,
            skipped_duplicates=0,
            newly_assigned=0,
            newly_missing=0,
            new_tags_added=new_tags_added,
            absorbed_from_pool=absorbed,
            total_pending_after=sum(1 for a in self.assignments if a.get("missing")),
        )

    def _maybe_propose_and_recheck(
        self,
        *,
        propose_threshold: int,
        concurrency: int,
        db_path: Path | None,
        propose_new_fn: Callable | None,
        records_fetcher: Callable | None,
        reverse_check_fn: Callable | None,
    ) -> tuple[list[str], int]:
        """If pending pool ≥ threshold, propose new tags, apply, re-check whole pool.

        Returns (new_tag_names, absorbed_from_pool).
        """
        pending = [a for a in self.assignments if a.get("missing")]
        if len(pending) < propose_threshold:
            return [], 0

        # Pass self.themes so propose_new uses cached themes + writes new ones back
        _propose_new = propose_new_fn or _default_propose_new(db_path, themes_cache=self.themes)
        candidates = _propose_new(self.vocab, self.assignments, "")  # focus="" — full pool
        if not candidates:
            return [], 0

        logger = get_default_logger()
        new_tag_names: list[str] = []
        for c in candidates:
            # Tolerate both dict and dataclass shapes
            name = c["name"] if isinstance(c, dict) else c.name
            definition = c["definition"] if isinstance(c, dict) else c.definition
            try:
                self.apply(NewTagProposal(name=name, definition=definition))
                new_tag_names.append(name)
            except (InvalidProposal, OrphanError) as e:
                # Expected: duplicate name, self-merge, orphan-creating deprecate
                print(f"  [ingest] skip new tag '{name}' (expected): {e}", flush=True)
                logger.event("ingest.apply_skipped", tag=name,
                             error_type=type(e).__name__, reason=str(e)[:200])
            # Other exceptions (DB error, OSError, etc.) propagate — caller decides.

        if not new_tag_names:
            return [], 0

        # Re-check the whole pending pool against the expanded vocab
        pending_ids = [a["record_id"] for a in pending]
        _fetch = records_fetcher or _default_records_fetcher(db_path)
        pending_raw = _fetch(pending_ids)
        _reverse_check = reverse_check_fn or _default_reverse_check
        re_checked = _reverse_check(pending_raw, self.vocab, concurrency)
        _filter_dangling_refs(re_checked, self.vocab)

        # Replace pool entries in assignments with re-checked results
        re_by_id = {a["record_id"]: a for a in re_checked}
        self.assignments = [
            re_by_id[a["record_id"]] if a["record_id"] in re_by_id else a
            for a in self.assignments
        ]

        absorbed = sum(1 for a in re_checked if not a.get("missing"))
        return new_tag_names, absorbed


# ── IngestResult ─────────────────────────────────────────────────────────────


@dataclass
class IngestResult:
    """Outcome of an ingest_batch / flush_pending call.

    Invariant: ingested == newly_assigned + newly_missing
    """
    ingested: int                  # fresh records actually processed (post-dedup)
    skipped_duplicates: int        # records skipped — record_id already present
    newly_assigned: int            # fresh records that got tags from existing vocab
    newly_missing: int             # fresh records that went into pending pool
    new_tags_added: list[str]      # tags added by this call (via propose_new)
    absorbed_from_pool: int        # pool records that became assigned after new tags
    total_pending_after: int       # current pool size after this call


# ── Default LLM-backed adapters (test hooks override) ────────────────────────


def _default_reverse_check(records: list[dict], vocab: list[dict], concurrency: int) -> list[dict]:
    from .measure import reverse_check_subset
    return reverse_check_subset(records, vocab, batch_size=10, concurrency=concurrency)


def _default_propose_new(db_path: Path | None, themes_cache: dict[str, str] | None = None) -> Callable:
    """Build a propose_new adapter closed over db_path + optional themes cache."""
    from .propose.new import propose_new_fn as _propose_new

    def adapter(vocab, assignments, focus):
        return _propose_new(vocab, assignments, focus, db_path=db_path, themes_cache=themes_cache)

    return adapter


def _default_records_fetcher(db_path: Path | None) -> Callable[[list[str]], list[dict]]:
    """Build a record fetcher that loads raw records from db by record_id."""
    if db_path is None:
        raise ValueError("db_path required when records_fetcher not provided")

    def fetch(record_ids: list[str]) -> list[dict]:
        from .measure import load_records
        all_records = load_records(db_path)
        by_id = {r["record_id"]: r for r in all_records}
        return [by_id[rid] for rid in record_ids if rid in by_id]

    return fetch

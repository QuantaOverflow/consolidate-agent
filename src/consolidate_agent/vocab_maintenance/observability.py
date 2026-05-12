"""Structured run logging — appendonly JSONL audit trail.

A RunLogger writes one JSON event per line to a file. Each event has a
timestamp + op name + arbitrary fields. Designed for post-hoc forensics
of agent runs (especially LLM-heavy ones).

Usage:
    from .observability import RunLogger, set_default_logger

    logger = RunLogger(Path("outputs/runs/2026-05-12_bootstrap.jsonl"))
    set_default_logger(logger)
    # ... downstream code calls get_default_logger().event(...)
    logger.close()
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


class RunLogger:
    """Thread-safe JSONL event sink.

    If `path` is None, all events are silently dropped (no-op logger).
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._fp = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(self.path, "a", encoding="utf-8")

    def event(self, op: str, **fields: Any) -> None:
        """Emit one event line: {ts, op, **fields}."""
        if self._fp is None:
            return
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "op": op,
            **fields,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            self._fp.write(line + "\n")
            self._fp.flush()

    @contextmanager
    def timed(self, op: str, **fields: Any):
        """Emits {op}.start + {op}.done with elapsed_s. On exception emits {op}.error and re-raises."""
        t0 = time.perf_counter()
        self.event(f"{op}.start", **fields)
        try:
            yield
        except Exception as e:
            self.event(
                f"{op}.error",
                elapsed_s=round(time.perf_counter() - t0, 2),
                error_type=type(e).__name__,
                error=str(e)[:500],
                **fields,
            )
            raise
        else:
            self.event(f"{op}.done", elapsed_s=round(time.perf_counter() - t0, 2), **fields)

    def close(self) -> None:
        with self._lock:
            if self._fp:
                self._fp.close()
                self._fp = None


# ── Module-level default logger ──────────────────────────────────────────────


_default_logger: RunLogger = RunLogger(path=None)


def set_default_logger(logger: RunLogger | None) -> None:
    """Set the process-wide default logger. Pass None to disable."""
    global _default_logger
    _default_logger = logger or RunLogger(path=None)


def get_default_logger() -> RunLogger:
    """Returns the current default logger (no-op if none configured)."""
    return _default_logger


# ── LLM invoke wrapper ───────────────────────────────────────────────────────


def invoke_with_retry(
    model: Any,
    msg: Any,
    *,
    retries: int = 3,
    caller: str = "unknown",
    logger: RunLogger | None = None,
) -> Any | None:
    """Invoke a structured-output model with retry on ValidationError / None.

    Logs each attempt + outcome. Returns the parsed object on success,
    or None if all retries exhausted.

    Callers should handle None (fall back, raise, or escalate).
    """
    from pydantic import ValidationError

    log = logger or get_default_logger()
    last_err: Any = None
    for attempt in range(1, retries + 1):
        t0 = time.perf_counter()
        try:
            result = model.invoke(msg)
        except ValidationError as e:
            elapsed = time.perf_counter() - t0
            last_err = e
            log.event(
                "llm.validation_error",
                caller=caller,
                attempt=attempt,
                retries=retries,
                elapsed_s=round(elapsed, 2),
                error=str(e)[:400],
            )
            continue
        except Exception as e:  # noqa: BLE001 — log unexpected LLM errors
            elapsed = time.perf_counter() - t0
            last_err = e
            log.event(
                "llm.unexpected_error",
                caller=caller,
                attempt=attempt,
                retries=retries,
                elapsed_s=round(elapsed, 2),
                error_type=type(e).__name__,
                error=str(e)[:400],
            )
            continue

        elapsed = time.perf_counter() - t0
        if result is None:
            last_err = "returned None"
            log.event(
                "llm.returned_none",
                caller=caller,
                attempt=attempt,
                retries=retries,
                elapsed_s=round(elapsed, 2),
            )
            continue

        log.event(
            "llm.success",
            caller=caller,
            attempt=attempt,
            elapsed_s=round(elapsed, 2),
        )
        return result

    log.event(
        "llm.retries_exhausted",
        caller=caller,
        retries=retries,
        last_error=str(last_err)[:400],
    )
    return None

"""Checkpointer factories.

SqliteSaver for production (persistent, cross-process resume).
MemorySaver for tests (no I/O).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def sqlite_checkpointer(db_path: Path):
    """SqliteSaver context manager. Ensures parent dir exists.

    Usage:
        with sqlite_checkpointer(Path("outputs/checkpoints.db")) as cp:
            graph = build_some_graph(cp)
            graph.invoke(state, config)
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(db_path)) as cp:
        cp.setup()
        yield cp


def memory_checkpointer() -> MemorySaver:
    """In-memory checkpointer for tests."""
    return MemorySaver()

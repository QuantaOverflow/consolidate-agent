"""RunLogger unit tests — no LLM, no real I/O beyond tmpfile."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import traceback
from pathlib import Path

from consolidate_agent.vocab_maintenance.observability import (
    RunLogger,
    get_default_logger,
    invoke_with_retry,
    set_default_logger,
)


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@test
def test_o1_event_writes_jsonl():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "log.jsonl"
        logger = RunLogger(p)
        logger.event("test.op", value=42, name="hello")
        logger.close()
        events = _read_jsonl(p)
        assert len(events) == 1
        assert events[0]["op"] == "test.op"
        assert events[0]["value"] == 42
        assert events[0]["name"] == "hello"
        assert "ts" in events[0]


@test
def test_o2_noop_when_path_none():
    logger = RunLogger(path=None)
    logger.event("test.op", x=1)  # must not raise
    logger.close()


@test
def test_o3_timed_context():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "log.jsonl"
        logger = RunLogger(p)
        with logger.timed("phase", phase_name="A"):
            pass
        logger.close()
        events = _read_jsonl(p)
        assert len(events) == 2
        assert events[0]["op"] == "phase.start"
        assert events[1]["op"] == "phase.done"
        assert "elapsed_s" in events[1]


@test
def test_o4_timed_captures_exception():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "log.jsonl"
        logger = RunLogger(p)
        try:
            with logger.timed("phase"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        logger.close()
        events = _read_jsonl(p)
        assert events[1]["op"] == "phase.error"
        assert events[1]["error_type"] == "RuntimeError"
        assert "boom" in events[1]["error"]


@test
def test_o5_thread_safety():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "log.jsonl"
        logger = RunLogger(p)
        N = 50

        def worker(idx):
            for i in range(10):
                logger.event("t", thread=idx, i=i)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        logger.close()

        events = _read_jsonl(p)
        assert len(events) == N * 10, f"expected {N*10}, got {len(events)}"


@test
def test_o6_default_logger_set_get():
    set_default_logger(None)
    assert isinstance(get_default_logger(), RunLogger)
    # Default logger has no path → events are no-ops
    get_default_logger().event("test", x=1)


@test
def test_o7_invoke_with_retry_success_first_try():
    class FakeModel:
        def __init__(self):
            self.calls = 0
        def invoke(self, msg):
            self.calls += 1
            return {"result": "ok"}
    model = FakeModel()
    out = invoke_with_retry(model, msg=None, retries=3, caller="test")
    assert out == {"result": "ok"}
    assert model.calls == 1


@test
def test_o8_invoke_with_retry_validation_error_then_succeeds():
    from pydantic import BaseModel, Field, ValidationError

    class Foo(BaseModel):
        x: int = Field(ge=0)

    class FakeModel:
        def __init__(self):
            self.calls = 0
        def invoke(self, msg):
            self.calls += 1
            if self.calls < 3:
                # Manually raise ValidationError as the SDK would
                try:
                    Foo(x=-1)
                except ValidationError as e:
                    raise e
            return Foo(x=42)

    model = FakeModel()
    out = invoke_with_retry(model, msg=None, retries=3, caller="test")
    assert out is not None
    assert out.x == 42
    assert model.calls == 3


@test
def test_o9_invoke_with_retry_exhausted_returns_none():
    from pydantic import BaseModel, Field, ValidationError

    class Foo(BaseModel):
        x: int = Field(ge=0)

    class FakeModel:
        def invoke(self, msg):
            try:
                Foo(x=-1)
            except ValidationError as e:
                raise e

    out = invoke_with_retry(FakeModel(), msg=None, retries=3, caller="test")
    assert out is None


@test
def test_o10_invoke_with_retry_returns_none_on_none_result():
    class FakeModel:
        def invoke(self, msg):
            return None

    out = invoke_with_retry(FakeModel(), msg=None, retries=2, caller="test")
    assert out is None


def main() -> int:
    passed = 0
    failed = 0
    print(f"\nRunning {len(TESTS)} observability tests…\n")
    print("─" * 50)
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}")
            print(f"     {type(e).__name__}: {e}")
            tb_lines = traceback.format_exc().split("\n")
            for line in tb_lines[-6:-1]:
                if line.strip():
                    print(f"     {line}")
            failed += 1

    print(f"\n{'─' * 50}")
    print(f"Total: {passed + failed}, Passed: {passed}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

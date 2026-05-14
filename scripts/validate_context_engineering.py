from __future__ import annotations

import argparse
from pathlib import Path

from consolidate_agent.context import SessionContextEngineer


def iter_session_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        root.rglob("*.jsonl"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Codex session context engineering output.")
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path.home() / ".codex" / "sessions",
        help="Directory containing Codex session JSONL files.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of parent sessions to process.")
    args = parser.parse_args()

    engineer = SessionContextEngineer(sessions_dir=args.sessions_dir)
    processed = []
    for path in iter_session_files(args.sessions_dir):
        session = engineer.process(path)
        if session is None or session.is_sub_agent:
            continue
        processed.append((path, session))
        if len(processed) >= args.limit:
            break

    if len(processed) < args.limit:
        print(f"Only found {len(processed)} processable parent sessions under {args.sessions_dir}")
        return 1

    ratios = []
    for index, (path, session) in enumerate(processed, start=1):
        ratio = session.stats.compression_ratio
        ratios.append(ratio)
        print("=" * 100)
        print(f"Session {index}: {path}")
        print(f"id: {session.session_id}")
        print(f"raw_chars: {session.stats.raw_chars}")
        print(f"processed_chars: {session.stats.processed_chars}")
        print(f"compression_ratio: {ratio:.2%}")
        print("-" * 100)
        print(session.xml)

    average = sum(ratios) / len(ratios)
    print("=" * 100)
    print(f"average_compression_ratio: {average:.2%}")
    if average <= 0.70:
        print("Average compression ratio must be > 70%.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

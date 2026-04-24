from __future__ import annotations

import json
from pathlib import Path

from consolidate_agent.types import ProcessedIndex, utc_now


def read_processed_index(path: Path) -> ProcessedIndex:
    if not path.exists():
        return ProcessedIndex()
    return ProcessedIndex.model_validate(json.loads(path.read_text(encoding="utf-8")))


def write_processed_index(path: Path, processed_index: ProcessedIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    processed_index.updated_at = utc_now()
    payload = processed_index.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    return files("consolidate_agent.prompts").joinpath(name).read_text(encoding="utf-8").strip()

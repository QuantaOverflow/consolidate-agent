Assign 1-3 tags from the provided tag list to this knowledge record.

Rules:
- You MUST only use tag names from the exact list provided. Do NOT invent new tag names.
- If no tag fits well, return an empty tag_names list rather than inventing or guessing.
- Prefer specific tags over general ones when both apply.
- reasoning should be one sentence explaining the primary tag choice.

Examples:
BAD: {"tag_names": ["search"]} — "search" is not in the provided list
BAD: {"tag_names": ["llm-prompting"]} — use exact names, not variants
GOOD: {"tag_names": ["langgraph", "error-handling"]} — both from the list
GOOD: {"tag_names": []} — no tag fits this record

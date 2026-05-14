#!/usr/bin/env python3
"""
通用知识提取 POC 脚本。
以 session 为单位，用 context engineering 后的 XML 作为输入，单次 structured LLM call 提取可迁移认知。
"""
from __future__ import annotations

import re
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen

from consolidate_agent.config import Settings
from consolidate_agent.context.engineer import SessionContextEngineer, ProcessedSession

# qwen-plus 128k token 上限，保守估计中英混合约 2 char/token，留 prompt 开销
MAX_SESSION_CHARS = 100_000


class KnowledgeItem(BaseModel):
    title: str = Field(min_length=3)
    insight: str = Field(min_length=10, description="核心认知，可迁移的那部分")
    applicability: str = Field(min_length=10, description="什么情况下有用")
    evidence_turns: list[int] = Field(default_factory=list, description="来源 turn index 列表")
    scope: str = Field(description="global / project_specific / session_specific")


class KnowledgeExtractionOutput(BaseModel):
    items: list[KnowledgeItem] = Field(default_factory=list)


def compute_evidence_score(item: KnowledgeItem, session: ProcessedSession) -> dict:
    """客观计算 evidence 质量指标。"""
    total_turns = len(re.findall(r"<turn ", session.xml))
    count = len(item.evidence_turns)
    if count >= 2 and total_turns > 1:
        spread = (max(item.evidence_turns) - min(item.evidence_turns)) / (total_turns - 1)
    else:
        spread = 0.0
    return {
        "evidence_count": count,
        "evidence_spread": round(spread, 2),
        "turn_count": total_turns,
    }


SYSTEM_PROMPT = """\
You extract meaningful, transferable knowledge from a Codex session.

The session is provided as structured XML where each <turn> contains one user request \
and the assistant's response with tool calls and results.

Knowledge is any insight, understanding, decision rationale, or analytical approach \
that was reached or validated during this session and could be useful in a different context.

This includes:
- Technical insights about tools, frameworks, or APIs
- Decision frameworks or analytical approaches used
- Domain understanding that took effort to reach
- Debugging or diagnostic reasoning that proved effective
- Architectural choices and their trade-offs

Do NOT extract:
- One-off task-specific actions with no generalization value
- Trivial observations obvious to any practitioner
- Session-specific context that cannot transfer elsewhere

For each knowledge item provide:
- title: concise label
- insight: the actual transferable cognition (what was learned/understood)
- applicability: when and where this insight applies
- evidence_turns: list of turn index numbers (integers) that support this knowledge
- scope: one of the following — be strict:
  - "global": applies regardless of language, framework, or domain; any developer on any project benefits
  - "project_specific": only useful for projects using the same tech stack, domain, or architectural pattern (e.g., "Playwright automation", "LangGraph agents", "A-share financial data")
  - "session_specific": a one-time workaround or decision tied to this exact codebase; do NOT extract these

Scope decision rule — ask these questions in order:
1. Does this insight reference a specific file, variable name, or project-internal concept? → session_specific (skip)
2. Does this insight only apply when using a specific tool/library/domain (e.g., Playwright, LangGraph, financial data APIs)? → project_specific
3. Is this a general engineering/design principle that any developer could apply regardless of tech stack? → global

Examples of global: "tool wrappers should return structured errors instead of raising exceptions", "validate LLM output at both per-item and aggregation layers", "decouple UI lifecycle hooks from core business logic"
Examples of project_specific: "in Playwright, CDP calls should silently catch exceptions", "LangGraph subgraph events need thread-scope correction", "AkShare net_flow field uses principal capital estimation"
Examples of session_specific (do NOT extract): "replace token in config.yaml", "add retry for this specific endpoint"

Return 2-6 items. Prefer fewer high-quality items over many mediocre ones.
"""

USER_PROMPT = """\
Session:
{{ session_xml }}

Extract transferable knowledge from this session.
"""


def build_extractor(settings: Settings):
    model = ChatQwen(
        model=settings.qwen_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_api_base,
        temperature=0,
    )
    structured = model.with_structured_output(KnowledgeExtractionOutput)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)],
        template_format="jinja2",
    )
    return structured, prompt


def extract_from_session(session: ProcessedSession, structured_model, prompt) -> KnowledgeExtractionOutput:
    prompt_value = prompt.invoke({"session_xml": session.xml})
    result = structured_model.invoke(prompt_value)
    return result or KnowledgeExtractionOutput()


def main():
    settings = Settings()
    if not settings.dashscope_api_key:
        print("ERROR: DASHSCOPE_API_KEY not set")
        sys.exit(1)

    sessions_dir = Path("~/.codex/sessions").expanduser()
    all_sessions = sorted(sessions_dir.rglob("*.jsonl"))

    random.seed(42)
    random.shuffle(all_sessions)

    engineer = SessionContextEngineer()
    structured_model, prompt = build_extractor(settings)

    processed = []
    for path in all_sessions:
        if len(processed) >= 5:
            break
        r = engineer.process(path)
        if r is None or r.is_sub_agent:
            continue
        if r.stats.processed_chars < 500:
            continue
        if r.stats.processed_chars > MAX_SESSION_CHARS:
            continue
        processed.append((path, r))

    print(f"处理 {len(processed)} 个 session (≤{MAX_SESSION_CHARS:,} chars)\n{'='*60}")

    for path, session in processed:
        print(f"\n[Session] {path.name}")
        print(f"  processed_chars={session.stats.processed_chars:,}  "
              f"compression={session.stats.compression_ratio:.1%}  "
              f"cwd={session.cwd or 'N/A'}")

        result = extract_from_session(session, structured_model, prompt)

        if not result.items:
            print("  → 未提取到知识")
            continue

        print(f"  → 提取到 {len(result.items)} 条知识:\n")
        for i, item in enumerate(result.items, 1):
            ev = compute_evidence_score(item, session)
            print(f"  [{i}] {item.title}  (scope={item.scope})")
            print(f"      evidence: turns={item.evidence_turns}  "
                  f"spread={ev['evidence_spread']:.2f}  total_turns={ev['turn_count']}")
            print(f"      insight:       {item.insight[:120]}")
            print(f"      applicability: {item.applicability[:100]}")
            print()

    print("="*60)
    print("完成")


if __name__ == "__main__":
    main()
